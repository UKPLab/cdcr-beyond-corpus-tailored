import itertools
import pprint
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.utils import check_random_state
from tabulate import tabulate

from python import TOPIC_ID, SUBTOPIC, DOCUMENT_ID, MENTION_ID, EVENT, EVENT_ID

NUM_PAIRS = "num-pairs"


class MentionPairGenerator:
    """
    Advanced generator of mention pairs for training, evaluation during hyperoptimization and final evaluation.
    """

    def __init__(self,
                 undersample_c: Optional[float] = None,
                 neg_to_pos_pair_ratio: Optional[float] = None,
                 serialization_dir: Optional[Path] = None):
        """
        Note: all of these parameters are only for the oracle-style generation.
        :param undersample_c: By default, this method generates n-choose-2 for each cluster. This can lead to large
                              clusters dominating the returned instances because they yield many more mention pairs than
                              smaller clusters which may cause bad performance because the variation in events in the
                              training pairs is low. With this parameter, undersampling can be controlled. See
                              `undersample_events` for details. Set to None to disable undersampling.
        :param neg_to_pos_pair_ratio: Ratio from positive to negative pairs to return. This option is applied after the
                                      cluster undersampling, and separately for each type of link (cross-topic, ...). It
                                      is an upper bound - the returned amount of negative pairs may be lower because the
                                      given dataset does not contain enough of them. If None, all negative pairs are
                                      returned. In case a dataset does not contain positive pairs of a type, this ratio
                                      is used to sample negative pairs for any missing higher-level types. Example:
                                      ratio set to 2, no positive cross-subtopic or cross-topic pairs exist, but we
                                      already decided to sample 50 negative within-subtopic pairs. This method wil then
                                      sample 2*50 negative cross-subtopic and 2*2*50 negative cross-topic pairs.
        :param serialization_dir: optional directory for storing debug information on pair sampling
        """
        self.undersample_c = undersample_c
        self.neg_to_pos_pair_ratio = neg_to_pos_pair_ratio
        self.serialization_dir = serialization_dir


    def _generate_oracle(self,
                         documents: pd.DataFrame,
                         mentions: pd.DataFrame,
                         random_state: Optional[Union[None, int, RandomState]]) -> Tuple[List, List]:
        assert EVENT in mentions
        mentions = mentions[EVENT]

        random_state = check_random_state(random_state)

        # --------------------------------------------------------------------------------------------------------------

        # We want to create an adjacency matrix for mentions. In order to sample a different number of pairs for
        # cross-topic, cross-subtopic, ... pairs, we need a way to represent the type of each mention pair (== each
        # possible correct or incorrect coreference link) in the matrix. To do that, we create a separate masking
        # variant for the matrix which indicates the type of each mention.

        # retrieve topic/subtopic information for each mention
        mentions_with_topic_info = documents.droplevel(DOCUMENT_ID).reset_index().merge(mentions.reset_index(), on=DOCUMENT_ID)

        # For the next operations, we need to make sure that subtopics and document IDs are globally unique in the
        # dataframe. Topics will be unique already since they are the top-level index. Concat each identifier as strings
        # and factorize them to get an int representation for each unique value.
        uniquefy = lambda columns: (columns[0].astype(str) + columns[1].astype(str)).factorize()[0]
        mentions_with_topic_info_mat_indexable = mentions_with_topic_info.copy()
        mentions_with_topic_info_mat_indexable[SUBTOPIC] = uniquefy(
            (mentions_with_topic_info_mat_indexable[TOPIC_ID], mentions_with_topic_info_mat_indexable[SUBTOPIC]))
        mentions_with_topic_info_mat_indexable[DOCUMENT_ID] = uniquefy(
            (mentions_with_topic_info_mat_indexable[SUBTOPIC], mentions_with_topic_info_mat_indexable[DOCUMENT_ID]))

        # These are the identifiers we use for each link type in the matrix
        _cross_topic = -1
        _cross_subtopic = -2
        _within_subtopic = -3
        _within_document = -4
        link_types_in_order = [_within_document, _within_subtopic, _cross_subtopic, _cross_topic]

        # Determine corpus properties (we need this for negative pair sampling later)
        num_mentions = len(mentions)
        num_documents = len(mentions_with_topic_info_mat_indexable[DOCUMENT_ID].unique())
        num_subtopics = len(mentions_with_topic_info_mat_indexable[SUBTOPIC].unique())
        num_topics = len(mentions_with_topic_info_mat_indexable[TOPIC_ID].unique())

        # We start by assuming that all pairs are WD pairs. We then find within-subtopic pairs and paste those onto the
        # matrix, then cross-subtopic, and so on.
        mat_link_types = np.full((num_mentions, num_mentions), fill_value=_within_document, dtype=np.int8)
        for link_type, identifier in [(DOCUMENT_ID, _within_subtopic), (SUBTOPIC, _cross_subtopic),
                                      (TOPIC_ID, _cross_topic)]:
            # Compare vector with transposed version of itself to get pairwise comparison
            col_vector = mentions_with_topic_info_mat_indexable[link_type].values
            mat_is_link_type = (col_vector.reshape((-1, 1)) != col_vector.reshape((1, -1))).astype(np.int8)
            mat_link_types = np.where(mat_is_link_type, mat_is_link_type * identifier, mat_link_types)

        # The coreference relation is symmetric, therefore an upper tri matrix is sufficient. Replace lower tri and diag
        # with zeros.
        mat_upper_tri = ~(np.tri(num_mentions).astype(bool))
        mat_link_types *= mat_upper_tri

        # --------------------------------------------------------------------------------------------------------------

        # For each cluster, determine number of positive pairs to sample
        mentions_per_event = mentions.value_counts()
        mentions_per_event = mentions_per_event.loc[mentions_per_event > 1]  # drop singletons
        if self.undersample_c is not None:
            pos_pairs_per_event = undersample_events(mentions_per_event, c=self.undersample_c)
        else:
            pos_pairs_per_event = mentions_per_event * (mentions_per_event - 1) // 2
        pos_pairs_per_event = pos_pairs_per_event.to_frame(NUM_PAIRS)

        # --------------------------------------------------------------------------------------------------------------

        # Now create two additional adjacency matrices which contains the positive pairs. Cells in the matrix are
        # either 0 if they are negative pairs, or they contain the cluster ID if the corresponding link/pair belongs to
        # that cluster. One matrix contains all positive pairs (this is important for identifying negative pairs) and
        # the other contains only the sampled positive pairs (this is important for identifying the correct number of
        # negative pairs to sample later on). We also collect the sampled positive pairs in a list-based format to use
        # for the model in the end.

        def pair_of_indices_to_pair_of_mention_identifiers(indices: np.array) -> List[Tuple[Tuple, Tuple]]:
            """
            Given a np.array of pairs indexing the mention adjacency matrix, convert those matrix indices back into
            pairs of mention identifiers, i.e. into one (doc_id, mention_id) tuple per mention.
            :param indices:
            :return:
            """
            assert indices.shape[1] == 2

            pairs_as_tuples = []
            doc_ids_and_mention_ids = mentions_with_topic_info[[DOCUMENT_ID, MENTION_ID]]
            a_mentions = doc_ids_and_mention_ids.take(indices[:, 0]).add_prefix("a-").reset_index(drop=True)
            b_mentions = doc_ids_and_mention_ids.take(indices[:, 1]).add_prefix("b-").reset_index(drop=True)
            sampled_pos_pairs_df = pd.concat([a_mentions, b_mentions], axis=1)
            for _, a_doc_id, a_mention_id, b_doc_id, b_mention_id in sampled_pos_pairs_df.itertuples():
                pair = ((a_doc_id, a_mention_id), (b_doc_id, b_mention_id))
                pairs_as_tuples.append(pair)
            return pairs_as_tuples

        mat_pos_pairs = np.zeros_like(mat_link_types)
        mat_pos_pairs_sampled = np.zeros_like(mat_link_types)
        all_pos_pairs = []
        # Assign cluster ID for each event. Start with 1 to reserve 0 as a tool for matrix masking operations.
        pos_pairs_per_event[EVENT_ID] = range(1, 1 + len(pos_pairs_per_event))
        for event, mentions_of_event in mentions_with_topic_info_mat_indexable.groupby(EVENT, as_index=False):
            # Skip singletons, we cannot create pairs for those
            if len(mentions_of_event) == 1:
                continue

            # Get all pairwise links in this cluster in the form of two lists of mention matrix indices, so we can index
            # with numpy easily
            pos_pairs_of_cluster = np.array(list(zip(*itertools.combinations(mentions_of_event.index.values, 2))))

            # In the adjacency matrix, set the matrix cell of each link to the event's cluster ID
            event_id = pos_pairs_per_event.at[event, EVENT_ID]
            mat_pos_pairs[pos_pairs_of_cluster[0], pos_pairs_of_cluster[1]] = event_id

            # Sample positive pairs: we need to transpose the pairs because permutation only works on the first axis
            num_samples = pos_pairs_per_event.at[event, NUM_PAIRS]
            sampled_pos_pair_indices = random_state.permutation(pos_pairs_of_cluster.transpose())[:num_samples]

            # collect them as a list of pairs of tuples --> [((doc_id_a, mention_id_a), (doc_id_b, mention_id_b)), ...]
            all_pos_pairs += pair_of_indices_to_pair_of_mention_identifiers(sampled_pos_pair_indices)

            # Transpose back and track sampled pairs in the corresponding matrix
            sampled_pos_pair_indices = sampled_pos_pair_indices.transpose()
            mat_pos_pairs_sampled[sampled_pos_pair_indices[0], sampled_pos_pair_indices[1]] = event_id

        # Track the types of the sampled links
        num_pos_pairs_sampled_per_type = {}
        for link_type in link_types_in_order:
            num_pos_pairs_sampled_of_type = np.sum((mat_pos_pairs_sampled > 0) & (mat_link_types == link_type))
            num_pos_pairs_sampled_per_type[link_type] = num_pos_pairs_sampled_of_type

        # --------------------------------------------------------------------------------------------------------------

        # Sample negative pairs for each pair type. If no ratio is specified, all negative pairs will be returned.
        # Also, track the number of negative pairs sampled per type.
        # ################## See neg_pair_sampling.tex for explanation ##################
        num_neg_pairs_sampled_per_type = {}
        all_neg_pairs = []
        num_objects_corpus_structure = [num_mentions, num_documents, num_subtopics, num_topics, 1]
        for i, link_type in enumerate(link_types_in_order):
            # Negative pair candidates are those cells in the adjacency matrix which are of the pair type in question
            # and which are no positive pairs (0-valued in mat_pos_pairs).
            all_neg_pairs_of_type = np.argwhere((mat_link_types == link_type) & (mat_pos_pairs == 0))

            if self.neg_to_pos_pair_ratio is None:
                # without a ratio, we will sample all negative pairs
                sampled_neg_pair_indices = all_neg_pairs_of_type
            else:
                # Compute number of negative pairs for this type based on the number of positive types or the corpus
                # structure, depending on what demands more pairs. Based on number of positive pairs:
                based_on_pos_pairs = int(self.neg_to_pos_pair_ratio * num_pos_pairs_sampled_per_type[link_type])

                # Based on the corpus structure:
                num_objects = num_objects_corpus_structure[i+1]
                num_objects_lower_level = num_objects_corpus_structure[i]
                based_on_corpus_structure = int(0.5 * (num_objects_lower_level/num_objects - 1) * (num_mentions/num_objects_lower_level)**2)

                # correct number of negative pairs to sample based on how many we actually have
                num_neg_pairs_to_sample = min(max(based_on_pos_pairs, based_on_corpus_structure), len(all_neg_pairs_of_type))
                sampled_neg_pair_indices = random_state.permutation(all_neg_pairs_of_type)[:num_neg_pairs_to_sample]

            all_neg_pairs += pair_of_indices_to_pair_of_mention_identifiers(sampled_neg_pair_indices)

            # track number of sampled pairs
            num_neg_pairs_sampled_per_type[link_type] = sampled_neg_pair_indices.shape[0]

        # --------------------------------------------------------------------------------------------------------------

        # Consolidate and create labels and we're done
        pairs = all_pos_pairs + all_neg_pairs
        labels = [True] * len(all_pos_pairs) + [False] * len(all_neg_pairs)

        # --------------------------------------------------------------------------------------------------------------

        # Write debug information if desired
        if self.serialization_dir is not None:
            self.serialization_dir.mkdir(exist_ok=True, parents=True)

            link_type_rename_dict = {_within_document: "wd",
                                     _within_subtopic: "ws",
                                     _cross_subtopic: "cs",
                                     _cross_topic: "ct"}

            # collect information on total number of pairs
            num_pos_pairs_per_type_total = {}
            num_neg_pairs_per_type_total = {}
            for link_type in link_types_in_order:
                num_pos_pairs_per_type_total[link_type] = np.sum((mat_link_types == link_type) & (mat_pos_pairs > 0))
                num_neg_pairs_per_type_total[link_type] = np.sum((mat_link_types == link_type) & (mat_pos_pairs == 0))
            num_pairs_per_type_total = pd.DataFrame([pd.Series(num_pos_pairs_per_type_total, name="positive"),
                                                     pd.Series(num_neg_pairs_per_type_total, name="negative")]).rename(columns=link_type_rename_dict)
            num_pairs_per_type_total.to_pickle(str(self.serialization_dir / "num_pairs_per_type_total.pkl"))

            # collect information on number of sampled pairs
            num_pairs_sampled_per_type = pd.DataFrame([pd.Series(num_pos_pairs_sampled_per_type, name="positive"),
                                                       pd.Series(num_neg_pairs_sampled_per_type, name="negative")]).rename(columns=link_type_rename_dict)
            num_pairs_sampled_per_type.to_pickle(str(self.serialization_dir / "num_pairs_sampled_per_type.pkl"))
            pairs_sampled_by_type_relative = num_pairs_sampled_per_type / num_pairs_sampled_per_type.values.sum()

            # print all that
            with (self.serialization_dir / "pairs_stats.txt").open("w") as f:
                f.write(f"""MPG debug info
--------------

object __dict__:
{pprint.pformat(self.__dict__)}

total pairs available:
{tabulate(num_pairs_per_type_total, headers="keys")}

pairs generated absolute:
{tabulate(num_pairs_sampled_per_type, headers="keys")}

pairs generated relative:
{tabulate(pairs_sampled_by_type_relative, headers="keys")}""")

            # plot adjacency matrices
            from matplotlib import pyplot as plt
            for name, mat, cmap, vmin in [("link_types", mat_link_types, "gray", None),
                                          ("pos_pairs", mat_pos_pairs, "rainbow", 1),
                                          ("pos_pairs_sampled", mat_pos_pairs_sampled, "rainbow", 1)]:
                fig, ax = plt.subplots()
                cmap = plt.get_cmap(cmap)
                cmap.set_under("black")
                ax.imshow(mat, cmap=cmap, vmin=vmin)

                # draw gridlines for small numbers of mentions
                if num_mentions < 50:
                    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.25)
                    ticks = np.arange(-0.5, num_mentions + 0.5, 1)
                    ax.set_xticks(ticks)
                    ax.set_yticks(ticks)

                fig.savefig(self.serialization_dir / f"{name}.png", dpi=300)
                plt.clf()
            plt.close('all')

        return pairs, labels

    @staticmethod
    def _generate_blind(mentions: pd.DataFrame,
                        mentions_to_gold_events: Optional[pd.Series]) -> Tuple[List, List]:
        pairs = list(itertools.combinations(mentions.index.values, 2))
        if mentions_to_gold_events is not None:
            # if the true labels are given, create the list of gold pair labels alongside
            pairs_of_event_names = itertools.combinations(mentions_to_gold_events.values, 2)
            labels = list(map(lambda pair: pair[0] == pair[1], pairs_of_event_names))
        else:
            labels = None
        return pairs, labels

    def generate(self,
                 documents: pd.DataFrame,
                 mentions: pd.DataFrame,
                 mentions_to_gold_events: Optional[pd.Series] = None,
                 random_state: Optional[Union[None, int, RandomState]] = None) -> Tuple[List, List]:
        if EVENT in mentions.columns:
            return self._generate_oracle(documents, mentions, random_state)
        else:
            return self._generate_blind(mentions, mentions_to_gold_events)


def undersample_events(events_to_num_mentions: pd.Series, c: float = 1.75) -> pd.Series:
    """
    Given the number of mentions per event in a data split, returns the number of positive pairs to sample from each
    cluster. For the smallest clusters, this method will recommend to sample all n*(n-1)/2 pairs. For the largest
    cluster, the number of pairs to sample decreases to (n-1)*c. Clusters in between smoothly transition from a
    quadratic to a linear number of pairs, based on the distribution of cluster sizes in the data.

    Notes on when it makes sense to apply undersampling:

    | task                  | undersample train splits | undersample evaluation split                        | reason                                                         |
    |-----------------------|--------------------------|-----------------------------------------------------|----------------------------------------------------------------|
    | feature selection     | yes                      | yes                                                 | Avoid bias from large clusters during training and evaluation. |
    | hyperopt classifier   | yes                      | yes                                                 | Avoid bias from large clusters during training and evaluation. |
    | hyperopt clustering   | yes                      | no                                                  | Avoid bias from large clusters during training. Evaluation is mention-based (not pair-based), therefore undersampling/removing certain pairs makes no sense. |
    | training classifier   | yes                      | need to set according to evaluation below           |                                                                |
    | training clustering   | yes                      | no                                                  |                                                                |
    | evaluation classifier | n/a                      | probably no, but depends on what we want to analyze |                                                                |
    | evaluation clustering | n/a                      | no                                                  |                                                                |

    :param events_to_num_mentions: series mapping from event identifier to number of mentions per event
    :param c: undersampling approaches (n-1)*c for the largest clusters
    :return: series mapping from event to number of positive pairs to sample per event
    """
    if c <= 0:
        raise ValueError

    # set up CDF telling us the percentage of clusters in the dataset which consist of <= i mentions
    num_events_with_n_mentions = events_to_num_mentions.value_counts().sort_index(ascending=True)
    num_events_with_n_mentions_mult = num_events_with_n_mentions * num_events_with_n_mentions.index.values
    cdf = num_events_with_n_mentions_mult.cumsum() / num_events_with_n_mentions_mult.sum()

    # get position in CDF for each event
    pos_cumu = events_to_num_mentions.map(cdf).values
    num_mentions = events_to_num_mentions.values

    # we want to sample at most k_target pairs for each event
    k_target = c + np.power(num_mentions, 1 - pos_cumu) - 1

    # k_target can exceed the number of pairs we actually have for an event, therefore apply an upper bound of n/2
    # (which is the maximum number of pairs we can sample per event)
    k = np.min([num_mentions / 2, k_target], axis=0)

    num_pairs_to_sample = np.ceil((num_mentions - 1) * k).astype(np.int)

    return pd.Series(num_pairs_to_sample, index=events_to_num_mentions.index)
