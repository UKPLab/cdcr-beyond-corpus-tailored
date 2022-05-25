from logging import Logger
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from anytree import AnyNode, find_by_attr
from anytree.util import commonancestors
from nltk.corpus import wordnet as wn
from sklearn.metrics import confusion_matrix
from stanfordnlp.protobuf import ParseTree, DependencyGraph
from tabulate import tabulate
from tqdm import tqdm

from python import *
from python.common_components import NLTK
from python.handwritten_baseline import MENTION_TYPE_COARSE, POS, LEMMA
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage
from python.util.util import get_filename_safe_string

PHRASE_TYPE = "phrase-type"
ROOT_TOKEN_IDX = "root-token-idx"
EXTENDED_ROOT_TOKEN_IDX_FROM = "root-" + TOKEN_IDX_FROM
EXTENDED_ROOT_TOKEN_IDX_TO = "root-" + TOKEN_IDX_TO
SYNSET = "synset"

class StatisticsStage(BaselineDataProcessorStage):

    def __init__(self, pos, config, config_global, logger):
        super(StatisticsStage, self).__init__(pos, config, config_global, logger)

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        # access NLTK live_object to trigger dependencies download
        _ = live_objects[NLTK]

        if self.config.get("write_general_statistics", False):
            self.logger.info("Preparing general statistics")
            self.write_general_statistics(dataset)
        if self.config.get("write_stacked_annotations", False):
            self.logger.info("Finding and preparing stacked annotations")
            self.write_stacked_annotations(dataset)
        if self.config.get("write_selection_of_clusters", False):
            self.logger.info("Preparing a selection of clusters")
            self.write_selection_of_clusters(dataset)
        if self.config.get("write_selection_of_full_documents", False):
            self.logger.info("Preparing a selection of full documents")
            self.write_selection_of_full_documents(dataset)
        if self.config.get("write_most_common_phrase_types_of_mentions", False):
            self.logger.info("Preparing the most common phrase types of mentions")
            self.write_most_common_phrase_types_of_mentions(dataset)
        if self.config.get("write_action_mention_span_length", False):
            self.logger.info("Preparing action mention length")
            self.write_action_mention_span_length(dataset)
        if self.config.get("write_phrase_head_statistics", False):
            self.logger.info("Preparing statistics on the phrase head")
            self.write_phrase_head_statistics(dataset)
        if self.config.get("write_mention_pos_occurrence", False):
            self.logger.info("Checking which POS appear in action mention spans")
            self.write_mention_pos_occurrence(dataset)
        return dataset

    @staticmethod
    def mark_mentions_in_sentences_and_join(df_mentions: pd.DataFrame, df_tokens: pd.DataFrame):
        """
        Wraps mentions in ">>> <<<". Performs in-place modifications of `df_tokens`.
        """
        tokens_around_mentions = df_mentions.reset_index().groupby([DOCUMENT_ID, SENTENCE_IDX]).apply(lambda row: df_tokens.loc[tuple(*row[[DOCUMENT_ID, SENTENCE_IDX]].drop_duplicates().values)])
        for idx, mention in df_mentions.iterrows():
            doc_id, mention_id = idx
            sent_idx = mention[SENTENCE_IDX]
            token_idx_from = mention[TOKEN_IDX_FROM]
            token_idx_to = mention[TOKEN_IDX_TO]
            tokens_around_mentions.at[(doc_id, sent_idx, token_idx_from), TOKEN] = ">>>" + tokens_around_mentions.at[(doc_id, sent_idx, token_idx_from), TOKEN]
            tokens_around_mentions.at[(doc_id, sent_idx, token_idx_to-1), TOKEN] = tokens_around_mentions.at[(doc_id, sent_idx, token_idx_to-1), TOKEN] + "<<<"
        sentences_around_mentions = tokens_around_mentions[TOKEN].groupby([DOCUMENT_ID, SENTENCE_IDX]).apply(lambda l: " ".join(l.values))
        return sentences_around_mentions

    def write_general_statistics(self, dataset: Dataset):
        num_topics = len(dataset.documents.index.get_level_values(TOPIC_ID).unique())
        num_subtopics = len(dataset.documents.index.to_frame()[[TOPIC_ID, SUBTOPIC]].drop_duplicates())
        num_documents = len(dataset.documents)

        avg_subtopics_per_topic = num_subtopics / num_topics
        avg_documents_per_subtopic = num_documents / num_subtopics
        avg_documents_per_topic = num_documents / num_topics

        num_tokens = len(dataset.tokens)
        num_types = len(dataset.tokens[TOKEN].unique())

        num_sentences = len(dataset.tokens.index.to_frame()[[DOCUMENT_ID, SENTENCE_IDX]].drop_duplicates())

        num_action_mentions = len(dataset.mentions_action)
        num_action_mentions_stacked = dataset.mentions_action.reset_index().duplicated(
            [DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX_FROM, TOKEN_IDX_TO], keep=False).sum()
        num_action_clusters = len(dataset.mentions_action[EVENT].unique())
        num_action_singleton_clusters = (dataset.mentions_action[EVENT].value_counts() == 1).sum()

        num_participant_mentions = len(
            dataset.mentions_participants) if dataset.mentions_participants is not None else 0
        num_location_mentions = len(dataset.mentions_location) if dataset.mentions_location is not None else 0
        num_time_mentions = len(dataset.mentions_time) if dataset.mentions_time is not None else 0

        clusters_by_size = dataset.mentions_action[EVENT].value_counts()
        cluster_size_distribution = clusters_by_size.value_counts().sort_index()
        cluster_size_distribution.name = "num-occurrences"
        cluster_size_distribution.index.name = "cluster-size"

        percentage_of_docs_with_publication_date = dataset.documents[PUBLISH_DATE].notna().value_counts(normalize=True).get(True, 0.0)

        # how many mentions have participants, time, location linked
        if dataset.semantic_roles is not None:
            # count number of participant, time and location mention for each action mention
            num_mentions_of_coarse_type_per_mention = dataset.semantic_roles.groupby([DOCUMENT_ID, MENTION_ID])[
                MENTION_TYPE_COARSE].value_counts()
            num_mentions_of_coarse_type_per_mention.name = "num-mentions"
            # pivot so that we have [doc-id, mention-id] as the index and the number of location/participant/time mentions for each mention as the columns
            num_mentions_of_coarse_type_per_mention = num_mentions_of_coarse_type_per_mention.reset_index().pivot(
                [DOCUMENT_ID, MENTION_ID], MENTION_TYPE_COARSE, "num-mentions")

            # number of mentions which have at least one argument (location/participant/time) linked, regardless of type
            num_mentions_with_linked_args = len(num_mentions_of_coarse_type_per_mention)

            # number of mentions which have at least one argument (location/participant/time) linked, by type - absolute and relative percentage
            num_mentions_with_args_by_type = {type_: len(
                num_mentions_of_coarse_type_per_mention.loc[num_mentions_of_coarse_type_per_mention[type_] > 0]) for
                                              type_ in ["location", "participants", "time"]}
            num_mentions_with_args_by_type_relative = {type_: num / num_mentions_with_linked_args for type_, num in
                                                       num_mentions_with_args_by_type.items()}
        else:
            num_mentions_with_linked_args = 0
            num_mentions_with_args_by_type = {}
            num_mentions_with_args_by_type_relative = {}

        # NOTE: we can get the number of coreference link by type (within-doc, within-subtopic, cross-subtopic, cross-topic from the mention pair generator during training), so we don't need to repeat that here

        with (self.stage_disk_location / "statistics.txt").open("w") as f:
            f.write(f"""
Number of topics: {num_topics}
Number of subtopics: {num_subtopics}
Number of documents: {num_documents}
Avg. subtopics per topic: {avg_subtopics_per_topic}
Avg. documents per subtopic: {avg_documents_per_subtopic}
Avg. documents per topic: {avg_documents_per_topic}
Number of tokens: {num_tokens}
Number of types: {num_types}
Number of sentences: {num_sentences}
Number of participant mentions: {num_participant_mentions}
Number of time mentions: {num_time_mentions}
Number of location mentions: {num_location_mentions}
Number of action mentions: {num_action_mentions}
Number of stacked action mentions (identical span): {num_action_mentions_stacked}
Number of event clusters: {num_action_clusters}
Number of singleton event clusters: {num_action_singleton_clusters}

Largest clusters:
{tabulate(clusters_by_size.head(10).to_frame("num-mentions"), headers="keys")}

Cluster size distribution:
{tabulate(cluster_size_distribution.to_frame(), headers="keys")}

Percentage of documents with publication date known:
{percentage_of_docs_with_publication_date}
""")

            if dataset.semantic_roles is not None:
                f.write(f"""
Number of action mentions with linked arguments: {num_mentions_with_linked_args}
Number of action mentions with linked arguments of type:
{num_mentions_with_args_by_type}

Number of action mentions with linked arguments of type (relative):
{num_mentions_with_args_by_type_relative}
""")

    def write_stacked_annotations(self, dataset: Dataset):
        mentions_action = dataset.mentions_action.reset_index()
        mentions_action_stacked = mentions_action.loc[
            mentions_action.duplicated([DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX_FROM, TOKEN_IDX_TO], keep=False)]
        sentences_with_stacked_actions = mentions_action_stacked[[DOCUMENT_ID, SENTENCE_IDX]].drop_duplicates()
        with (self.stage_disk_location / "stacked_annotations.txt").open("w") as f:
            for _, row in sentences_with_stacked_actions.iterrows():
                sentence = " ".join(dataset.tokens.loc[tuple(row), TOKEN])
                f.write(f"{row.values} - {sentence}\n")

    def write_selection_of_clusters(self, dataset: Dataset):
        clusters_by_size = dataset.mentions_action[EVENT].value_counts()
        cluster_size_distribution = clusters_by_size.value_counts().sort_index()

        # for a sampled set of clusters, print sampled set of sentences which reference that event
        out_path = self.stage_disk_location / "event_mentions_with_surrounding_sentences"
        out_path.mkdir(parents=True, exist_ok=True)
        for size in cluster_size_distribution.index.values:
            clusters_of_size = clusters_by_size.loc[clusters_by_size == size]
            cluster = clusters_of_size.sample(n=1, random_state=0).index.item()
            mentions_in_cluster = dataset.mentions_action.loc[dataset.mentions_action[EVENT] == cluster]
            sentences_around_mentions_in_cluster = StatisticsStage.mark_mentions_in_sentences_and_join(mentions_in_cluster, dataset.tokens)
            with (out_path / get_filename_safe_string(f"{size:05d}__mentions__{cluster}.txt")).open("w") as f:
                f.write(tabulate(sentences_around_mentions_in_cluster.to_frame("sentence"), headers="keys", tablefmt="grid"))

        # for each event cluster, collect all sentences referencing that event
        # sentences_with_event_mentions = dataset.mentions_action.reset_index()[[DOCUMENT_ID, SENTENCE_IDX]].drop_duplicates()
        # sentences_with_event_mentions = sentences_with_event_mentions.merge(dataset.tokens.reset_index(), on=[DOCUMENT_ID, SENTENCE_IDX])
        # sentences_with_event_mentions = sentences_with_event_mentions.groupby([DOCUMENT_ID, SENTENCE_IDX])[TOKEN].apply(list)
        # mentions_action_with_sentences = dataset.mentions_action.reset_index().merge(sentences_with_event_mentions, on=[DOCUMENT_ID, SENTENCE_IDX])
        # mentions_action_with_sentences.to_json(self.stage_disk_location / "event_mentions_with_sentences.json", orient="records")

    def write_selection_of_full_documents(self, dataset: Dataset):
        # for a sampled set of documents which contain mentions, print full document with event actions highlighted
        documents_with_mentions = pd.Series(dataset.mentions_action.index.unique(DOCUMENT_ID))
        sampled_documents = documents_with_mentions.sample(n=min(len(documents_with_mentions), 100), random_state=0)
        sentences = StatisticsStage.mark_mentions_in_sentences_and_join(dataset.mentions_action.loc[sampled_documents],
                                                                        dataset.tokens.loc[documents_with_mentions].copy())
        documents = sentences.groupby(DOCUMENT_ID).apply(lambda l: " ".join(l.values))
        documents.to_csv(self.stage_disk_location / "documents_with_event_actions_marked.csv")

    @staticmethod
    def parse_tree_to_anytree(tree, next_token_id: int = 0) -> Tuple[AnyNode, int]:
        """
        Recursively converts CoreNLP parse tree object into anytree tree, where leaf nodes have a TOKEN_IDX attribute.
        """
        node = AnyNode(value=tree.value)
        if not tree.child:
            setattr(node, TOKEN_IDX, next_token_id)
            return node, next_token_id + 1
        children = []
        for child in tree.child:
            child_node, next_token_id = StatisticsStage.parse_tree_to_anytree(child, next_token_id)
            children.append(child_node)
        node.children = children
        return node, next_token_id

    @staticmethod
    def get_mention_phrase_type(mentions: pd.DataFrame, constituency: pd.DataFrame) -> pd.DataFrame:
        phrase_type_rows = []
        for (doc_id, mention_id), mention in tqdm(mentions.iterrows(),
                                                  total=len(mentions),
                                                  desc="Finding phrase type",
                                                  mininterval=10):
            sent_idx = mention[SENTENCE_IDX]
            phrase_type_row = {DOCUMENT_ID: doc_id, MENTION_ID: mention_id}

            # obtain parse tree - not every mention may have been constituency parsed
            if (doc_id, sent_idx) in constituency.index:
                tree_bytes = constituency.loc[(doc_id, sent_idx), CONSTITUENCY_TREE]
                tree = ParseTree()
                tree.ParseFromString(tree_bytes)

                # determine phrase type
                tree = StatisticsStage.parse_tree_to_anytree(tree)[0]
                from_node = find_by_attr(tree, name=TOKEN_IDX, value=mention[TOKEN_IDX_FROM])
                to_node = find_by_attr(tree, name=TOKEN_IDX, value=mention[TOKEN_IDX_TO]-1)
                phrase_type_row[PHRASE_TYPE] = commonancestors(from_node, to_node)[-1].value
            phrase_type_rows.append(phrase_type_row)
        phrase_types = pd.DataFrame(phrase_type_rows).set_index([DOCUMENT_ID, MENTION_ID])[PHRASE_TYPE]
        return phrase_types

    def write_most_common_phrase_types_of_mentions(self, dataset: Dataset):
        assert dataset.has(CONSTITUENCY_TREE)
        trees = dataset.get(CONSTITUENCY_TREE)

        phrase_types = StatisticsStage.get_mention_phrase_type(dataset.mentions_action, trees)
        dataset.mentions_action[PHRASE_TYPE] = phrase_types

        # write to disk in various ways
        out_dest = self.stage_disk_location / "phrase_types"
        out_dest.mkdir(parents=True)
        dataset.mentions_action.to_csv(out_dest / "mentions_action_with_phrase_type.csv")

        # sample some mentions of each type and print them
        phrase_type_out_dest = out_dest / "by_type"
        phrase_type_out_dest.mkdir(parents=True)
        for phrase_type, mentions_of_type in dataset.mentions_action.groupby(PHRASE_TYPE):
            sampled_mentions = mentions_of_type.sample(n=min(5, len(mentions_of_type)), random_state=0)
            sentences_marked = StatisticsStage.mark_mentions_in_sentences_and_join(sampled_mentions, dataset.tokens)
            documents = sentences_marked.groupby(DOCUMENT_ID).apply(lambda l: " ".join(l.values))
            documents.to_csv(phrase_type_out_dest / f"{phrase_type}__marked_mentions_with_sentence_context.csv")

        # analyze frequency of types
        phrase_type_frequencies = phrase_types.value_counts()
        phrase_type_frequencies.index.name = "phrase-type"
        phrase_type_frequencies.name = "num-occurrences"
        phrase_type_frequencies.to_csv(out_dest / "phrase_type_frequencies.csv")

    def write_action_mention_span_length(self, dataset: Dataset):
        token_length_distribution = (dataset.mentions_action[TOKEN_IDX_TO] - dataset.mentions_action[TOKEN_IDX_FROM]).value_counts()
        token_length_distribution.name = "num-occurrences"
        token_length_distribution.index.name = "length"
        token_length_distribution.to_csv(self.stage_disk_location / "action_mention_token_length_distribution.csv")


    @staticmethod
    def dependency_graph_to_anytree(graph: DependencyGraph):
        nodes = []
        for n in graph.node:
            node = AnyNode()
            setattr(node, TOKEN_IDX, n.index - 1)
            nodes.append(node)

        for e in graph.edge:
            target = nodes[e.target-1]
            target.parent = nodes[e.source-1]
            target.dep = e.dep

        roots = [nodes[i-1] for i in graph.root]
        return roots

    @staticmethod
    def get_mention_root(mentions: pd.DataFrame, dependency: pd.DataFrame, logger: Logger) -> pd.DataFrame:
        mentions_with_root = []
        for (doc_id, mention_id), mention in tqdm(mentions.iterrows(),
                                                  total=len(mentions),
                                                  desc="Phrase head analysis",
                                                  mininterval=10):
            mention_with_root = {DOCUMENT_ID: doc_id, MENTION_ID: mention_id}
            sent_idx = mention[SENTENCE_IDX]

            # obtain dependency graph
            from_node, to_node = None, None
            if (doc_id, sent_idx) in dependency.index:
                graph_bytes = dependency.loc[(doc_id, sent_idx), DEPENDENCY_TREE]
                graph = DependencyGraph()
                graph.ParseFromString(graph_bytes)

                # determine phrase type
                graph = StatisticsStage.dependency_graph_to_anytree(graph)
                if len(graph) > 1:
                    raise ValueError("There should only ever be one graph per sentence.")
                if graph:
                    graph = graph[0]
                    from_node = find_by_attr(graph, name=TOKEN_IDX, value=mention[TOKEN_IDX_FROM])
                    to_node = find_by_attr(graph, name=TOKEN_IDX, value=mention[TOKEN_IDX_TO] - 1)

            # bail out early if dep parsing fails
            if from_node is None or to_node is None:
                logger.warning(f"Could not determine head for ({doc_id}, {mention_id}).")
                mentions_with_root.append(mention_with_root)
                continue

            # find common parent of mention span: move towards the root of the tree level by level until the two paths
            # meet at the same node: the root node of the span
            lookup_failure = False
            while getattr(from_node, TOKEN_IDX) != getattr(to_node, TOKEN_IDX):
                if from_node.depth > to_node.depth:
                    from_node = from_node.parent
                elif from_node.depth < to_node.depth:
                    to_node = to_node.parent
                else:
                    if from_node.depth == 0:  # if both nodes are on the same depth but not equal, a key assertion is broken
                        lookup_failure = True
                        break
                    from_node, to_node = from_node.parent, to_node.parent
            assert not lookup_failure
            mention_root = to_node
            root_token_idx = getattr(mention_root, TOKEN_IDX)

            # Cases when this happens: https://abcnews.go.com/Health/staggering-number-measles-cases-us-part-300-global/story?id=62427655
            # 1. "That can be seen playing out currently in Brooklyn, New York, where >>>five anonymous mothers are
            #    pushing back<<< against Mayor Bill de Blasio's >>>emergency order calling for<<< everyone over the age
            #    of six months to be vaccinated in certain at-risk communities."
            #    Here, the head found for "emergency order calling for" will be "pushing", but that lies outside of the
            #    original span (and inside another span, which caused trouble elsewhere...).
            # 2. Much more often though, the dependency parser is just not good enough and attaches a PP incorrectly.
            #    For example, in "That is the same decision as >>>during the most recent Winter Olympics in South Korea last year<<<."
            #    it attaches "year" to "decision" instead of "Olympics", meaning the root lies outside the original
            #    span. 2021-05: Cross-checking these cases against newer CoreNLP / stanza versions, it seems to happen
            #    less there. therefore TODO: Update CoreNLP/stanza models.
            root_lies_inside_original_span = root_token_idx >= mention[TOKEN_IDX_FROM] and \
                                             root_token_idx < mention[TOKEN_IDX_TO]
            if not root_lies_inside_original_span:
                logger.warning(f"The phrase head of mention ({doc_id}, {mention_id}) lies outside its own anchor text. Skipped.")
                mentions_with_root.append(mention_with_root)
                continue

            # expand the root a bit and iteratively include compounds (https://universaldependencies.org/en/dep/compound.html)
            root_to_token_idx = root_token_idx
            root_from_token_idx = root_to_token_idx
            compound_iter_node = mention_root
            while True:
                expand_nodes = [n for n in compound_iter_node.children if n.dep in ["compound", "flat"]]
                if not expand_nodes:
                    break
                compound_iter_node = expand_nodes[0]
                root_from_token_idx = getattr(compound_iter_node, TOKEN_IDX)

            mention_with_root.update({ROOT_TOKEN_IDX: root_token_idx,
                                      EXTENDED_ROOT_TOKEN_IDX_FROM: root_from_token_idx,
                                      EXTENDED_ROOT_TOKEN_IDX_TO: root_to_token_idx + 1,    # exclusive end boundary as usual
                                      })
            mentions_with_root.append(mention_with_root)
        mentions_with_root = pd.DataFrame(mentions_with_root).set_index([DOCUMENT_ID, MENTION_ID])
        return mentions_with_root

    @staticmethod
    def _map_corenlp_pos_to_wordnet_pos(pos: str):
        if pos.startswith("J"):
            return wn.ADJ
        elif pos.startswith("V"):
            return wn.VERB
        elif "RB" in pos:
            return wn.ADV
        else:
            return wn.NOUN

    def write_phrase_head_statistics(self, dataset: Dataset):
        assert dataset.has(DEPENDENCY_TREE)
        trees = dataset.get(DEPENDENCY_TREE)

        mention_roots = StatisticsStage.get_mention_root(dataset.mentions_action, trees, self.logger)
        dataset.mentions_action[ROOT_TOKEN_IDX] = mention_roots[ROOT_TOKEN_IDX]
        dataset.mentions_action[EXTENDED_ROOT_TOKEN_IDX_FROM] = mention_roots[EXTENDED_ROOT_TOKEN_IDX_FROM]
        dataset.mentions_action[EXTENDED_ROOT_TOKEN_IDX_TO] = mention_roots[EXTENDED_ROOT_TOKEN_IDX_TO]

        # look up POS for extended mention root: merge mentions df with tokens df
        mentions_with_root_token = dataset.mentions_action.dropna(subset=[ROOT_TOKEN_IDX]).reset_index().merge(dataset.tokens, left_on=[DOCUMENT_ID, SENTENCE_IDX, ROOT_TOKEN_IDX], right_on=[DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX]).set_index([DOCUMENT_ID, MENTION_ID])
        mentions_with_root_token["wordnet-pos"] = mentions_with_root_token[POS].map(StatisticsStage._map_corenlp_pos_to_wordnet_pos)
        mentions_with_root_token["synset"] = mentions_with_root_token[[LEMMA, "wordnet-pos"]].apply(lambda row: wn.synsets(row[LEMMA], pos=row["wordnet-pos"]), axis=1)
        # pick the first synset or use the lemma as a fallback
        mentions_with_root_token["synset"] = mentions_with_root_token[["synset", LEMMA]].apply(lambda row: row["synset"][0].name() if row["synset"] else row[LEMMA], axis=1)

        # work with rough POS, first letter only
        mentions_with_root_token[POS] = mentions_with_root_token[POS].str.get(0)

        # analyze frequency of POS and synsets
        for col in [POS, SYNSET]:
            frequencies = mentions_with_root_token[col].value_counts()
            frequencies.index.name = col
            frequencies.name = "num-occurrences"
            frequencies.to_csv(self.stage_disk_location / f"{col}_of_mention_root.csv")

        # now the entropy statistics
        LEXICAL_TRIGGER = "lexical-trigger"
        NORM_TRIGGER_ENTROPY = "norm-trigger-entropy"
        NUM_UNIQUE_TRIGGERS = "num-unique-triggers"
        NUM_MENTIONS = "num-mentions"

        mentions_per_event = mentions_with_root_token[EVENT].value_counts()
        non_singleton_events = mentions_per_event.loc[mentions_per_event > 1]
        non_singleton_mentions = mentions_with_root_token.loc[mentions_with_root_token[EVENT].isin(non_singleton_events.index)].copy()

        def get_mention_text_from_mention(row: pd.Series) -> str:
            return " ".join(dataset.tokens.loc[(row.name[0], row[SENTENCE_IDX], slice(row[EXTENDED_ROOT_TOKEN_IDX_FROM], row[EXTENDED_ROOT_TOKEN_IDX_TO] - 1)), TOKEN].values)
        non_singleton_mentions[LEXICAL_TRIGGER] = non_singleton_mentions.apply(get_mention_text_from_mention, axis=1).str.lower()

        # per event, a list of integers where each integer represents one unique mention trigger
        factorized_triggers = non_singleton_mentions.groupby(EVENT).apply(lambda df: pd.Series(df[LEXICAL_TRIGGER].factorize()[0]))

        # Compute entropy of lexical triggers per cluster. The number of unique lexical triggers per cluster varies between
        # clusters (i.e. the domain of the random variable differs between clusters), therefore we normalize by the number of
        # unique lexical triggers per cluster. See https://en.wikipedia.org/wiki/Entropy_(information_theory)#Efficiency_(normalized_entropy)
        p_xi = factorized_triggers.groupby(EVENT).value_counts(normalize=True)
        num_unique_triggers = factorized_triggers.groupby(EVENT).nunique()
        norm_entropy = -1 * (p_xi * p_xi.map(np.log2) / np.log2(num_unique_triggers)).groupby(EVENT).sum()

        # get rid of "-0.0" and inaccuracies with entropies slightly larger than 1.0
        norm_entropy = norm_entropy.clip(lower=0.0, upper=1.0)

        # sort clusters by difficulty (easy clusters have many mentions with low entropy)
        event_variation = pd.DataFrame({NORM_TRIGGER_ENTROPY: norm_entropy, NUM_UNIQUE_TRIGGERS: num_unique_triggers, NUM_MENTIONS: non_singleton_events}).sort_values(by=[NORM_TRIGGER_ENTROPY, NUM_UNIQUE_TRIGGERS, NUM_MENTIONS], ascending=[True, False, False])
        event_variation.index.name = EVENT
        event_variation.to_csv(self.stage_disk_location / "lexical_trigger_variation_per_event.csv")

        # export a selection of example clusters from different percentiles (easy to hard)
        num_events_per_percentile = 10
        n = len(event_variation)
        event_ids_at_percentiles = np.hstack([event_variation.index[i:i + num_events_per_percentile].values for i in [0, n // 4, n // 2, 3 * n // 4, n - num_events_per_percentile]])
        mentions_at_percentiles = non_singleton_mentions.loc[non_singleton_mentions[EVENT].isin(event_ids_at_percentiles)]
        triggers_of_mentions = mentions_at_percentiles.groupby(EVENT)[LEXICAL_TRIGGER].apply(lambda s: ", ".join(s.values))
        events_at_percentiles = event_variation.merge(triggers_of_mentions.to_frame("triggers"), left_index=True, right_index=True)
        with (self.stage_disk_location / "lexical_trigger_variation_per_event_examples.txt").open("w") as f:
            f.write(tabulate(events_at_percentiles, headers="keys"))

    def write_mention_pos_occurrence(self, dataset: Dataset):
        HAS_V = "has-v"
        HAS_N = "has-n"
        pos_occurrences = []
        for (doc_id, mention_id), mention in dataset.mentions_action.iterrows():
            coarse_pos_in_mention_span = dataset.tokens.loc[(doc_id, mention[SENTENCE_IDX], slice(mention[TOKEN_IDX_FROM], mention[TOKEN_IDX_TO])), POS].str.get(0).values
            pos_occurrences.append({DOCUMENT_ID: doc_id,
                                    MENTION_ID: mention_id,
                                    HAS_V: "V" in coarse_pos_in_mention_span,
                                    HAS_N: "N" in coarse_pos_in_mention_span})
        pos_occurrences = pd.DataFrame(pos_occurrences).set_index([DOCUMENT_ID, MENTION_ID])

        # create contingency table for "has noun", "has verb" in mention spans
        pos_occurrences_contingency = pd.DataFrame(
            confusion_matrix(pos_occurrences[HAS_V], pos_occurrences[HAS_N], labels=[True, False], normalize="all"),
            index=["has verb", "no verb"], columns=["has noun", "no noun"])

        with (self.stage_disk_location / "verb_noun_occurrence_in_mention_span.txt").open("w") as f:
            f.write(tabulate(pos_occurrences_contingency, tablefmt="grid", headers="keys"))

component = StatisticsStage
