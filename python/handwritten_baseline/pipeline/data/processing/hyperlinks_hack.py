from itertools import chain
from typing import Dict, List, Set

import kahypar
import numpy as np
import pandas as pd

from python import *
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage
from python.handwritten_baseline.pipeline.data.processing.statistics import StatisticsStage, \
    EXTENDED_ROOT_TOKEN_IDX_FROM, EXTENDED_ROOT_TOKEN_IDX_TO
from python.util.util import PROJECT_RESOURCES_PATH

CLAUSE_LABELS = ["S", "SBAR", "SBARQ", "SINV", "SQ"]
SOME_PHRASE_LABELS = ["PP", "ADJP", "ADVP", "FRAG", "INTJ"]


class HyperlinksHackStage(BaselineDataProcessorStage):
    """
    This aptly named stage contains a few hacks to make hyperlinks data more compatible with regular unsuspecting CDCR
    implementations:
    - dropping phrase-level mentions
    - reducing mention spans to syntactic heads
    - creating fake topics to combat explosion of negative pairs
    TODO make these transformations applicable to any corpus (not just action mentions, re-topification, ...)
    """

    def __init__(self, pos, config, config_global, logger):
        super(HyperlinksHackStage, self).__init__(pos, config, config_global, logger)

        # these have to be created prior from corenlp_pos_ner.py, only for sentences containing mentions!
        self._constituency = config["constituency"]
        self._dependency = config["dependency"]

        # Preliminary experiments with ABC/BBC showed that dropping phrasal mentions made results worse. Kept for
        # for legacy reasons.
        self._drop_phrasal_mentions = config["drop_phrasal_mentions"]
        self._reduce_span_to_dependency_head = config["reduce_span_to_dependency_head"]

        # to not overburden other CDCR systems with a single topic of 10k documents (generating tons of pairs),
        # we portion the documents into fake topics of a certain target number of documents
        self._target_num_docs_per_fake_topic = config["target_num_docs_per_fake_topic"]

        # overall limit for the data returned by this stage, measured as the number of mentions
        self._total_mentions_limit = config["total_mentions_limit"]

    @staticmethod
    def _randomized_fake_topic_creation(mentions: pd.DataFrame,
                                        total_mentions_limit: int,
                                        target_num_docs_per_fake_topic: int) -> List[Set[str]]:
        """
        *Deprecated, do not use!*

        Partitions the documents of the given mentions into fake topics by randomly sampling one event and greedily
        keeping all connected documents (unless those were already assigned to a different fake topic previously) until
        the current topic as enough documents.
        By filling topics event by event, many cluster edges may be cut depending on the (randomized) order in which
        events are drawn. The result usually is quite bad for this reason, as many singleton events are produced (or
        3-clusters turning into 2-clusters, etc.).
        """
        # create fake topics
        # create shuffled set of events to pick from
        set_of_events = mentions[EVENT].unique()
        random = np.random.RandomState(seed=0)
        random.shuffle(set_of_events)

        mentions_in_topics = []
        topics = []     # type: List[Set[str]]
        documents_in_topics = set()
        while len(mentions_in_topics) < total_mentions_limit and set_of_events.size:
            topic = set()   # set of document IDs
            while len(topic) < target_num_docs_per_fake_topic and set_of_events.size:
                # pick random event, find its cluster (mentions of that event)
                event, set_of_events = set_of_events[0], set_of_events[1:]
                cluster = mentions.loc[mentions[EVENT] == event]

                # across all topics, documents must stay unique, therefore remove documents at this point which were
                # already picked for other topics
                cluster = cluster.drop(index=documents_in_topics, level=DOCUMENT_ID, errors="ignore")

                # collect for loop termination criterion and later
                mentions_in_topics += cluster.index.values.tolist()

                # update topic in the making with documents from this cluster
                documents_of_cluster = cluster.index.unique(DOCUMENT_ID)
                topic |= set(documents_of_cluster)
            topics.append(topic)
            documents_in_topics |= topic
        return topics

    @staticmethod
    def _hypergraph_fake_topic_partitioning(mentions: pd.DataFrame,
                                            total_mentions_limit: int,
                                            target_num_docs_per_fake_topic: int,
                                            epsilon: float = 0.03) -> List[Set[str]]:
        """
        Uses "proper" hypergraph partitioning to produce fake topics with minimal cuts to coreference cluster edges.
        """

        # kahypar does not support hyperedges where one vertex appears twice (== documents with two or more hyperlinks
        # to the same other document), see https://github.com/kahypar/kahypar/issues/78#issuecomment-768531716 . We
        # handle such cases by counting the number of edges between a source and target document and using that as edge
        # weights.
        edges_agg_by_doc = mentions.reset_index().groupby([DOCUMENT_ID, EVENT]).size()
        edges_agg_by_doc.name = "weight"
        edges_agg_by_doc = edges_agg_by_doc.reset_index()

        num_nodes = edges_agg_by_doc[DOCUMENT_ID].nunique()
        num_nets = edges_agg_by_doc[EVENT].nunique()

        # convert documents (==pins) into integers by factorizing
        edges_agg_by_doc["doc-id-factorized"] = edges_agg_by_doc[DOCUMENT_ID].factorize()[0]

        # per event, collect integer IDs of documents which contain mentions of that event, i.e. the edge vector of
        # each hypergraph
        hyperedges_by_event = edges_agg_by_doc.groupby(EVENT)["doc-id-factorized"].apply(list)
        hyperedges = list(chain(*hyperedges_by_event.values.flat))
        hyperedge_indices = hyperedges_by_event.map(len).cumsum()
        hyperedge_indices = [0, *hyperedge_indices.values.tolist()]
        edge_weights = edges_agg_by_doc["weight"].values.tolist()
        node_weights = [1] * num_nodes

        k = num_nodes // target_num_docs_per_fake_topic
        hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)

        context = kahypar.Context()
        context.loadINIconfiguration(str(PROJECT_RESOURCES_PATH / "kahypar" / "cut_kKaHyPar_sea20.ini"))
        context.setK(k)
        context.setEpsilon(epsilon)

        kahypar.partition(hypergraph, context)
        # retrieve target partition per document
        doc_ids_to_factorized_doc_ids = edges_agg_by_doc[[DOCUMENT_ID, "doc-id-factorized"]].drop_duplicates().set_index(DOCUMENT_ID)
        doc_id_to_partition = doc_ids_to_factorized_doc_ids["doc-id-factorized"].map(hypergraph.blockID)

        # apply total number of mentions limit: sort partitions by their number of mentions (many mentions to few
        # mentions), then keep as many partitions as necessary to reach the target
        mentions["partition"] = mentions.index.get_level_values(DOCUMENT_ID).map(doc_id_to_partition)
        partitions_by_num_mentions = mentions.groupby("partition").size().sort_values(ascending=False).cumsum()
        max_index_to_keep = pd.Index(partitions_by_num_mentions.values).get_loc(total_mentions_limit, method="nearest")
        # max index is inclusive, +1 it
        partitions_to_keep = partitions_by_num_mentions.iloc[:max_index_to_keep+1].index
        
        selected_mentions = mentions.loc[mentions["partition"].isin(partitions_to_keep)]
        
        # now produce topics; a list of set of document ids
        topics = selected_mentions.reset_index().groupby("partition")[DOCUMENT_ID].apply(set).values.tolist()
        return topics

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        mentions = dataset.mentions_action
        documents = dataset.documents
        tokens = dataset.tokens

        num_documents_before = len(documents)
        num_mentions_before = len(mentions)

        # sideload parse results
        constituency = pd.read_hdf(self._constituency)      # type: pd.DataFrame
        dependency = pd.read_hdf(self._dependency)          # type: pd.DataFrame

        mentions_to_drop = pd.Series(False, index=mentions.index)
        if self._drop_phrasal_mentions:
            phrase_types_of_mentions = StatisticsStage.get_mention_phrase_type(mentions, constituency)
            phrase_types_to_drop = CLAUSE_LABELS + SOME_PHRASE_LABELS
            mentions_to_drop = phrase_types_of_mentions.isin(phrase_types_to_drop)
            self.logger.info(f"Dropped phrasal mentions (reduction by {mentions_to_drop.value_counts().get(True, 0)} mentions).")

        if self._reduce_span_to_dependency_head:
            mention_roots = StatisticsStage.get_mention_root(mentions, dependency, self.logger)

            # fill the failed ones with -1 (these will be removed in a moment anyway) to succeed with the type conversion
            mentions[EXTENDED_ROOT_TOKEN_IDX_FROM] = mention_roots[EXTENDED_ROOT_TOKEN_IDX_FROM].fillna(-1).astype(int)
            mentions[EXTENDED_ROOT_TOKEN_IDX_TO] = mention_roots[EXTENDED_ROOT_TOKEN_IDX_TO].fillna(-1).astype(int)

            # we need to sort out failed instances here: mentions with NaN or invalid boundaries
            num_mentions_before = mentions_to_drop.sum()
            mentions_to_drop |= (mentions[EXTENDED_ROOT_TOKEN_IDX_FROM] == -1) | \
                                (mentions[EXTENDED_ROOT_TOKEN_IDX_TO] == -1) | \
                                (mentions[EXTENDED_ROOT_TOKEN_IDX_TO] <= mentions[EXTENDED_ROOT_TOKEN_IDX_FROM])
            num_mentions_after = mentions_to_drop.sum()
            self.logger.info(f"Shrunk mention spans to the compound-extended syntactic head of each span. {num_mentions_after - num_mentions_before} mentions had to be removed because of parsing issues.")

            # The act of extending spans to compounds can sometimes introduce overlapping mentions (in cases like
            # ">>>Tesla<<< owner >>>Elon Musk<<<" where Elon Musk would include Tesla when being extended) - we detect
            # these cases and do not apply the mention root parsing on them. For other mentions we overwrite the token
            # spans with the parsed extended spans.
            mentions.reset_index(inplace=True)      # pd.duplicated necessitates resetting the index bc. of DOCUMENT_ID
            is_overlap_candidate = mentions.duplicated([DOCUMENT_ID, SENTENCE_IDX, EXTENDED_ROOT_TOKEN_IDX_FROM], keep=False)
            is_valid_parse = (mentions[EXTENDED_ROOT_TOKEN_IDX_FROM] != -1) & (mentions[EXTENDED_ROOT_TOKEN_IDX_TO] != -1)
            is_actual_extension = (mentions[TOKEN_IDX_FROM] != mentions[EXTENDED_ROOT_TOKEN_IDX_FROM]) | (mentions[TOKEN_IDX_TO] != mentions[EXTENDED_ROOT_TOKEN_IDX_TO])
            do_update_mention_to_parsed_extended_version = ~(is_overlap_candidate & is_valid_parse & is_actual_extension)

            mentions.loc[do_update_mention_to_parsed_extended_version, TOKEN_IDX_FROM] = mentions[EXTENDED_ROOT_TOKEN_IDX_FROM]
            mentions.loc[do_update_mention_to_parsed_extended_version, TOKEN_IDX_TO] = mentions[EXTENDED_ROOT_TOKEN_IDX_TO]

            # undo temporary changes
            mentions.set_index([DOCUMENT_ID, MENTION_ID], inplace=True)
            mentions.drop(columns=[EXTENDED_ROOT_TOKEN_IDX_FROM, EXTENDED_ROOT_TOKEN_IDX_TO], inplace=True)

        mentions.drop(mentions.index[mentions_to_drop], inplace=True)

        # drop any singleton clusters that the previous filtering steps may have created
        num_mentions_per_event = mentions[EVENT].value_counts()
        singleton_events = num_mentions_per_event.loc[num_mentions_per_event == 1]
        self.logger.info(f"{len(singleton_events)} events which turned into singletons after the previous filtering steps were dropped.")
        mentions.drop(mentions.index[mentions[EVENT].isin(singleton_events.index)], inplace=True)

        # old alternative (kept for reference): HyperlinksHackStage._randomized_fake_topic_creation
        partitioning_func = HyperlinksHackStage._hypergraph_fake_topic_partitioning
        topics = partitioning_func(mentions, self._total_mentions_limit, self._target_num_docs_per_fake_topic)

        # apply topics on documents dataframe
        documents_of_fake_topics = []
        for i, topic in enumerate(topics):
            topic_name = f"FAKE_TOPIC_{i:04}"

            documents_of_topic = documents.loc[documents.index.get_level_values(DOCUMENT_ID).isin(topic)].rename(index={"hyperlinks": topic_name})
            documents_of_fake_topics.append(documents_of_topic)
        documents_of_fake_topics = pd.concat(documents_of_fake_topics).sort_index()

        # remove any tokens, mentions not in the fake topics
        tokens_of_fake_topics = tokens.loc[tokens.index.get_level_values(DOCUMENT_ID).isin(documents_of_fake_topics[DOCUMENT_ID])]
        mentions_of_fake_topics = mentions.loc[mentions.index.get_level_values(DOCUMENT_ID).isin(documents_of_fake_topics[DOCUMENT_ID])].sort_index()

        self.logger.info(f"Created {len(topics)} fake topics. Overall number of documents reduced from {num_documents_before} to {len(documents_of_fake_topics)}. Mention count reduced from {num_mentions_before} to {len(mentions_of_fake_topics)}.")

        dataset.documents, dataset.tokens, dataset.mentions_action = documents_of_fake_topics, tokens_of_fake_topics, mentions_of_fake_topics

        return dataset


component = HyperlinksHackStage
