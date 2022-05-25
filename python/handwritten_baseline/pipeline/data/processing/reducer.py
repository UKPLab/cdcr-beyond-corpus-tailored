from typing import Dict
import pandas as pd

from python import *
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage


class DataReducerStage(BaselineDataProcessorStage):

    def __init__(self, pos, config, config_global, logger):
        super(DataReducerStage, self).__init__(pos, config, config_global, logger)

        # for topical structure reduction
        self._num_topics = config.get("num_topics", None)
        self._topics_to_select = config.get("topics_to_select", None)
        if self._num_topics is not None and self._topics_to_select is not None:
            raise ValueError("num_topics and topics_to_select are mutually exclusive!")
        self._num_docs_per_topic = config.get("num_docs_per_topic", None)

        # for cluster reduction
        self._event_cluster_size_interval_to_keep = config.get("event_cluster_size_interval_to_keep", None)
        if self._event_cluster_size_interval_to_keep is not None and len(self._event_cluster_size_interval_to_keep) != 2:
            raise ValueError("cluster_size_interval_to_keep must be a list of two integers.")

        # for textual content reduction
        self._drop_sentences_without_action_mentions = config.get("drop_sentences_without_action_mentions", False)

    def _reduce_topical_structure(self, dataset: Dataset):
        docs = dataset.documents

        # select subset of topics
        if self._num_topics is not None:
            actual_num_topics = len(docs.index.unique(TOPIC_ID))
            if self._num_topics > actual_num_topics:
                raise ValueError(
                    f"This dataset only has {actual_num_topics} topics, but you asked for a subset of {self._num_topics} topics.")

            topics_to_use = docs.index.unique(TOPIC_ID).to_series().sample(self._num_topics, random_state=0).values
            selected_docs = docs.loc[docs.index.get_level_values(TOPIC_ID).isin(topics_to_use)]
        elif self._topics_to_select is not None:
            to_select = [str(o) for o in self._topics_to_select]
            try:
                selected_docs = docs.loc[to_select]
            except KeyError as e:
                raise ValueError("Cannot select topics which are not part of the dataset.", e)
        else:
            selected_docs = docs

        # select subset of documents per topic
        if self._num_docs_per_topic is not None:
            selected_docs = selected_docs.groupby(TOPIC_ID, as_index=False).apply(
                lambda df: df.sample(min(len(df), self._num_docs_per_topic), random_state=0))
            selected_docs.index = selected_docs.index.droplevel(0)
        selected_docs.sort_index(inplace=True)

        self.logger.warning(f"Number of documents limited to {len(selected_docs)}!")
        dataset.documents = selected_docs
        selected_doc_ids = selected_docs[DOCUMENT_ID]

        dataset.tokens = dataset.tokens.loc[dataset.tokens.index.get_level_values(DOCUMENT_ID).isin(selected_doc_ids)]

        for attr in ["mentions_action", "mentions_time", "mentions_participants", "mentions_location", "mentions_other"]:
            df = getattr(dataset, attr)
            if df is not None:
                df_updated = df.loc[df.index.get_level_values(DOCUMENT_ID).isin(selected_doc_ids)]
                setattr(dataset, attr, df_updated)

    def _reduce_clusters(self, dataset):
        if self._event_cluster_size_interval_to_keep:
            _from, _to = self._event_cluster_size_interval_to_keep      # inclusive, inclusive
            mentions_action = dataset.mentions_action

            mentions_per_event = mentions_action[EVENT].value_counts()
            mentions_to_select = mentions_per_event.loc[(mentions_per_event >= _from) & (mentions_per_event <= _to)]
            mentions_action_reduced = mentions_action.loc[mentions_action[EVENT].isin(mentions_to_select.index)]

            dataset.mentions_action = mentions_action_reduced
            self.logger.info(f"Reduced action mentions to those from clusters with size {_from} to {_to} (from {len(mentions_action)} to {len(mentions_action_reduced)} mentions total).")

    def _reduce_textual_content(self, dataset):
        if self._drop_sentences_without_action_mentions:
            sentences_with_action_mentions = pd.MultiIndex.from_frame(dataset.mentions_action.reset_index()[[DOCUMENT_ID, SENTENCE_IDX]].drop_duplicates())
            is_token_inside_sentence_with_action_mentions = dataset.tokens.index.droplevel(TOKEN_IDX).isin(sentences_with_action_mentions)
            dataset.tokens = dataset.tokens.loc[is_token_inside_sentence_with_action_mentions]

            dataset.documents = dataset.documents.loc[dataset.documents[DOCUMENT_ID].isin(sentences_with_action_mentions.get_level_values(DOCUMENT_ID).unique())]

            # remove other mentions from sentences just removed
            for attr in ["mentions_time", "mentions_participants", "mentions_location", "mentions_other"]:
                df = getattr(dataset, attr)
                if df is not None:
                    df_no_idx = df.reset_index()
                    df_no_idx = df_no_idx.loc[df_no_idx[[DOCUMENT_ID, SENTENCE_IDX]].apply(lambda row: tuple(row.values) in sentences_with_action_mentions, axis=1)]
                    df_updated = df_no_idx.set_index([DOCUMENT_ID, MENTION_ID])
                    setattr(dataset, attr, df_updated)

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        self._reduce_topical_structure(dataset)
        self._reduce_clusters(dataset)
        self._reduce_textual_content(dataset)
        return dataset


component = DataReducerStage
