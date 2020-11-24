from typing import Dict

from python import DOCUMENT_ID, TOPIC_ID
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage


class DataReducerStage(BaselineDataProcessorStage):

    def __init__(self, pos, config, config_global, logger):
        super(DataReducerStage, self).__init__(pos, config, config_global, logger)
        self._num_topics = config.get("num_topics", None)
        self._num_docs_per_topic = config.get("num_docs_per_topic", None)

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        docs = dataset.documents

        # select subset of topics
        if self._num_topics is not None:
            actual_num_topics = len(docs.index.unique(TOPIC_ID))
            if self._num_topics > actual_num_topics:
                raise ValueError(
                    f"This dataset only has {actual_num_topics} topics, but you asked for a subset of {self._num_topics} topics.")

            topics_to_use = docs.index.unique(TOPIC_ID).to_series().sample(self._num_topics, random_state=0).values
            selected_docs = docs.loc[docs.index.get_level_values(TOPIC_ID).isin(topics_to_use)]
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
        selected_doc_ids = dataset.documents[DOCUMENT_ID]

        dataset.tokens = dataset.tokens.loc[dataset.tokens.index.get_level_values(DOCUMENT_ID).isin(selected_doc_ids)]
        dataset.mentions_action = dataset.mentions_action.loc[
            dataset.mentions_action.index.get_level_values(DOCUMENT_ID).isin(selected_doc_ids)]

        if dataset.mentions_time is not None:
            dataset.mentions_time = dataset.mentions_time.loc[
                dataset.mentions_time.index.get_level_values(DOCUMENT_ID).isin(selected_doc_ids)]

        if dataset.mentions_location is not None:
            dataset.mentions_location = dataset.mentions_location.loc[
                dataset.mentions_location.index.get_level_values(DOCUMENT_ID).isin(selected_doc_ids)]

        if dataset.mentions_participants is not None:
            dataset.mentions_participants = dataset.mentions_participants.loc[
                dataset.mentions_participants.index.get_level_values(DOCUMENT_ID).isin(selected_doc_ids)]

        if dataset.mentions_other is not None:
            dataset.mentions_other = dataset.mentions_other.loc[
                dataset.mentions_other.index.get_level_values(DOCUMENT_ID).isin(selected_doc_ids)]
        return dataset


component = DataReducerStage
