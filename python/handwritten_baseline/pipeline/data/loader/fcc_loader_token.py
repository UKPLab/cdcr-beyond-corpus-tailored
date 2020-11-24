from pathlib import Path

from overrides import overrides

from python import EVENT, DOCUMENT_ID
from python.handwritten_baseline.pipeline.data.loader import football_reader_utils
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.data.loader.fcc_loader_base import FccLoaderBaseStage


class FccLoaderTokenLevelStage(FccLoaderBaseStage):

    def __init__(self, pos, config, config_global, logger):
        super(FccLoaderTokenLevelStage, self).__init__(pos, config, config_global, logger)

        self._token_level_data_dir = config.get("token_level_data_dir", None)
        if self._token_level_data_dir is not None:
            self._token_level_data_dir = Path(self._token_level_data_dir)
            assert self._token_level_data_dir.exists()

        self._drop_other_event_cluster = config["drop_other_event_cluster"]

    @overrides
    def _load_dataset(self) -> Dataset:
        self.logger.info("Reading raw data")
        # load full dataset
        tuples = football_reader_utils.read_split_data(self._sentence_level_data_dir, token_level_data_dir=self._token_level_data_dir)

        assert len(tuples) == 2, "Token-level extensions are mandatory"
        sentence_level_data, token_level_data = tuples

        # create Dataset object from sentence-level annotated data
        documents, tokens, _, _, _ = sentence_level_data
        mentions_action, mentions_participants, mentions_time, mentions_location, semantic_roles = token_level_data

        if self._drop_other_event_cluster:
            mentions_action = mentions_action.loc[mentions_action[EVENT] != "other_event"]

        # We may now have some documents which do not contain any mentions. Remove those to avoid trouble in the
        # model code later on.
        documents_without_mentions = set(documents[DOCUMENT_ID].unique()) - set(mentions_action.index.get_level_values(DOCUMENT_ID).unique())
        documents = documents.loc[~documents[DOCUMENT_ID].isin(documents_without_mentions)]
        tokens = tokens.loc[documents[DOCUMENT_ID]].sort_index()
        mentions_participants = mentions_participants.loc[~mentions_participants.index.get_level_values(DOCUMENT_ID).isin(documents_without_mentions)]
        mentions_location = mentions_location.loc[~mentions_location.index.get_level_values(DOCUMENT_ID).isin(documents_without_mentions)]
        mentions_time = mentions_time.loc[~mentions_time.index.get_level_values(DOCUMENT_ID).isin(documents_without_mentions)]
        semantic_roles = semantic_roles.loc[~semantic_roles[DOCUMENT_ID].isin(documents_without_mentions)]

        documents.sort_index(inplace=True)
        tokens.sort_index(inplace=True)
        mentions_action.sort_index(inplace=True)
        mentions_participants.sort_index(inplace=True)
        mentions_time.sort_index(inplace=True)
        mentions_location.sort_index(inplace=True)

        dataset = Dataset(documents,
                          tokens,
                          mentions_action,
                          mentions_time=mentions_time,
                          mentions_location=mentions_location,
                          mentions_participants=mentions_participants,
                          semantic_roles=semantic_roles)

        return dataset

component = FccLoaderTokenLevelStage