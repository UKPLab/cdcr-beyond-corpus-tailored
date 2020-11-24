from overrides import overrides

from python import TOKEN_IDX, SENTENCE_IDX, DOCUMENT_ID, TOKEN_IDX_FROM, TOKEN_IDX_TO, MENTION_ID
from python.handwritten_baseline.pipeline.data.loader import football_reader_utils
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.data.loader.fcc_loader_base import FccLoaderBaseStage


class FccLoaderSentenceLevelStage(FccLoaderBaseStage):

    def __init__(self, pos, config, config_global, logger):
        super(FccLoaderSentenceLevelStage, self).__init__(pos, config, config_global, logger)

    @overrides
    def _load_dataset(self) -> Dataset:
        self.logger.info("Reading raw data")
        # load full dataset
        tuples = football_reader_utils.read_split_data(self._sentence_level_data_dir)

        # create Dataset object from sentence-level annotated data
        documents, tokens, mentions_action, _, _ = tuples[0]

        # add token indices for action mentions, so that the format of this dataframe matches that of the other corpora
        max_token_index_in_sentence = tokens.index.to_frame(index=False).groupby([DOCUMENT_ID, SENTENCE_IDX])[TOKEN_IDX].max()
        mentions_action_with_max_token = mentions_action.reset_index().merge(max_token_index_in_sentence, on=[DOCUMENT_ID, SENTENCE_IDX]).rename(columns={TOKEN_IDX: TOKEN_IDX_TO})
        mentions_action_with_max_token[TOKEN_IDX_TO] += 1   # remember, we use exclusive span boundaries
        mentions_action_with_max_token[TOKEN_IDX_FROM] = 0
        mentions_action = mentions_action_with_max_token.set_index([DOCUMENT_ID, MENTION_ID])

        # We may now have some documents which do not contain any mentions. Remove those to avoid trouble in the
        # model code later on.
        documents_without_mentions = set(documents[DOCUMENT_ID].unique()) - set(mentions_action.index.get_level_values(DOCUMENT_ID).unique())
        documents = documents.loc[~documents[DOCUMENT_ID].isin(documents_without_mentions)]
        tokens = tokens.loc[documents[DOCUMENT_ID]].sort_index()

        documents.sort_index(inplace=True)
        tokens.sort_index(inplace=True)
        mentions_action.sort_index(inplace=True)

        dataset = Dataset(documents,
                          tokens,
                          mentions_action)

        return dataset

component = FccLoaderSentenceLevelStage