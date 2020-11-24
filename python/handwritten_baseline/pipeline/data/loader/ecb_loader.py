from pathlib import Path
from typing import List, Optional

from overrides import overrides

from python import MENTION_TYPE, TOKEN_IDX_TO, TOKEN_IDX_FROM, TOPIC_ID, DOCUMENT_ID, EVENT, ENTITY, \
    MENTION_TYPES_ACTION, MENTION_TYPES_TIME, MENTION_TYPES_LOCATION, MENTION_TYPES_PARTICIPANTS, HUMAN_PART_GPE
from python.handwritten_baseline.pipeline.data.loader import ecb_reader_utils
from python.handwritten_baseline.pipeline.data.base import BaselineDataLoaderStage, Dataset


class EcbLoaderStage(BaselineDataLoaderStage):

    def __init__(self, pos, config, config_global, logger):
        super(EcbLoaderStage, self).__init__(pos, config, config_global, logger)

        self._path_to_data_split = Path(config["path_to_data_split"])
        self._sentence_filter_csv = config.get("sentence_filter_csv", None)
        self._topics_to_load = config.get("topics", None)  # type: Optional[List]
        assert self._path_to_data_split.exists()

        if self._sentence_filter_csv is None:
            logger.warning("No sentence filter CSV specified for ECB+. This should be used for comparable evaluation! See corpus readme.")

    @overrides
    def _load_dataset(self) -> Dataset:
        self.logger.info("Reading raw data")
        documents, tokens, mentions, entities_events = ecb_reader_utils.read_split_data(self._path_to_data_split,
                                                                                        self._sentence_filter_csv)

        # remove invalid cross-sentence mentions - there is for example one in 36_4ecbplus
        mentions_valid = mentions.loc[mentions[TOKEN_IDX_FROM] < mentions[TOKEN_IDX_TO]]
        if len(mentions_valid) < len(mentions):
            self.logger.warning(f"Removed {len(mentions) - len(mentions_valid)} invalid mention(s) present in the gold data.")
        mentions = mentions_valid

        # in 41_4ecb there is a participant mention with type "HUMAN_PART" which should be "HUMAN_PART_GPE"
        mentions[MENTION_TYPE] = mentions[MENTION_TYPE].replace({"HUMAN_PART": HUMAN_PART_GPE})

        if self._topics_to_load is not None:
            # perform topic selection
            topics_to_load = {str(v) for v in self._topics_to_load}
            topics_in_split = set(documents.index.get_level_values(TOPIC_ID).unique())

            topics_not_present = topics_to_load - topics_in_split
            if topics_not_present:
                self.logger.warning(f"Cannot load these topics because they are not part of the split: {', '.join(sorted(topics_not_present))}")
            topics_to_load = list(topics_in_split & topics_to_load)
            if not topics_to_load:
                raise ValueError("At least one topic has to be selected")
            self.logger.info(f"Using topic(s) {', '.join(sorted(topics_to_load))}")

            # subselect
            documents = documents.loc[list(topics_to_load)].sort_index()
            tokens = tokens.loc[documents[DOCUMENT_ID]].sort_index()
            mentions = mentions.loc[documents[DOCUMENT_ID]].sort_index()

        # obtain action mentions
        mentions_action = mentions.loc[mentions[MENTION_TYPE].isin(MENTION_TYPES_ACTION)].copy()

        # remove documents which contain no action mentions
        documents_without_mentions = set(documents[DOCUMENT_ID].unique()) - set(mentions_action.index.get_level_values(DOCUMENT_ID).unique())
        if documents_without_mentions:
            self.logger.info(f"The following documents contain no action mentions and were removed: {', '.join(sorted(documents_without_mentions))}")
        documents = documents.loc[~documents[DOCUMENT_ID].isin(documents_without_mentions)]
        tokens = tokens.loc[documents[DOCUMENT_ID]].sort_index()
        mentions = mentions.loc[~mentions.index.get_level_values(DOCUMENT_ID).isin(documents_without_mentions)]

        # now divide the remainder of mentions
        mentions_time = mentions.loc[mentions[MENTION_TYPE].isin(MENTION_TYPES_TIME)].rename(columns={EVENT: ENTITY})
        mentions_location = mentions.loc[mentions[MENTION_TYPE].isin(MENTION_TYPES_LOCATION)].rename(columns={EVENT: ENTITY})
        mentions_participants = mentions.loc[mentions[MENTION_TYPE].isin(MENTION_TYPES_PARTICIPANTS)].rename(columns={EVENT: ENTITY})
        assert len(mentions) == sum([len(df) for df in [mentions_action, mentions_time, mentions_location, mentions_participants]])

        dataset = Dataset(documents,
                          tokens,
                          mentions_action,
                          mentions_time=mentions_time,
                          mentions_location=mentions_location,
                          mentions_participants=mentions_participants)
        return dataset

component = EcbLoaderStage