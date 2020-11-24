import time
from typing import Dict, Optional

import pandas as pd
from mosestokenizer import MosesDetokenizer
from requests import HTTPError
from tqdm import tqdm

from python import DOCUMENT_ID, TOKEN, TOKEN_IDX_FROM, TOKEN_IDX_TO, CHARS_START, CHARS_END, TOKEN_IDX, SENTENCE_IDX, \
    MENTION_ID
from python.handwritten_baseline import MENTION_TEXT, PARTICIPANTS, ACTION, TIME, LOCATION, OTHER, MENTION_TYPE_COARSE
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage, MODE_EXTEND, \
    MODE_INTERSECT
from python.handwritten_baseline.pipeline.data.processing import left_join_predictions, outer_join_predictions
from python.util.spans import get_monotonous_character_alignment_func


class BaseEntityLinkingStage(BaselineDataProcessorStage):
    """
    Runs a generic entity linker (though mostly geared towards DBPedia Spotlight) on each document. Mention spans found
    by the entity linker are mapped to those determined in previous data processing steps as best as possible.
    TODO move more of the spotlight-specific stuff into DbPediaSpotlight class to untangle this
    """

    def __init__(self,
                 pos,
                 config,
                 config_global,
                 logger,
                 entity_linker_name: str):
        super(BaseEntityLinkingStage, self).__init__(pos, config, config_global, logger)

        self._entity_linker_name = entity_linker_name
        self._entity_linker_cache = self._provide_cache(self._entity_linker_name, human_readable=False)

        # note: we are not using NLTK TreebankWordDetokenizer here, because that one replaces double quotes with two
        # single quotes which makes mappings between the tokenized and detokenized strings needlessly complicated
        self._detokenizer = MosesDetokenizer("en")

    def _query_entity_linker(self, doc_detokenized: str, live_objects: Dict) -> Optional[object]:
        raise NotImplementedError

    def _convert_el_response_to_dataframe(self, obj, live_objects: Dict) -> pd.DataFrame:
        """

        :param obj: the response from the entity linker
        :return: A dataframe with one row per mention. There must be columns CHARS_START and CHARS_END. Any additional
        columns are kept and will end up in the final dataframe(s) of mentions.
        """
        raise NotImplementedError

    def _get_waiting_time_between_requests_seconds(self, live_objects: Dict) -> int:
        raise NotImplementedError

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        errors = False
        mentions_el = []

        time_of_last_query = 0
        for doc_id, df in tqdm(dataset.tokens.groupby(DOCUMENT_ID),
                                                desc=f"EL with {self._entity_linker_name} on documents",
                                                mininterval=10):
            doc_conjoined = "".join(df[TOKEN].values)
            doc_detokenized = self._detokenizer(df[TOKEN].values.tolist())

            # obtain response from entity linker: from cache if possible, otherwise create it fresh
            if not doc_detokenized in self._entity_linker_cache:
                now = time.time()
                try:
                    # apply rate limiting: make sure at least self._wait_between_requests_seconds seconds are between each request
                    time_to_sleep = max(0, self._get_waiting_time_between_requests_seconds(live_objects) - (now - time_of_last_query))
                    time.sleep(time_to_sleep)
                    response = self._query_entity_linker(doc_detokenized, live_objects)
                except (ValueError, HTTPError) as e:
                    self.logger.error(f"Entity linking error for {doc_id}", e)
                    errors = True
                    continue
                finally:
                    time_of_last_query = now

                self._entity_linker_cache[doc_detokenized] = response
            else:
                response = self._entity_linker_cache[doc_detokenized]

            if response is None:
                self.logger.info(f"No entities found for {doc_id}.")
                continue
            response_df = self._convert_el_response_to_dataframe(response, live_objects)

            # we first need to map the detokenized character offsets into our tokenized character offsets
            get_alignment = get_monotonous_character_alignment_func(doc_conjoined, doc_detokenized)
            response_df[CHARS_START] = response_df[CHARS_START].map(get_alignment)
            # we need to work around exclusive span boundaries here
            response_df[CHARS_END] = (response_df[CHARS_END] -1).map(get_alignment) + 1

            # now, we need to move from character offsets to tokens:
            # start offsets: the first token is associated with character 0, the second token with len(token[0]) and so on
            token_start_offsets = df[TOKEN].str.len().cumsum().shift(1, fill_value=0)
            response_df = response_df.merge(token_start_offsets.reset_index(), left_on=CHARS_START, right_on=TOKEN)
            response_df = response_df.drop(columns=[CHARS_START, TOKEN, SENTENCE_IDX, DOCUMENT_ID]).rename(columns={TOKEN_IDX: TOKEN_IDX_FROM})
            # end offsets: We work with exclusive boundaries. If a mention lies at the end of a sentence, then its
            # TOKEN_IDX_TO needs to be +1 the index of the last token in the sentence (basically going out of bounds).
            token_end_offsets = df[TOKEN].str.len().cumsum()
            response_df = response_df.merge(token_end_offsets.reset_index(), left_on=CHARS_END, right_on=TOKEN)
            response_df = response_df.drop(columns=[CHARS_END, TOKEN]).rename(columns={TOKEN_IDX: TOKEN_IDX_TO})
            response_df[TOKEN_IDX_TO] = response_df[TOKEN_IDX_TO] + 1   # here we +1 the token index for correct exclusive boundaries

            # final dataframe format:
            #   - index: doc_id, mention_id
            #   - values: all the things we want to keep: support, types, similarityScore, percentageOfSecondsRank, dbpedia-uri
            mentions_el_in_doc = response_df.reset_index().rename(columns={"index": MENTION_ID}).set_index([DOCUMENT_ID, MENTION_ID])
            mentions_el.append(mentions_el_in_doc)
        if errors:
            raise ValueError("Stopping because there were errors in the process.")

        mentions_el = pd.concat(mentions_el)

        # remove invalid spans, if any exist TODO fix the actual problem which is causing them
        mentions_el_valid = mentions_el.loc[mentions_el[TOKEN_IDX_FROM] < mentions_el[TOKEN_IDX_TO]]
        if len(mentions_el_valid) < len(mentions_el):
            self.logger.warning(f"Removed {len(mentions_el) - len(mentions_el_valid)} invalid mention(s) after DBpedia entity linking")
        mentions_el = mentions_el_valid

        if not self.mode in [MODE_INTERSECT, MODE_EXTEND]:
            raise ValueError

        # set coarse entity type for each predicted entity mention
        coarse_type_to_dbpedia_type = {ACTION: "DBpedia:Event",
                                       PARTICIPANTS: "DBpedia:Agent",
                                       LOCATION: "DBpedia:Place",
                                       TIME: "DBpedia:TimePeriod"}
        for coarse_type, dbo_type in coarse_type_to_dbpedia_type.items():
            mentions_el.loc[mentions_el["types"].str.contains(dbo_type), MENTION_TYPE_COARSE] = coarse_type
        mentions_el[MENTION_TYPE_COARSE] = mentions_el[MENTION_TYPE_COARSE].fillna(OTHER)

        # Enrich all gold mentions with new info from entity linking
        coarse_type_to_dataset_attr = {ACTION: "mentions_action",
                                       PARTICIPANTS: "mentions_participants",
                                       LOCATION: "mentions_location",
                                       TIME: "mentions_time",
                                       OTHER: "mentions_other"}
        for coarse_type, attr in sorted(coarse_type_to_dataset_attr.items()):
            mentions_el_of_coarse_type = mentions_el.loc[mentions_el[MENTION_TYPE_COARSE] == coarse_type].drop(columns=MENTION_TYPE_COARSE)

            # If the dataset did not contain any mentions of this type, simply assign all predicted mentions. Otherwise
            # left-join all the new columns produced by the entity linking to the gold annotations. We make sure only
            # to join entities which match the type of the gold annotations. Otherwise "The Real Housewives of Beverly
            # Hills" will be joined to "in Beverly Hills", which causes more trouble than necessary.
            dataset_mentions = getattr(dataset, attr, None)
            if dataset_mentions is None:
                new_dataset_mentions = mentions_el_of_coarse_type
            else:
                columns_keep_gold = dataset_mentions.columns
                columns_keep_system = mentions_el.columns.drop([TOKEN_IDX_FROM, TOKEN_IDX_TO, SENTENCE_IDX, MENTION_TEXT, MENTION_TYPE_COARSE])
                new_dataset_mentions = left_join_predictions(dataset_mentions, mentions_el_of_coarse_type, columns_keep_gold, columns_keep_system)
            setattr(dataset, attr, new_dataset_mentions)

        if self.mode == MODE_INTERSECT:
            self.logger.info("Intersected new annotations with dataset from previous pipeline stages.")
        elif self.mode == MODE_EXTEND:
            self.logger.info("Extending dataset entities with those found during entity linking...")

            # add all non-overlapping mentions found via entity linking to the dataset
            mentions_el_to_add = outer_join_predictions(mentions_el, dataset).copy()

            for coarse_type, attr in coarse_type_to_dataset_attr.items():
                # skipping the extension for actions is of crucial importance here, otherwise we would be adding
                # additional event mentions to the dataset!
                if coarse_type == ACTION:
                    continue
                mentions_el_to_add_of_coarse_type = mentions_el_to_add.loc[mentions_el_to_add[MENTION_TYPE_COARSE] == coarse_type].drop(columns=MENTION_TYPE_COARSE)

                dataset_mentions = getattr(dataset, attr, None)
                assert dataset_mentions is not None     # this can't be since we must have assigned something in the similar loop above

                new_dataset_mentions = pd.concat([dataset_mentions, mentions_el_to_add_of_coarse_type]).sort_index()
                setattr(dataset, attr, new_dataset_mentions)

        # assert that there are no "backwards spans", this has caused issues way too many times...
        for attr in coarse_type_to_dataset_attr.values():
            mentions_df = getattr(dataset, attr)
            assert mentions_df.loc[mentions_df[TOKEN_IDX_FROM] >= mentions_df[TOKEN_IDX_TO]].empty

        # make sure to add the mention text to each mention
        def get_mention_text_from_mention(row: pd.Series) -> str:
            return " ".join(dataset.tokens.loc[(row.name[0], row[SENTENCE_IDX], slice(row[TOKEN_IDX_FROM], row[TOKEN_IDX_TO] - 1)), TOKEN].values)
        dataset.mentions_action[MENTION_TEXT] = dataset.mentions_action.apply(get_mention_text_from_mention, axis=1)
        dataset.mentions_participants[MENTION_TEXT] = dataset.mentions_participants.apply(get_mention_text_from_mention, axis=1)
        dataset.mentions_time[MENTION_TEXT] = dataset.mentions_time.apply(get_mention_text_from_mention, axis=1)
        dataset.mentions_location[MENTION_TEXT] = dataset.mentions_location.apply(get_mention_text_from_mention, axis=1)
        if dataset.mentions_other is not None:
            dataset.mentions_other[MENTION_TEXT] = dataset.mentions_other.apply(get_mention_text_from_mention, axis=1)

        return dataset
