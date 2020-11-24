import string
from typing import Dict, List

import numpy as np
import pandas as pd

from python import TOKEN_IDX_FROM, TOKEN_IDX_TO, SENTENCE_IDX, DOCUMENT_ID, PUBLISH_DATE, TOKEN
from python.handwritten_baseline import WIKIDATA_QID, DBPEDIA_URI, MENTION_TYPE_COARSE, PARTICIPANTS, LOCATION, TIME
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage


class MentionMaskingStage(BaselineDataProcessorStage):
    """
    Replaces all tokens which are marked as action, participant, time or location with dummy text. Additionally sets all
    columns of these event components which were filled with extra bits to None.
    """

    def __init__(self, pos, config, config_global, logger):
        super(MentionMaskingStage, self).__init__(pos, config, config_global, logger)
        self._mask_what = config["event_components_to_mask"]

    @staticmethod
    def _fill_columns_with_na(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for c in columns:
            if not c in columns:
                continue
            df[c] = pd.Series(None, dtype=df[c].dtype)
        return df

    @staticmethod
    def _mask_tokens(tokens: pd.DataFrame, mentions: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces token spans in a mentions dataframe with random tokens. The replacement tokens all have the same length
        in characters and are randomly sampled from [a-zA-Z]. Previously, we had replaced tokens with "action_1",
        "action_2", ... here, but this results in high lexical overlap between spans, which introduces a new kind of
        bias on its own. Replacing token spans with random real tokens (replacing locations with "New York", "London",
        ...) might also inadvertently create associations between mentions which are not supposed to be there.
        :param tokens:
        :param mentions:
        :return:
        """
        assert all(c in mentions.columns for c in [TOKEN_IDX_FROM, TOKEN_IDX_TO, SENTENCE_IDX])

        # sample tokens of fixed length - in batches, reproducibly, until we have as many as we need
        number_of_random_tokens_needed = (mentions[TOKEN_IDX_TO] - mentions[TOKEN_IDX_FROM]).sum()
        token_character_length = 5  # we can hardcode this... if we run out of random tokens of length 5 for a dataset, we will have huge problems elsewhere anyway
        random = np.random.RandomState(seed=0)
        random_token_pool = set()
        while len(random_token_pool) < number_of_random_tokens_needed:
            random_characters = random.choice(list(string.ascii_letters), (50, token_character_length), replace=True)
            random_tokens = ["".join(characters) for characters in random_characters]
            random_token_pool |= set(random_tokens)

        for doc_id, mentions_in_doc in mentions.groupby(DOCUMENT_ID):
            for idx, mention in mentions_in_doc.iterrows():
                for token_idx in range(mention[TOKEN_IDX_FROM], mention[TOKEN_IDX_TO]):
                    tokens.at[(doc_id, mention[SENTENCE_IDX], token_idx), TOKEN] = random_token_pool.pop()
        return tokens

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        # masking an event component entails replacing all mention tokens with a random dummy token, followed by
        # nulling all additional preprocessing columns to keep features from working off of those
        if "action" in self._mask_what:
            mentions_action = dataset.mentions_action
            tokens = self._mask_tokens(dataset.tokens, mentions_action)
            mentions_action = self._fill_columns_with_na(mentions_action, [DBPEDIA_URI, WIKIDATA_QID])

            dataset.mentions_action = mentions_action
            dataset.tokens = tokens

        if "participants" in self._mask_what:
            mentions_participants = dataset.mentions_participants
            semantic_roles = dataset.semantic_roles
            tokens = self._mask_tokens(dataset.tokens, mentions_participants)

            # remove all participant mentions and corresponding SRL entries
            mentions_participants = mentions_participants.iloc[0:0]
            semantic_roles = semantic_roles.loc[semantic_roles[MENTION_TYPE_COARSE] != PARTICIPANTS]

            dataset.mentions_participants = mentions_participants
            dataset.semantic_roles = semantic_roles
            dataset.tokens = tokens

        if "location" in self._mask_what:
            mentions_location = dataset.mentions_location
            semantic_roles = dataset.semantic_roles
            tokens = self._mask_tokens(dataset.tokens, mentions_location)

            # remove all location mentions and corresponding SRL entries
            mentions_location = mentions_location.iloc[0:0]
            semantic_roles = semantic_roles.loc[semantic_roles[MENTION_TYPE_COARSE] != LOCATION]

            dataset.mentions_location = mentions_location
            dataset.semantic_roles = semantic_roles
            dataset.tokens = tokens

        if "time" in self._mask_what:
            mentions_time = dataset.mentions_time
            semantic_roles = dataset.semantic_roles
            tokens = self._mask_tokens(dataset.tokens, mentions_time)

            # remove all temporal mentions and corresponding SRL entries
            mentions_time = mentions_time.iloc[0:0]
            semantic_roles = semantic_roles.loc[semantic_roles[MENTION_TYPE_COARSE] != TIME]

            dataset.mentions_time = mentions_time
            dataset.semantic_roles = semantic_roles
            dataset.tokens = tokens

        if "publish_date" in self._mask_what:
            documents = dataset.documents
            if PUBLISH_DATE in documents.columns:
                documents.drop(columns=PUBLISH_DATE, inplace=True)
            dataset.documents = documents

        return dataset


component = MentionMaskingStage
