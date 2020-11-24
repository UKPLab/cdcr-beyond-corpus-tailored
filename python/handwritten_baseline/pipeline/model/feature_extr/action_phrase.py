import pprint
from typing import Optional, List, Tuple, Set, Dict

import numpy as np
from overrides import overrides
from textdistance import mlipns, levenshtein

from python import SENTENCE_IDX, TOKEN_IDX_FROM, TOKEN_IDX_TO, TOKEN
from python.handwritten_baseline import LEMMA
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.model.feature_extr import LEMMA_EXTR
from python.handwritten_baseline.pipeline.model.feature_extr.base_mixin import FeatureExtractorMixin


class LemmaFeatureExtractor(FeatureExtractorMixin):
    """
    Checks if the token spans of two mentions match (exact or fuzzy matching).
    """

    def __init__(self,
                 use_cache: bool,
                 features_to_select: Optional[List[str]]):
        super(LemmaFeatureExtractor, self).__init__(LEMMA_EXTR, use_cache, features_to_select)

    @staticmethod
    def get_mention_tokens(idx, mentions, tokens):
        doc_id, _ = idx
        mention = mentions.loc[idx]
        mention_tokens = tokens.loc[
            (doc_id, mention[SENTENCE_IDX], slice(mention[TOKEN_IDX_FROM], mention[TOKEN_IDX_TO] - 1))]
        return mention_tokens

    @staticmethod
    def get_lemma(idx, mentions, tokens):
        mention_tokens = LemmaFeatureExtractor.get_mention_tokens(idx, mentions, tokens)
        lemma = mention_tokens[LEMMA].str.cat(sep=" ")
        return lemma

    @staticmethod
    def get_surface_form(idx, mentions, tokens):
        mention_tokens = LemmaFeatureExtractor.get_mention_tokens(idx, mentions, tokens)
        surface_form = mention_tokens[TOKEN].str.cat(sep=" ")
        return surface_form

    @overrides
    def _transform(self, dataset: Dataset, pairs: List[Tuple[Tuple, Tuple]], unique_mentions: Set[Tuple]):
        mentions = dataset.mentions_action
        tokens = dataset.tokens

        precomputed = {}
        for mention in unique_mentions:
            assert len(mention) == 2  # (doc_id, mention_id)
            precomputed[mention] = {"lemma": self.get_lemma(mention, mentions, tokens),
                                    "surface-form": self.get_surface_form(mention, mentions, tokens)}

        raw_features = []
        for a_idx, b_idx in pairs:
            is_surface_form_identical = precomputed[a_idx]["surface-form"] == precomputed[b_idx]["surface-form"]
            is_lemma_identical = precomputed[a_idx]["lemma"] == precomputed[b_idx]["lemma"]

            fuzzy_distances = [func(precomputed[a_idx]["surface-form"], precomputed[b_idx]["surface-form"]) for func in [mlipns, levenshtein]]
            raw_features.append([is_surface_form_identical,
                                 is_lemma_identical,
                                 *fuzzy_distances])

        return np.array(raw_features, dtype=self.dtype)

    @overrides
    def _get_plain_names_of_all_features(self):
        return ["is-surface-form-identical",
                "is-lemma-identical",
                "surface-form-mlinps-distance",
                "surface-form-levenshtein-distance"]

    @classmethod
    @overrides
    def from_params(cls, config: Dict):
        use_cache = config.pop("use_cache", False)
        features_to_select = config.pop("features_to_select", None)
        obj = LemmaFeatureExtractor(use_cache, features_to_select)
        if config:
            raise ValueError("Leftover configuration: " + pprint.pformat(config))
        return obj