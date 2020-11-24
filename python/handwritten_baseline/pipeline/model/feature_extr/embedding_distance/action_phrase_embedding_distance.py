import pprint
from typing import Optional, List, Tuple, Set, Dict

import numpy as np
from overrides import overrides

from python.handwritten_baseline import ACTION_PHRASE_EMBEDDINGS
from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.model.feature_extr import ACTION_PHRASE_EMBEDDING_EXTR
from python.handwritten_baseline.pipeline.model.feature_extr.base_mixin import FeatureExtractorMixin
from python.handwritten_baseline.pipeline.model.feature_extr.util import batch_cosine_similarity

ACTION_PHRASE = "action-phrase"


class ActionPhraseEmbeddingDistanceFeature(FeatureExtractorMixin):

    def __init__(self,
                 use_cache: bool,
                 features_to_select: Optional[List[str]]):
        super(ActionPhraseEmbeddingDistanceFeature, self).__init__(ACTION_PHRASE_EMBEDDING_EXTR, use_cache, features_to_select)

    @overrides
    def _transform(self, dataset: Dataset, pairs: List[Tuple[Tuple, Tuple]], unique_mentions: Set[Tuple]):
        # obtain embeddings
        assert dataset.has(ACTION_PHRASE_EMBEDDINGS)
        action_phrase_embeddings = dataset.get(ACTION_PHRASE_EMBEDDINGS)  # type: Tuple[Dict[Tuple[str, int], int], np.array]
        embedding_index, embedding_mat = action_phrase_embeddings

        pairs_transform = lambda idx: embedding_index[idx]

        feature_column = batch_cosine_similarity(pairs, embedding_mat, pairs_transform=pairs_transform, desc=self.name)
        feature_matrix = feature_column.reshape((-1, 1))
        return feature_matrix

    @overrides
    def _get_plain_names_of_all_features(self) -> List[str]:
        return [ACTION_PHRASE]

    @classmethod
    @overrides
    def from_params(cls, config: Dict):
        use_cache = config.pop("use_cache", False)
        features_to_select = config.pop("features_to_select", None)
        obj = ActionPhraseEmbeddingDistanceFeature(use_cache, features_to_select)
        if config:
            raise ValueError("Leftover configuration: " + pprint.pformat(config))
        return obj
