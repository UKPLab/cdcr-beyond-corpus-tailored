import pprint
from typing import Optional, List, Tuple, Set, Dict

import numpy as np
from overrides import overrides

from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.model.feature_extr import DEBUG_EXTR
from python.handwritten_baseline.pipeline.model.feature_extr.base_mixin import FeatureExtractorMixin


class DebugFeatureExtractor(FeatureExtractorMixin):
    """
    Returns constant or random feature value for testing purposes.
    """
    def __init__(self,
                 strategy: str,
                 num_features: int,
                 use_cache: bool,
                 features_to_select: Optional[List[str]]):
        super(DebugFeatureExtractor, self).__init__(DEBUG_EXTR, use_cache, features_to_select)
        self.strategy = strategy
        self.num_features = num_features

    @overrides
    def _transform(self, dataset: Dataset, pairs: List[Tuple[Tuple, Tuple]], unique_mentions: Set[Tuple]):
        if self.strategy == "random":
            return np.random.normal(0, 1, (len(pairs), self.num_features))
        elif self.strategy == "zero":
            return np.zeros((len(pairs), self.num_features))
        elif self.strategy == "mix":
            num_zero_features = self.num_features // 2
            print(f"Generating {num_zero_features} zero features and {self.num_features - num_zero_features} random features.")
            zero_features = np.zeros((len(pairs), num_zero_features))
            random_features = np.random.normal(0, 1, (len(pairs), self.num_features - num_zero_features))
            feature_matrix = np.hstack([zero_features, random_features])
            np.random.shuffle(np.transpose(feature_matrix))
            return feature_matrix

    @overrides
    def _get_plain_names_of_all_features(self) -> List[str]:
        return [str(i) for i in range(self.num_features)]

    @classmethod
    @overrides
    def from_params(cls, config: Dict):
        strategy = config.pop("strategy")
        num_features = config.pop("num_features")

        use_cache = config.pop("use_cache", False)
        features_to_select = config.pop("features_to_select", None)
        obj = DebugFeatureExtractor(strategy, num_features, use_cache, features_to_select)
        if config:
            raise ValueError("Leftover configuration: " + pprint.pformat(config))
        return obj