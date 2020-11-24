import copy
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Set, Dict

import numpy as np
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin

from python.handwritten_baseline.pipeline.data.base import Dataset
from python.handwritten_baseline.pipeline.model.feature_extr import FEATURE_EXTRACTOR_FEATURE_NAME_SEPARATOR
from python.util.util import get_dict_hash


class FeatureExtractorMixin(BaseEstimator, TransformerMixin):
    """
    Abstract class for custom mention pair features.
    See https://scikit-learn.org/0.19/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py
    """

    def __init__(self,
                 name: str,
                 use_cache: bool,
                 features_to_select: Optional[List[str]]):
        """

        :param name: name of this feature extractor
        :param use_cache: enable caching for transform() calls
        :param features_to_select: The names of features to return in transform() -> these should not be prefixed with
                                   the name of the feature extractor! If None, all features will be returned.
        """
        self.name = name
        self.use_cache = use_cache
        self.features_to_select = features_to_select

    @property
    def dtype(self):
        return np.dtype("float32")

    @staticmethod
    def from_np_array_back_to_list_of_tuples(pairs: np.array) -> List[Tuple[Tuple, Tuple]]:
        """
        Convert pairs of mention identifiers from a numpy array back into the list of tuples of tuples format we have
        been using for features all the time. This method makes strong assumptions over the input (and thereby the
        whole dataset) format, which is good. If it leads to a crash, we're in trouble.
        :param pairs:
        :return:
        """
        return [((pair[0], int(pair[1])), (pair[2], int(pair[3]))) for pair in pairs]

    def fit(self, X, y=None):
        dataset, pairs, labels, unique_mentions = X
        self._fit(dataset, FeatureExtractorMixin.from_np_array_back_to_list_of_tuples(pairs), unique_mentions)
        return self

    def _fit(self, dataset: Dataset, pairs: List[Tuple[Tuple, Tuple]], unique_mentions: Set[Tuple]):
        pass

    def transform(self, X: Tuple):
        dataset, pairs, labels, unique_mentions = X

        if self.use_cache:
            # We want to cache feature transformation outputs similar to what is asked for / proposed here:
            # (1) https://mail.python.org/pipermail/scikit-learn/2017-August/001828.html
            # (2) https://gist.github.com/jnothman/019d594d197c98a3d6192fa0cb19c850
            # We cannot implement the caching 1:1 as in the github gist because our feature extractors have constructor
            # parameters which change the output of transform(), i.e. we want one cache for each set of parameters. To
            # do this conveniently, we take the __dict__ of a feature extractor, remove irrelevant entries and hash the
            # result. Irrelevant entries are the features to select (read-only modification) and any data-dependent
            # attributes ending with an underscore (see https://scikit-learn.org/stable/developers/develop.html#estimated-attributes)
            attrs = copy.deepcopy(self.__dict__)
            attrs = {k:v for k,v in attrs.items() if not k.endswith("_") and not k in ["name", "features_to_select"]}
            cache_key = get_dict_hash(attrs)
            cache_location = Path(tempfile.gettempdir()) / f"feature_{self.name}_{cache_key}"
            memory = Memory(cache_location, verbose=0)

            feature_matrix = memory.cache(self._transform)(dataset, FeatureExtractorMixin.from_np_array_back_to_list_of_tuples(pairs), unique_mentions)
        else:
            feature_matrix = self._transform(dataset, FeatureExtractorMixin.from_np_array_back_to_list_of_tuples(pairs), unique_mentions)

        # filter feature matrix according to feature selection
        if self.features_to_select:
            all_feature_names = self._get_plain_names_of_all_features()

            # sanity check: we can only select what we can extract
            for fname in self.features_to_select:
                if not fname in all_feature_names:
                    raise ValueError("Cannot select unknown feature name: " + fname)

            mask = np.array([fname in self.features_to_select for fname in all_feature_names])
            filtered_feature_matrix = feature_matrix[:, mask]
            return filtered_feature_matrix
        else:
            return feature_matrix

    def _transform(self, dataset: Dataset, pairs: List[Tuple[Tuple, Tuple]], unique_mentions: Set[Tuple]):
        raise NotImplementedError

    def get_feature_names(self) -> List[str]:
        """
        Returns the names of all features this feature extractor will extract (== not all features, only the ones
        specified in the constructor), prefixed with the name of this feature.
        extractor.
        :return:
        """
        feature_names = self.features_to_select if self.features_to_select is not None else self._get_plain_names_of_all_features()
        assert not any(FEATURE_EXTRACTOR_FEATURE_NAME_SEPARATOR in fname for fname in feature_names)

        feature_names_with_extractor_prefix = [self.name + FEATURE_EXTRACTOR_FEATURE_NAME_SEPARATOR + fname for fname in
                                               feature_names]
        return feature_names_with_extractor_prefix

    def _get_plain_names_of_all_features(self) -> List[str]:
        """
        Returns the names of all features this feature extractor can extract.
        :return:
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, config: Dict):
        raise NotImplementedError