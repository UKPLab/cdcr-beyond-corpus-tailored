from typing import Dict, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from python.handwritten_baseline.pipeline.model.data_prep.pipeline_data_input import NO_LABEL


class PredictOnTransformClassifierWrapper(BaseEstimator, TransformerMixin):
    """
    Sklearn expects users to place classifiers as the last step in a pipeline which is why they have a predict method
    and no transform method. In our sklearn pipeline, we train and predict with a mention pair classifier prior to
    clustering the pairs, so we need our classifier to transform. This class wraps a classifier and calls its
    predict(...) method inside transform(...). Calls to fit(...) are forwarded to the classifier.
    We also take apart the X object passed from the previous pipeline stage into the actual X and y parts (nesting makes
    things complicated, sorry).
    """

    def __init__(self, classifier_, classifier_fit_params: Dict):
        """

        :param classifier_:
        :param classifier_fit_params: used for xgboost
        """
        self.classifier_ = classifier_
        self.classifier_fit_params = classifier_fit_params

    @staticmethod
    def _take_apart_X(X: np.array) -> Tuple[np.array, np.array]:
        shape = X.shape
        assert shape[1] >= 2

        # if the pipeline is set up accordingly, the labels should be in the first column and all features in the remaining columns
        actual_y = X[:, 0]
        actual_X = X[:, 1:]

        # Messing up the previous assignments would be catastrophic, so some sanity checks are due:
        # At prediction time, all label values are set to a specific constant far from 0 or 1. Check this.
        is_prediction_time = np.all(actual_y == NO_LABEL)
        if not is_prediction_time:
            # At training time, actual_y should contain boolean values as ints (0 or 1), in particular:
            # - there should be no floats in the label column, so there should be no difference to the same column as integers
            assert np.abs((actual_y.astype(int) - actual_y)).sum() == 0, "Order of steps in preceding sklearn pipeline might be incorrect."
            # - values should never be below 0 or above 1
            assert np.sum(actual_y > 1) == 0 and np.sum(actual_y < 0) == 0, "Order of steps in preceding sklearn pipeline might be incorrect."
        actual_X, actual_y = check_X_y(actual_X, actual_y)
        actual_X = check_array(actual_X)

        return actual_X, actual_y

    def fit(self, X, y=None, **fit_params):
        actual_X, actual_y = self._take_apart_X(X)

        all_fit_params = {**fit_params, **self.classifier_fit_params}
        self.classifier_.fit(actual_X, actual_y, **all_fit_params)
        return self

    def predict_proba(self, X):
        check_is_fitted(self.classifier_)
        actual_X, _ = self._take_apart_X(X)
        return self.classifier_.predict_proba(actual_X)

    def predict(self, X) -> np.array:
        class_probabilities = self.predict_proba(X)

        # The classifier will return an array of class probabilities per instances with shape (num_instances,
        # num_classes). We are only interested in the probability of the True class (probability whether two mentions
        # corefer)
        target_class = True

        assert hasattr(self.classifier_, "classes_")
        search_result = np.nonzero(self.classifier_.classes_ == target_class)[0]
        if search_result.size == 0:
            raise ValueError(f"Could not find class {repr(target_class)} in classes {repr(self.classifier_.classes_)} returned by the classifier.")
        index_of_class = search_result[0]

        mention_coref_probability = class_probabilities[:, index_of_class].reshape((-1, 1))
        return mention_coref_probability

    def transform(self, X, y=None):
        return self.predict(X)