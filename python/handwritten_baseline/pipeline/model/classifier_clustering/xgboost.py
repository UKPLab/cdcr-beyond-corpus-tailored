import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier


class ConvenientXGBClassifier(XGBClassifier):
    """
    XGBClassifier which has a `validation_fraction` parameter for splitting off a validation set just like i
    SGDClassifier. In this class it's a fit_params parameter whereas for SGDClassifier it's a constructor argument.
    """

    def _make_validation_split(self, y: np.array, validation_fraction: float):
        """Split the dataset between training set and validation set.
        Largely copied from sklearn.linear_model._stochastic_gradient.BaseSGD._make_validation_split

        Parameters
        ----------
        y : ndarray of shape (n_samples, )
            Target values.
        validation_fraction: float between 0 and 1 to determine the size of the validation split

        Returns
        -------
        validation_mask : ndarray of shape (n_samples, )
            Equal to 1 on the validation set, 0 on the training set.
        """
        if not (0.0 < validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in range (0, 1)")

        n_samples = y.shape[0]
        validation_mask = np.zeros(n_samples, dtype=np.uint8)

        cv = StratifiedShuffleSplit(test_size=validation_fraction, random_state=0)
        idx_train, idx_val = next(cv.split(np.zeros(shape=(y.shape[0], 1)), y))
        if idx_train.shape[0] == 0 or idx_val.shape[0] == 0:
            raise ValueError(
                "Splitting %d samples into a train set and a validation set "
                "with validation_fraction=%r led to an empty set (%d and %d "
                "samples). Please either change validation_fraction, increase "
                "number of samples, or disable early_stopping."
                % (n_samples, self.validation_fraction, idx_train.shape[0],
                   idx_val.shape[0]))

        validation_mask[idx_val] = 1
        return validation_mask.astype(bool)

    def fit(self, X, y, sample_weight=None, base_margin=None, validation_fraction: float = 0.1, eval_metric=None,
            early_stopping_rounds=None, verbose=True, xgb_model=None, sample_weight_eval_set=None, callbacks=None):

        if early_stopping_rounds is not None:
            validation_mask = self._make_validation_split(y, validation_fraction)

            train_X = X[~validation_mask]
            train_y = y[~validation_mask]
            dev_X = X[validation_mask]
            dev_y = y[validation_mask]

            # eval_set: A list of (X, y) tuple pairs to use as validation sets, for which metrics will be computed.
            eval_set = [(dev_X, dev_y)]
        else:
            train_X = X
            train_y = y
            eval_set = None

        return super().fit(train_X, train_y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds,
                           verbose, xgb_model, sample_weight_eval_set, callbacks)