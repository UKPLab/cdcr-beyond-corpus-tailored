import numpy as np

FEATURE_EXTRACTOR_FEATURE_NAME_SEPARATOR = "#"

LEMMA_EXTR = "lemma"
TFIDF_EXTR = "tfidf"
DEBUG_EXTR = "debug"
TIME_EXTR = "time"
LOCATION_EXTR = "location"
SENTENCE_EMBEDDING_EXTR = "sentence-embedding"
ACTION_PHRASE_EMBEDDING_EXTR = "action-phrase-embedding"
WIKIDATA_EMBEDDING_EXTR = "wikidata-embedding"


def fix_all_nan_columns(X: np.array):
    """
    Given a feature matrix, fills all columns which are entirely NaN with zero. This is to circumvent an issue with
    sklearn where applying an Imputer silently removes NaN features entirely. See https://github.com/scikit-learn/scikit-learn/issues/16426
    :param X: feature matrix
    :return: feature matrix will all columns that are entirely NaN filled with zeros
    """
    mask = np.all(np.isnan(X), axis=0)
    X[:, mask] = 0
    return X