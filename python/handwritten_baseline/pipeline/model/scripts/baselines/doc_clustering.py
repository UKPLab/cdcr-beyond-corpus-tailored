import numpy as np
import pandas as pd
import scipy.sparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from python import DOCUMENT_ID, TOKEN, EVENT_ID, PUBLISH_DATE
from python.handwritten_baseline import TIMEX_NORMALIZED_PARSED


class DocumentPublishDateExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        dataset, documents, _ = X[0]

        # Find first temporal expression in each document, remove timezone information and clean them up a bit
        first_temporal_expression = dataset.mentions_time.groupby(DOCUMENT_ID)[TIMEX_NORMALIZED_PARSED].first()
        first_temporal_expression = pd.to_datetime(first_temporal_expression.map(lambda v: v.replace(tzinfo=None), na_action="ignore"), errors="coerce")

        # For documents without temporal expression, fill in the document publish date as a fallback.
        publish_dates = dataset.documents[[DOCUMENT_ID, PUBLISH_DATE]].set_index(DOCUMENT_ID)[PUBLISH_DATE]
        docs_by_time = first_temporal_expression.reindex(publish_dates.index).fillna(publish_dates)

        docs_by_time = (docs_by_time - docs_by_time.min()).dt.total_seconds() / 3600
        return list(docs_by_time.iteritems())


def get_hours_elapsed_from_X(X):
    return np.array([date for _, date in X]).reshape((-1, 1))


class DocumentTextExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        dataset, documents, _ = X[0]

        # look up tokens for documents in X
        tokens_df = dataset.tokens.loc[documents[DOCUMENT_ID], TOKEN]
        docs = []
        for doc_id, df in tokens_df.groupby(DOCUMENT_ID):
            text = " ".join(df.values.tolist())
            docs.append((doc_id, text))
        return docs


class DocumentClustering(BaseEstimator, TransformerMixin):

    def __init__(self, threshold: float, metric: str):
        self.threshold = threshold
        self.metric = metric

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        if scipy.sparse.issparse(X):
            X = X.todense()

        # X is now a matrix of document vectors
        Y = pdist(X, metric=self.metric)
        Z = linkage(Y, method="average")
        clusters = fcluster(Z, t=self.threshold, criterion="distance")
        return clusters.reshape((-1, 1))


def get_text_from_X(X):
    return [text for _, text in X]


def get_doc_id_from_X(X):
    return np.array([doc_id for doc_id, _ in X]).reshape((-1, 1))


class DocumentClusteringPrettifier(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def predict(self, X, y=None):
        # X is a np.array of shape (num_docs, 2) with document identifier and cluster id
        clustering = pd.DataFrame({DOCUMENT_ID: X[:, 0], EVENT_ID: X[:, 1].astype(int)})
        return clustering


def create_doc_clustering_pipeline(threshold: float,
                                   signal: str) -> Pipeline:
    if signal == "tfidf":
        feature_extractor = DocumentTextExtractor()
        pipe_segment = ("tfidf_clustering", Pipeline([
            ("get_text", FunctionTransformer(get_text_from_X)),
            ("tfidf", TfidfVectorizer(tokenizer=str.split, lowercase=True, token_pattern=None, min_df=3, stop_words="english")),
            ("clustering", DocumentClustering(threshold, "cosine"))
        ]))
    elif signal == "time":
        feature_extractor = DocumentPublishDateExtractor()
        pipe_segment = ("temporal_clustering", Pipeline([
            ("get_hours_elapsed_from_X", FunctionTransformer(get_hours_elapsed_from_X)),
            ("clustering", DocumentClustering(threshold, "cityblock"))
        ]))
    else:
        raise ValueError

    p = Pipeline([("feature_extractor", feature_extractor),
                  ("identifier_and_cluster_id", FeatureUnion([
                      ("get_doc_id", FunctionTransformer(get_doc_id_from_X)),
                      pipe_segment
                  ])),
                  ("prettify", DocumentClusteringPrettifier())])

    return p
