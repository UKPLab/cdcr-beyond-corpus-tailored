import pprint
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, ClusterMixin

from python import DOCUMENT_ID, MENTION_ID


class ScipyClustering(BaseEstimator, ClusterMixin):
    """
    Runs agglomerative clustering. Mentions can only participate in one cluster each.
    :param threshold: clustering threshold
    :param linkage_method: linkage method, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    :param cluster_criterion: clustering criterion, see https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.fcluster.html
    :param cluster_depth: depth for 'inconsistent' criterion, irrelevant for the other criteria
    :return: the resulting clustering
    """

    def __init__(self,
                 threshold: float,
                 linkage_method: str,
                 cluster_criterion: str,
                 cluster_depth: int,
                 hard_document_clusters: Iterable[Iterable[str]] = None):
        self.threshold = threshold
        self.linkage_method = linkage_method
        self.cluster_criterion = cluster_criterion
        self.cluster_depth = cluster_depth
        self.hard_document_clusters = hard_document_clusters

    def fit(self, X, y=None):
        return self

    def predict(self, X: np.ndarray):
        """
        We take the following steps here:
          - convert X into dataframe of records
          - convert dataframe into pairwise distance matrix, efficiently
          - performing clustering and return predictions
        :param X:
        :return: clustering
        """

        assert X.shape[1] == 5  # [doc_id, mention_id, doc_id, mention_id, coref distance]

        # create records dataframe
        doc_id_a = DOCUMENT_ID + "_a"
        doc_id_b = DOCUMENT_ID + "_b"
        mention_id_a = MENTION_ID + "_a"
        mention_id_b = MENTION_ID + "_b"
        df = pd.DataFrame.from_records(X,columns=[doc_id_a, mention_id_a, doc_id_b, mention_id_b, "prob_coref"])
        df[mention_id_a] = df[mention_id_a].astype(int)
        df[mention_id_b] = df[mention_id_b].astype(int)

        # to cluster, we need to flip the predictions from "mention pair coref probability" into "mention pair distance"
        df["distance"] = 1 - (df["prob_coref"].astype(np.float32))
        df.drop(columns="prob_coref", inplace=True)

        # at this point, insert hard document clustering if available
        if self.hard_document_clusters is not None:
            document_clusters = pd.Series({doc_id: cluster_id for cluster_id, cluster in enumerate(self.hard_document_clusters) for doc_id in cluster})
            document_clusters.index.name = DOCUMENT_ID
            # assert that the given document clustering is complete w.r.t. the mentions we are working on
            for col in [doc_id_a, doc_id_b]:
                assert all(doc_id in document_clusters for doc_id in df[col].unique())
            # determine which mention pairs originate from different document clusters and set their distance to some
            # arbitrarily high value outside the usual distance domain (0.0 - 1.0)
            to_be_masked = df[doc_id_a].map(document_clusters) != df[doc_id_b].map(document_clusters)
            df["distance"].where(~to_be_masked, 1000, inplace=True)

        # make a dataframe of all unique (doc_id, mention_id) pairs
        all_mention_indices = pd.DataFrame(
            np.vstack([df[[doc_id_a, mention_id_a]].values, df[[doc_id_b, mention_id_b]].values]),
            columns=[DOCUMENT_ID, MENTION_ID])
        unique_mention_indices = all_mention_indices.drop_duplicates().sort_values(by=[DOCUMENT_ID, MENTION_ID])

        # using this dataframe, create a mapping from (doc_id, mention_id) pairs to integers -> this way, we can keep
        # track of which row/col in our pairwise distance matrix belongs to which mention pair
        index = pd.MultiIndex.from_frame(unique_mention_indices)

        # map mention indices to indices in the distance matrix
        find_mention_id_in_index = lambda r: index.get_loc(tuple(r))
        rows = df[[doc_id_a, mention_id_a]].apply(find_mention_id_in_index, axis=1).values
        cols = df[[doc_id_b, mention_id_b]].apply(find_mention_id_in_index, axis=1).values

        # Note that we add 1 here. The default value in scipy sparse matrices is 0, which we want to replace with
        # 1 in a later step. 0 can be an actual mention distance returned by our model, so we shift all values by 1 to
        # retain these "true" zeroes.
        distances = df["distance"] + 1
        n = len(index)
        mat = csr_matrix((distances, (rows, cols)), shape=(n, n))
        mat_dist_symmetric = (mat + mat.transpose()).todense()

        # set diagonal to "zero" -> distance of mention to itself should be 0
        mat_dist_symmetric += np.eye(n)

        # We want a symmetric distance matrix with distance 1 in all places where we do NOT have a pairwise distance
        # computed. There is no way to set the default value in a scipy sparse matrix, so we start with a dense matrix
        # of ones and fill in our actual distance values.
        mat_dist = np.ones((n, n))
        mat_dist = np.where(mat_dist_symmetric > 0, mat_dist_symmetric - 1, mat_dist)

        mat_dist_condensed = squareform(mat_dist)

        # apply clustering
        Z = linkage(mat_dist_condensed, method=self.linkage_method)
        clusters = fcluster(Z, t=self.threshold, criterion=self.cluster_criterion, depth=self.cluster_depth)
        system_prediction = pd.Series(clusters, index=index).sort_index()

        return system_prediction

    @classmethod
    def from_params(cls, config: Dict):
        threshold = config.pop("threshold", 0.5)
        linkage_method = config.pop("linkage_method", "average")
        cluster_criterion = config.pop("cluster_criterion", "inconsistent")
        cluster_depth = config.pop("cluster_depth", 1)

        obj = ScipyClustering(threshold,
                              linkage_method,
                              cluster_criterion,
                              cluster_depth)

        if config:
            raise ValueError("Unused config entries: " + pprint.pformat(config))

        return obj