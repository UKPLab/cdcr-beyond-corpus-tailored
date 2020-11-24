from typing import List

from scipy.spatial.distance import cdist
from scipy.stats import stats
from scipy.stats.stats import DescribeResult


def create_feature_names(single_value_features: List[str], distribution_features: List[str]):
    """
    :param single_value_features: names of features modeles by a single value
    :param distribution_features: names of features modeled by min/max/mean/variance
    """
    all_names = [feature_name for feature_name in single_value_features]
    for area in distribution_features:
        for descr in ["mean", "variance", "min", "max"]:
            all_names.append("-".join([area, descr]))
    return all_names


def compute_pairwise_embedding_distance_features(a_mat, b_mat) -> List:
    """
    Computes pairwise embedding distance features.
    :param a_mat:
    :param b_mat:
    :return: Note that the order of values is the same as in `create_feature_names` (mean, variance, min, max)
    """
    if a_mat is None or b_mat is None or a_mat.size == 0 or b_mat.size == 0:
        return [None] * 4
    else:
        dists = cdist(a_mat, b_mat, "cosine")
        if dists.size == 1:
            # scipy would raise "FloatingPointError: invalid value encountered in double_scalars" when calling describe on a 1x1 matrix, so we use this workaround
            return [dists.item(),
                    0,
                    dists.item(),
                    dists.item()]
        else:
            dists_stats = stats.describe(dists, axis=None)  # type: DescribeResult
            return [dists_stats.mean,
                    0 if dists_stats.variance is None else dists_stats.variance,
                    dists_stats.minmax[0],
                    dists_stats.minmax[1]]