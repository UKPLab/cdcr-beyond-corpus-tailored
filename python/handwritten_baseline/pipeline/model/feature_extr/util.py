from typing import Callable, Tuple, List, Any, Optional

import numpy as np
from more_itertools import chunked
from tqdm import tqdm


def batch_cosine_similarity(pairs: List[Tuple],
                            vectors: Any,
                            pairs_transform: Optional[Callable] = None,
                            batch_size: int = 128,
                            desc: str = "") -> np.array:
    """
    Batch-computes cosine similarity for the given pairs and vectors.
    :param pairs: list of mention index pairs
    :param vectors: maps from something to the vector over which cosine sim is to be computed, list-like or dict-like works
    :param pairs_transform: optional transformation for each mention pair index
    :param batch_size:
    :param desc: description, just for progress bar
    :return: column vector with cosine similarities between all pairs
    """
    num_batches = np.ceil(len(pairs) / batch_size)
    description = "Batch-computing cosine similarity"
    if desc:
        description += f" for {desc}"

    iterator = chunked(pairs, batch_size)
    progress_iterator = tqdm(iterator,
                             desc=description,
                             mininterval=10,
                             total=num_batches)

    feature_column = []
    for i,list_of_pairs in enumerate(progress_iterator):
        a_indices, b_indices = zip(*list_of_pairs)

        if pairs_transform is not None:
            a_indices = [pairs_transform(idx) for idx in a_indices]
            b_indices = [pairs_transform(idx) for idx in b_indices]

        mat_a = np.vstack([vectors[idx] for idx in a_indices])
        mat_b = np.vstack([vectors[idx] for idx in b_indices])

        # batch cosine similarity
        num = np.sum(mat_a * mat_b, axis=1)
        denom = np.linalg.norm(mat_a, axis=1) * np.linalg.norm(mat_b, axis=1)
        # catch divbyzero cases, see https://stackoverflow.com/a/37977222
        cs = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
        cs = cs.reshape((-1, 1)).astype(np.float32)
        feature_column.append(cs)
    feature_column = np.vstack(feature_column)
    return feature_column
