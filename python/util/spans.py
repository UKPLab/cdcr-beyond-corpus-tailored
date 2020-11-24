from bisect import bisect_right
from typing import Tuple, List, Dict, Callable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


def span_matching(tagging_A: List[Tuple[int, int]],
                  tagging_B: List[Tuple[int, int]],
                  keep_A: bool = False) -> Dict[int, int]:
    """
    Assume we have a list of tokens which was tagged with spans by two different approaches A and B.
    This method tries to find the best 1:1 assignment of spans from B to spans from A. If there are more spans in A than
    in B, then spans from B will go unused and vice versa. The quality of an assignment between two spans depends on
    their overlap in tokens. This method removes entirely disjunct pairs of spans.
    Note: In case A contains two (or more) spans of the same length which are a single span in B (or vice versa),
    either of the spans from A may be mapped to the span in B. Which exact span from A is mapped is undefined.
    :param tagging_A: list of spans, defined by (start, end) token offsets (exclusive!), must be non-overlapping!
    :param tagging_B: a second list of spans over the same sequence in the same format as tagging_A
    :param keep_A: include unmatched spans from A as [idx_A, None] in the returned value
    :return: Dict[int,int] where keys are indices from A and values are indices from B
    """
    if not tagging_A:
        return {}
    elif not tagging_B:
        if keep_A:
            return {i:None for i in range(len(tagging_A))}
        else:
            return {}

    # Our cost function is span overlap:
    # (1) the basis: min(end indices) - max(start indices)
    # (2) If two spans are entirely disjunct, the result of (1) will be negative. Use max(0, ...) to set those
    #     cases to 0.
    # (3) High overlap should result in low costs, therefore multiply by -1
    overlap = lambda idx_a, idx_b: -1 * max(0,
                                            (min([tagging_A[idx_a][1],
                                                  tagging_B[idx_b][1]]) -
                                             max([tagging_A[idx_a][0],
                                                  tagging_B[idx_b][0]])))
    cost_matrix = np.fromfunction(np.vectorize(overlap), (len(tagging_A), len(tagging_B)), dtype=np.int)    # type: np.ndarray
    a_indices, b_indices = linear_sum_assignment(cost_matrix)

    # throw away mappings which have no token overlap at all (i.e. costs == 0)
    assignment_costs = cost_matrix[a_indices, b_indices]
    valid_assignments = [i for i in range(len(a_indices)) if assignment_costs[i] < 0]

    # dropped_assignments = len(a_indices) - len(valid_assignments)
    # if dropped_assignments:
    #     self.logger.debug(f"Threw away {dropped_assignments} assignment without token overlap")

    # collect valid assignments
    assignments = {a_idx: b_idx for i, (a_idx, b_idx) in enumerate(zip(a_indices, b_indices)) if i in valid_assignments}

    if keep_A:
        a_to_none = {i: None for i in range(len(tagging_A))}
        a_to_none.update(assignments)
        assignments = a_to_none
    return assignments


def get_monotonous_character_alignment_func(orig: str, longer: str) -> Callable[[int], Optional[int]]:
    """
    Assume you detokenized a sequence of tokens and you applied span detection on the result. You now have character
    offsets into the detokenized sequence, but you want them for the original untokenized sequence. This method returns
    a function which, given a character offset into the detokenized version, returns the character offset in the
    original untokenized sequence. If the given offset does not have a corresponding character in the original sequence,
    `None` is returned. This offset does not necessarily need to conform to token boundaries, but there are other
    methods for fixing this.
    :param orig: `"".join(your_token_sequence)`
    :param longer: `detokenizer(your_token_sequence)` -> must contain the same characters as `orig` with additional
    ones in between! Otherwise, the result of this function is undefined.
    :return: function as described above
    """

    # TODO there might be computationally more efficient approaches, but this one works
    assert len(longer) > len(orig)

    checkpoints_orig = []
    checkpoints_longer = []

    idx_longer = 0
    for idx_orig in range(len(longer)):
        idx_orig_in_bounds = idx_orig < len(orig)

        does_char_match = idx_orig_in_bounds and longer[idx_longer] == orig[idx_orig]
        if does_char_match and len(checkpoints_longer) > 0:
            idx_longer += 1
        else:
            if not does_char_match:
                # create checkpoint for non-matching chars
                checkpoints_longer.append(idx_longer)
                checkpoints_orig.append(None)

                if not idx_orig_in_bounds:
                    # If we reach this point, we are in a situation where longer has superfluous characters at its end.
                    # We only need to create one last checkpoint for this case (which we have done above), therefore
                    # bail out.
                    break

                # advance index for longer until we find a matching char again
                while idx_longer < len(longer) and longer[idx_longer] != orig[idx_orig]:
                    idx_longer += 1

                # If we make it to this point, we have exhausted all characters in s_longer because we did not find a match for
                # the previous character in s. This means the strings are unalignable
                if idx_longer == len(longer):
                    raise ValueError

            # create checkpoint for matching pair
            checkpoints_longer.append(idx_longer)
            checkpoints_orig.append(idx_orig)
            idx_longer += 1


    assert checkpoints_longer[0] == 0   # the first char should always receive a checkpoint

    def func(i: int) -> Optional[int]:
        if i < 0 or i >= len(longer):
            raise IndexError

        # find the closest checkpoint preceding the given integer
        i_of_closest_preceding_checkpoint = bisect_right(checkpoints_longer, i) - 1

        aligned_value = checkpoints_orig[i_of_closest_preceding_checkpoint]
        if aligned_value is not None:
            # if there is a corresponding character in the shorter string, compute its exact index using the checkpoint
            distance_to_checkpoint = i - checkpoints_longer[i_of_closest_preceding_checkpoint]
            return aligned_value + distance_to_checkpoint
        else:
            # if there is no corresponding character, return None
            return None

    return func