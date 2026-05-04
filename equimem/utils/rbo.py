"""Rank-Biased Overlap (RBO) distance.

Implements the RBO complement used in rho_align (Eq. 7).
RBO is robust to partial-list overlap, unlike Kendall's tau.

Reference: Webber et al., "A Similarity Measure for Indefinite Rankings", 2010.
"""

import numpy as np


def rbo_distance(list1: list, list2: list, p: float = 0.9) -> float:
    """Compute 1 - RBO(list1, list2).

    Args:
        list1: First ranked list.
        list2: Second ranked list.
        p: Persistence parameter (default 0.9, top-weighted).

    Returns:
        Distance in [0, 1]. 0 = identical rankings, 1 = disjoint.
    """
    if not list1 and not list2:
        return 0.0
    if not list1 or not list2:
        return 1.0

    k = min(len(list1), len(list2))
    rbo_sum = 0.0

    for d in range(1, k + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        overlap = len(set1 & set2)
        agreement = overlap / d
        rbo_sum += p ** (d - 1) * agreement

    rbo_val = (1 - p) * rbo_sum
    return 1.0 - rbo_val
