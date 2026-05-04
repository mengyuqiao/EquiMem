"""Trust-discounted retrieval (Section 4.3).

Embedding: scale vectors by sqrt(rho) at index time.
Graph: weight paths by product of rho along edges.
"""

import numpy as np


def trust_discount_embedding(vector: np.ndarray, rho: float) -> np.ndarray:
    """Scale embedding by sqrt(rho) for trust-discounted retrieval.

    Low-rho entries are pulled toward the origin, making them
    less likely to appear in top-k similarity results.
    """
    return np.sqrt(max(0.0, rho)) * vector


def trust_discount_path(edge_rhos: list[float]) -> float:
    """Compute path strength as product of edge rho values.

    A path through any low-trust edge is automatically
    discounted, so low-confidence edges contribute less
    to downstream inference.
    """
    if not edge_rhos:
        return 0.0
    result = 1.0
    for rho in edge_rhos:
        result *= max(0.0, rho)
    return result
