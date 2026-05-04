"""Relation compatibility estimation.

Derives R_compat from edge-type co-occurrence statistics
in the graph (no external ontology required).
"""


def relation_compatibility(r1: str, r2: str, graph) -> float:
    """Compute compatibility score between two relation types.

    Based on co-occurrence: how often r1 and r2 appear on edges
    sharing a common endpoint in the graph.

    Args:
        r1: First relation type.
        r2: Second relation type (or composite path type).
        graph: Graph object with edge statistics.

    Returns:
        Compatibility score in [0, 1].
    """
    if r1 == r2:
        return 1.0

    # Handle composite path types (e.g., "located_in->contains")
    if "->" in r2:
        components = r2.split("->")
        scores = [relation_compatibility(r1, c, graph) for c in components]
        return max(scores)

    cooccur = graph.get_cooccurrence(r1, r2)
    total_r1 = graph.get_relation_count(r1)
    total_r2 = graph.get_relation_count(r2)

    if total_r1 == 0 or total_r2 == 0:
        return 0.0

    # Jaccard-like: co-occurrence / union
    return cooccur / (total_r1 + total_r2 - cooccur + 1e-8)
