"""Bidirectional BFS for graph path enumeration.

Used by s_path to find existing multi-hop paths
between two nodes, bounded by L_max = ceil(log|V|).
"""

from dataclasses import dataclass
from collections import deque


@dataclass
class GraphPath:
    """A multi-hop path in the knowledge graph."""
    nodes: list[str]
    edges: list[tuple[str, str, str]]
    relation_type: str  # composite type string

    @property
    def length(self) -> int:
        return len(self.edges)


def bidirectional_bfs(graph, source: str, target: str,
                      max_depth: int = 4) -> list[GraphPath]:
    """Find all paths from source to target up to max_depth.

    Uses bidirectional BFS for efficiency: O(d^(L/2)) instead
    of O(d^L), where d is average degree.

    Args:
        graph: Graph object with .get_neighbors(node) method.
        source: Start node.
        target: End node.
        max_depth: Maximum path length (default: ceil(log|V|)).

    Returns:
        List of GraphPath objects.
    """
    if source == target:
        return []

    # Forward BFS from source
    forward = {source: [([source], [])]}
    forward_queue = deque([(source, 0)])

    # Backward BFS from target
    backward = {target: [([target], [])]}
    backward_queue = deque([(target, 0)])

    half_depth = max_depth // 2
    paths = []

    # Expand forward
    while forward_queue:
        node, depth = forward_queue.popleft()
        if depth >= half_depth:
            continue
        for u, r, v in graph.get_neighbors(node):
            neighbor = v if u == node else u
            new_nodes = [p[0] + [neighbor] for p in forward[node]]
            new_edges = [p[1] + [(u, r, v)] for p in forward[node]]
            if neighbor not in forward:
                forward[neighbor] = []
                forward_queue.append((neighbor, depth + 1))
            forward[neighbor].extend(list(zip(new_nodes, new_edges)))

    # Expand backward
    while backward_queue:
        node, depth = backward_queue.popleft()
        if depth >= max_depth - half_depth:
            continue
        for u, r, v in graph.get_neighbors(node):
            neighbor = u if v == node else v
            new_nodes = [[neighbor] + p[0] for p in backward[node]]
            new_edges = [[(u, r, v)] + p[1] for p in backward[node]]
            if neighbor not in backward:
                backward[neighbor] = []
                backward_queue.append((neighbor, depth + 1))
            backward[neighbor].extend(list(zip(new_nodes, new_edges)))

    # Find meeting points
    for mid_node in set(forward.keys()) & set(backward.keys()):
        for f_nodes, f_edges in forward[mid_node]:
            for b_nodes, b_edges in backward[mid_node]:
                full_nodes = f_nodes + b_nodes[1:]
                full_edges = f_edges + b_edges
                if len(full_edges) <= max_depth:
                    rel_type = "->".join(e[1] for e in full_edges)
                    paths.append(GraphPath(
                        nodes=full_nodes,
                        edges=full_edges,
                        relation_type=rel_type,
                    ))

    return paths
