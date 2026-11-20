"""Sparse baseline clustering for Phase 2."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def connected_components(ids: list[str], candidate_pairs: list[dict]) -> list[list[str]]:
    graph = {story_id: set() for story_id in ids}
    for pair in candidate_pairs:
        graph[pair["a_id"]].add(pair["b_id"])
        graph[pair["b_id"]].add(pair["a_id"])
    components: list[list[str]] = []
    unseen = set(ids)
    for start in ids:
        if start not in unseen:
            continue
        stack = [start]
        unseen.remove(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(graph[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def cluster_headlines(
    ids: list[str],
    candidate_pairs: list[dict],
    tau: float,
    method: str = "average",
) -> list[list[str]]:
    candidate_scores = {
        tuple(sorted((pair["a_id"], pair["b_id"]))): pair["cosine"]
        for pair in candidate_pairs
    }
    labels: list[list[str]] = []
    for component in connected_components(ids, candidate_pairs):
        if len(component) == 1:
            labels.append(component)
            continue
        distance = np.ones((len(component), len(component)), dtype=np.float64)
        positions = {story_id: i for i, story_id in enumerate(component)}
        np.fill_diagonal(distance, 0.0)
        for (a_id, b_id), score in candidate_scores.items():
            if a_id in positions and b_id in positions:
                pair_distance = float(np.clip(1.0 - score, 0.0, 2.0))
                distance[positions[a_id], positions[b_id]] = pair_distance
                distance[positions[b_id], positions[a_id]] = pair_distance
        if np.max(distance) <= 0:
            labels.append(component)
            continue
        tree = linkage(squareform(distance, checks=False), method=method)
        assignments = fcluster(tree, t=1.0 - tau, criterion="distance")
        groups: dict[int, list[str]] = {}
        for story_id, assignment in zip(component, assignments, strict=True):
            groups.setdefault(int(assignment), []).append(story_id)
        labels.extend(sorted((sorted(group) for group in groups.values()), key=lambda x: x[0]))
    return sorted(labels, key=lambda group: group[0])
