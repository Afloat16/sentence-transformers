from __future__ import annotations

from copy import deepcopy
from itertools import combinations

import pytest

from sentence_transformers.sentence_transformer.evaluation import ParaphraseMiningEvaluator


@pytest.mark.parametrize("shape", ["chain", "star", "complete", "isolated"])
@pytest.mark.parametrize("false_edges", [False, True])
def test_duplicate_closure_matches_boolean_reachability(shape: str, false_edges: bool) -> None:
    nodes = [f"node{i}" for i in range(8)]
    graph = {node: {} for node in nodes}
    for group in (nodes[:5], nodes[5:7]):
        if shape == "chain":
            edges = zip(group, group[1:])
        elif shape == "star":
            edges = [(group[0], node) for node in group[1:]]
        elif shape == "complete":
            edges = combinations(group, 2)
        else:
            edges = []
        for first, second in edges:
            graph[first][second] = graph[second][first] = True
    if false_edges:
        graph[nodes[0]][nodes[-1]] = graph[nodes[-1]][nodes[0]] = False
        graph[nodes[2]][nodes[-2]] = graph[nodes[-2]][nodes[2]] = False
    graph[nodes[0]][nodes[0]] = False
    original = deepcopy(graph)

    # Boolean Warshall closure is independent of the queue traversal.
    reachable = [[first == second or original[first].get(second, False) for second in nodes] for first in nodes]
    for middle in range(len(nodes)):
        for first in range(len(nodes)):
            for second in range(len(nodes)):
                reachable[first][second] |= reachable[first][middle] and reachable[middle][second]
    expected = deepcopy(original)
    for first, second in combinations(range(len(nodes)), 2):
        if reachable[first][second]:
            expected[nodes[first]][nodes[second]] = expected[nodes[second]][nodes[first]] = True

    assert ParaphraseMiningEvaluator.add_transitive_closure(graph) is graph
    assert graph == expected


@pytest.mark.parametrize("size", [1, 3, 17])
def test_duplicate_closure_constructor_counts_all_reachable_pairs(size: int) -> None:
    sentences = {str(index): f"sentence {index}" for index in range(size)}
    duplicates = [(str(index), str(index + 1)) for index in range(size - 1)]
    evaluator = ParaphraseMiningEvaluator(sentences, duplicates_list=duplicates, add_transitive_closure=True)
    assert evaluator.total_num_duplicates == size * (size - 1) // 2
    for first, second in combinations(sentences, 2):
        assert evaluator.duplicates[first][second]
        assert evaluator.duplicates[second][first]
