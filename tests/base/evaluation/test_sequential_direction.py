from __future__ import annotations

import pytest

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
from sentence_transformers.base.evaluation.sequential import SequentialEvaluator


class DirectionEvaluator(BaseEvaluator):
    def __init__(self, name: str, score: float, greater_is_better: bool):
        super().__init__()
        self.primary_metric = name
        self.score = score
        self.greater_is_better = greater_is_better

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        return {self.primary_metric: self.score}


@pytest.mark.parametrize("greater_is_better", [False, True])
@pytest.mark.parametrize("container", [list, iter], ids=["list", "iterator"])
def test_default_score_inherits_last_evaluator_direction(greater_is_better, container):
    last = DirectionEvaluator("loss", 3.0, greater_is_better)
    evaluator = SequentialEvaluator(container([DirectionEvaluator("accuracy", 0.5, not greater_is_better), last]))

    assert evaluator.greater_is_better is greater_is_better
    for score in (3.0, 2.0):
        last.score = score
        assert evaluator(None)[evaluator.primary_metric] == score
        assert evaluator.greater_is_better is greater_is_better


def test_default_score_refreshes_direction_after_evaluation():
    class RuntimeDirectionEvaluator(DirectionEvaluator):
        def __call__(self, *args, **kwargs):
            self.greater_is_better = False
            return super().__call__(*args, **kwargs)

    evaluator = SequentialEvaluator([RuntimeDirectionEvaluator("loss", 2.0, True)])

    assert evaluator.greater_is_better is True
    assert evaluator(None)["sequential_score"] == 2.0
    assert evaluator.greater_is_better is False


def test_custom_score_function_keeps_its_own_direction():
    last = DirectionEvaluator("loss", 2.0, False)
    evaluator = SequentialEvaluator([last], main_score_function=lambda scores: -scores[-1])

    assert evaluator.greater_is_better is True
    assert evaluator(None)["sequential_score"] == -2.0
    assert evaluator.greater_is_better is True

    evaluator.main_score_function = lambda scores: scores[-1]
    evaluator.greater_is_better = False
    assert evaluator(None)["sequential_score"] == 2.0
    assert evaluator.greater_is_better is False


def test_nested_default_evaluators_inherit_minimization():
    evaluator = SequentialEvaluator([SequentialEvaluator([DirectionEvaluator("loss", 2.0, False)])])

    assert evaluator.greater_is_better is False
    assert evaluator(None)["sequential_score"] == 2.0
    assert evaluator.greater_is_better is False


def test_legacy_scalar_evaluator_keeps_maximization():
    evaluator = SequentialEvaluator([lambda *args: 0.75])

    assert evaluator.greater_is_better is True
    assert evaluator(None) == {"evaluator_0": 0.75, "sequential_score": 0.75}
    assert evaluator.greater_is_better is True


def test_empty_evaluator_keeps_zero_score():
    evaluator = SequentialEvaluator([])

    assert evaluator.greater_is_better is True
    assert evaluator(None) == {"sequential_score": 0.0}
