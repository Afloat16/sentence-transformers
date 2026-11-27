from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from sentence_transformers.sentence_transformer.evaluation.paraphrase_mining import ParaphraseMiningEvaluator


@pytest.mark.parametrize("label_format", ["list", "dict", "dict_with_negative_pairs"])
@pytest.mark.parametrize("add_transitive_closure", [False, True])
def test_evaluation_does_not_store_unlabeled_pairs(monkeypatch, label_format, add_transitive_closure):
    sentences = {name: name for name in "abcd"}
    labels = [("a", "b"), ("b", "c")]
    if label_format == "list":
        kwargs = {"duplicates_list": labels}
    else:
        kwargs = {"duplicates_dict": {"a": {"b": True}, "b": {"a": True, "c": True}, "c": {"b": True}}}
        if label_format == "dict_with_negative_pairs":
            kwargs["duplicates_dict"]["b"]["d"] = False
    evaluator = ParaphraseMiningEvaluator(
        sentences, add_transitive_closure=add_transitive_closure, write_csv=False, **kwargs
    )
    original = {key: dict(neighbors) for key, neighbors in evaluator.duplicates.items()}
    pairs = [(0, 1), (2, 3), (1, 2), (0, 2), (0, 3), (1, 3)]
    model = SimpleNamespace(similarity=None)
    module = sys.modules[ParaphraseMiningEvaluator.__module__]

    for ordered_pairs in (pairs, list(reversed(pairs))):
        scores = np.linspace(0.9, 0.4, len(ordered_pairs))
        candidates = [(float(score), i, j) for score, (i, j) in zip(scores, ordered_pairs)]
        monkeypatch.setattr(module, "paraphrase_mining", lambda *args, **kwargs: candidates)
        metrics = evaluator.evaluate(model)

        targets = [
            original.get(evaluator.ids[i], {}).get(evaluator.ids[j], False)
            or original.get(evaluator.ids[j], {}).get(evaluator.ids[i], False)
            for i, j in ordered_pairs
        ]
        assert metrics["average_precision"] == pytest.approx(average_precision_score(targets, scores))
        assert metrics["f1"] == pytest.approx(max(f1_score(targets, scores >= threshold) for threshold in scores))
        assert metrics["f1"] == pytest.approx(f1_score(targets, scores >= metrics["threshold"]))
        assert evaluator.duplicates == original
