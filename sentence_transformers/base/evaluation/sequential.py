from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel


class SequentialEvaluator(BaseEvaluator):
    """
    This evaluator allows that multiple sub-evaluators are passed. When the model is evaluated,
    the data is passed sequentially to all sub-evaluators.

    All scores are passed to 'main_score_function', which derives one final score value

    Args:
        evaluators (Iterable[BaseEvaluator]): A collection of BaseEvaluator objects.
        main_score_function (function, optional): A function that takes a list of scores and returns the main score.
            Defaults to selecting the last score in the list and inheriting the last evaluator's
            ``greater_is_better`` setting. Custom score functions default to higher scores being better;
            set ``greater_is_better`` on this evaluator if the custom score should be minimized.

    Example:
        ::

            evaluator1 = BinaryClassificationEvaluator(...)
            evaluator2 = InformationRetrievalEvaluator(...)
            evaluator3 = MSEEvaluator(...)
            seq_evaluator = SequentialEvaluator([evaluator1, evaluator2, evaluator3])
    """

    def __init__(
        self,
        evaluators: Iterable[BaseEvaluator],
        main_score_function: Callable[[list[float]], float] | None = None,
    ):
        super().__init__()
        self.evaluators = list(evaluators)
        self._use_last_score = main_score_function is None
        self.main_score_function = main_score_function if main_score_function is not None else lambda scores: scores[-1]
        if self._use_last_score and self.evaluators:
            self.greater_is_better = getattr(self.evaluators[-1], "greater_is_better", True)

    def __call__(
        self, model: BaseModel, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        evaluations = []
        scores = []
        for evaluator_idx, evaluator in enumerate(self.evaluators):
            evaluation = evaluator(model, output_path, epoch, steps)

            if not isinstance(evaluation, dict):
                scores.append(evaluation)
                evaluation = {f"evaluator_{evaluator_idx}": evaluation}
            else:
                if hasattr(evaluator, "primary_metric") and evaluator.primary_metric is not None:
                    scores.append(evaluation[evaluator.primary_metric])
                else:
                    scores.append(evaluation[list(evaluation.keys())[0]])

            evaluations.append(evaluation)

        self.primary_metric = "sequential_score"
        if not scores:
            main_score = 0.0
        else:
            main_score = self.main_score_function(scores)
            if self._use_last_score:
                self.greater_is_better = getattr(self.evaluators[-1], "greater_is_better", True)
        results = {key: value for evaluation in evaluations for key, value in evaluation.items()}
        results["sequential_score"] = main_score
        return results
