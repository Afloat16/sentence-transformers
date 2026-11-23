from __future__ import annotations

import copy
import tracemalloc
import types

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss
from tests.sentence_transformer.losses.utils import disable_dropout


def inputs_for(batch_size, candidate_columns):
    return [[f"query {row}" for row in range(batch_size)]] + [
        [f"column {column} row {row}" for row in range(batch_size)] for column in range(candidate_columns)
    ]


def original_negatives(inputs, num_negatives):
    batch = len(inputs[0])
    candidates = [text for column in inputs[1:] for text in column]
    mask = ~torch.eye(batch, dtype=torch.bool).repeat(1, len(inputs) - 1)
    count = len(candidates) - (len(inputs) - 1)
    if num_negatives is not None and num_negatives < count:
        indices = torch.multinomial(mask.float(), num_negatives)
    else:
        indices = torch.arange(len(candidates)).repeat(batch, 1)[mask].reshape(batch, -1)
    return [[candidates[index] for index in column] for column in indices.T]


def original_pairs(inputs, num_negatives):
    columns = [inputs[1], *original_negatives(inputs, num_negatives), *inputs[2:]]
    return [(query, document) for column in columns for query, document in zip(inputs[0], column)]


class LargestTensor(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.elements = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for tensor in result if isinstance(result, (tuple, list)) else [result]:
            if isinstance(tensor, torch.Tensor):
                self.elements = max(self.elements, tensor.numel())
        return result


@pytest.mark.parametrize("num_negatives", [None, 4])
def test_sampling_does_not_materialize_full_pair_matrix(reranker_bert_tiny_model_v54, num_negatives):
    inputs = inputs_for(1024, 3)
    loss = MultipleNegativesRankingLoss(reranker_bert_tiny_model_v54, num_negatives=num_negatives)
    with LargestTensor() as work:
        first_column = next(loss.get_in_batch_negatives(inputs[0], inputs[1:]))
    assert len(first_column) == 1024
    # A generous fixed working-tensor bound, independent of the 3,145,728
    # candidate pairs in this batch. The returned column needs only 1024 items.
    assert work.elements <= 1_100_000


class CapturedPairs(Exception):
    pass


def capture_pairs(loss, inputs):
    captured = []

    def intercept(*, pairs, **kwargs):
        assert isinstance(pairs, list)
        captured.append(pairs)
        raise CapturedPairs

    loss.predict_minibatch_iter = intercept
    with pytest.raises(CapturedPairs):
        loss(inputs)
    return captured[0]


@pytest.mark.parametrize("num_negatives", [None, 4])
def test_cached_pair_preparation_has_bounded_python_memory(reranker_bert_tiny_model_v54, num_negatives):
    inputs = inputs_for(512, 3)
    loss = CachedMultipleNegativesRankingLoss(reranker_bert_tiny_model_v54, num_negatives=num_negatives)
    first_batch = []

    def intercept(*, pairs, **kwargs):
        first_batch.extend(pairs)
        raise CapturedPairs

    loss.predict_minibatch = intercept
    tracemalloc.start()
    try:
        with pytest.raises(CapturedPairs):
            loss(inputs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert first_batch == list(zip(inputs[0][:32], inputs[1][:32]))
    assert peak < 1_000_000


@pytest.mark.parametrize("batch_size", [1, 2, 7])
@pytest.mark.parametrize("candidate_columns", [1, 3])
@pytest.mark.parametrize("num_negatives", [None, 100])
def test_all_negative_columns_keep_exact_order(
    reranker_bert_tiny_model_v54, batch_size, candidate_columns, num_negatives
):
    inputs = inputs_for(batch_size, candidate_columns)
    loss = MultipleNegativesRankingLoss(reranker_bert_tiny_model_v54, num_negatives=num_negatives)
    state = torch.get_rng_state()
    actual = list(loss.get_in_batch_negatives(inputs[0], inputs[1:]))
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert actual == original_negatives(inputs, num_negatives)


@pytest.mark.parametrize("batch_size", [7, 1024])
def test_sampled_columns_are_distinct_allowed_and_repeatable(reranker_bert_tiny_model_v54, batch_size):
    inputs = inputs_for(batch_size, 3)
    loss = MultipleNegativesRankingLoss(reranker_bert_tiny_model_v54, num_negatives=4)
    torch.manual_seed(29)
    actual = list(loss.get_in_batch_negatives(inputs[0], inputs[1:]))
    torch.manual_seed(29)
    assert actual == list(loss.get_in_batch_negatives(inputs[0], inputs[1:]))
    assert len(actual) == 4
    for row in range(batch_size):
        values = [column[row] for column in actual]
        assert len(set(values)) == 4
        assert all(value not in [column[row] for column in inputs[1:]] for value in values)


@pytest.mark.parametrize("num_negatives", [None, 4])
def test_custom_pair_iterator_keeps_original_list_snapshot(reranker_bert_tiny_model_v54, num_negatives):
    inputs = inputs_for(7, 3)
    torch.manual_seed(31)
    expected = original_pairs(inputs, num_negatives)
    torch.manual_seed(31)
    pairs = capture_pairs(
        CachedMultipleNegativesRankingLoss(reranker_bert_tiny_model_v54, num_negatives=num_negatives), inputs
    )
    for column in inputs:
        column[:] = ["changed"] * len(column)
    assert list(pairs) == expected
    assert pairs[3:23:2] == expected[3:23:2]
    assert pairs[::-1] == expected[::-1]
    assert pairs[-1] == expected[-1]
    with pytest.raises(IndexError):
        pairs[len(pairs)]


@pytest.mark.parametrize("override", ["instance", "subclass", "base_class"])
def test_cached_loss_honors_custom_negative_sampler(reranker_bert_tiny_model_v54, monkeypatch, override):
    def custom(self, anchors, candidates):
        yield ["custom negative"] * len(anchors)

    if override == "instance":
        loss = CachedMultipleNegativesRankingLoss(reranker_bert_tiny_model_v54)
        loss.get_in_batch_negatives = types.MethodType(custom, loss)
    elif override == "base_class":
        monkeypatch.setattr(MultipleNegativesRankingLoss, "get_in_batch_negatives", custom)
        loss = CachedMultipleNegativesRankingLoss(reranker_bert_tiny_model_v54)
    else:

        class CustomLoss(CachedMultipleNegativesRankingLoss):
            get_in_batch_negatives = custom

        loss = CustomLoss(reranker_bert_tiny_model_v54)
    inputs = inputs_for(3, 2)
    pairs = capture_pairs(loss, inputs)
    expected_columns = [inputs[1], ["custom negative"] * 3, inputs[2]]
    assert list(pairs) == [
        (query, document) for column in expected_columns for query, document in zip(inputs[0], column)
    ]


@pytest.mark.parametrize("num_negatives", [None, 4])
@pytest.mark.parametrize("mini_batch_size", [3, 16])
def test_pretrained_cached_loss_matches_original_eager_pairs(
    reranker_bert_tiny_model_v54, num_negatives, mini_batch_size
):
    class OriginalSampler(CachedMultipleNegativesRankingLoss):
        def get_in_batch_negatives(self, anchors, candidates):
            yield from original_negatives([anchors, *candidates], self.num_negatives)

    model = reranker_bert_tiny_model_v54.train()
    disable_dropout(model, canary_inputs=[["query", "document"], ["other", "answer"]])
    inputs = inputs_for(4, 2)
    outputs = []
    for cls in [OriginalSampler, CachedMultipleNegativesRankingLoss]:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(71)
        loss = cls(model, num_negatives=num_negatives, mini_batch_size=mini_batch_size)(copy.deepcopy(inputs))
        loss.backward()
        gradients = {
            name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None
        }
        outputs.append((loss.detach(), gradients))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    assert outputs[0][1]
    assert outputs[0][1].keys() == outputs[1][1].keys()
    for name, gradient in outputs[0][1].items():
        torch.testing.assert_close(gradient, outputs[1][1][name], rtol=0, atol=0)


@pytest.mark.parametrize("num_negatives", [None, 4])
def test_cached_backward_replays_snapshot_after_input_lists_change(reranker_bert_tiny_model_v54, num_negatives):
    model = reranker_bert_tiny_model_v54.train()
    loss_function = CachedMultipleNegativesRankingLoss(model, num_negatives=num_negatives, mini_batch_size=3)
    inputs = inputs_for(4, 2)
    seen = []
    predict = loss_function.predict_minibatch

    def record(**kwargs):
        logits, random_state = predict(**kwargs)
        seen.append((copy.deepcopy(kwargs["pairs"]), logits.detach().clone()))
        return logits, random_state

    loss_function.predict_minibatch = record
    model.zero_grad(set_to_none=True)
    loss = loss_function(inputs)
    forward_count = len(seen)
    for column in inputs:
        column[:] = ["changed"] * len(column)
    loss.backward()
    assert len(seen) == 2 * forward_count
    for before, after in zip(seen[:forward_count], seen[forward_count:]):
        assert before[0] == after[0]
        torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("hook", ["prediction", "replay"])
def test_custom_iterators_keep_mutable_pair_lists(reranker_bert_tiny_model_v54, hook):
    seen = []

    class CustomPrediction(CachedMultipleNegativesRankingLoss):
        def predict_minibatch_iter(self, pairs, **kwargs):
            assert isinstance(pairs, list)
            seen.append(len(pairs))
            yield from super().predict_minibatch_iter(pairs=pairs.copy(), **kwargs)

    class CustomReplay(CachedMultipleNegativesRankingLoss):
        def embed_minibatch_iter(self, sentence_feature, **kwargs):
            assert isinstance(sentence_feature["pairs"], list)
            seen.append(len(sentence_feature["pairs"]))
            features = {**sentence_feature, "pairs": sentence_feature["pairs"].copy()}
            yield from super().embed_minibatch_iter(sentence_feature=features, **kwargs)

    model = reranker_bert_tiny_model_v54.train()
    model.zero_grad(set_to_none=True)
    cls = CustomPrediction if hook == "prediction" else CustomReplay
    cls(model, num_negatives=None, mini_batch_size=3)(inputs_for(3, 1)).backward()
    assert seen and all(count == 9 for count in seen)
