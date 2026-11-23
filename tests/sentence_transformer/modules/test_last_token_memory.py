from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling


class MultiplyWork(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.largest_output = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.mul.Tensor:
            self.largest_output = max(self.largest_output, result.numel())
        return result


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lasttoken_does_not_multiply_a_full_token_tensor(requires_grad, dtype):
    embeddings = torch.randn(8, 127, 61, dtype=dtype, requires_grad=requires_grad)
    mask = torch.arange(127)[None, :] < torch.arange(120, 128)[:, None]
    with MultiplyWork() as work:
        result = Pooling(61, pooling_mode="lasttoken")({"token_embeddings": embeddings, "attention_mask": mask})[
            "sentence_embedding"
        ]
    assert result.shape == (8, 61)
    assert work.largest_output <= 8 * 61
    if requires_grad:
        result.sum().backward()
        assert torch.isfinite(embeddings.grad).all()


def reference_last(embeddings, mask, prompt_length):
    rows = []
    for values, row_mask in zip(embeddings, mask):
        effective_mask = row_mask.clone()
        attended = effective_mask.nonzero().flatten().tolist()
        if prompt_length and attended:
            effective_mask[: attended[0] + prompt_length] = 0
        attended = effective_mask.nonzero().flatten().tolist()
        index = attended[-1] if attended else 0
        rows.append(values[index] * effective_mask[index].to(values.dtype))
    return torch.stack(rows)


@pytest.mark.parametrize("padding", ["left", "right", "gapped"])
@pytest.mark.parametrize("prompt_length", [0, 1, 9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_lasttoken_keeps_values_gradients_and_input(padding, prompt_length, dtype):
    masks = {
        "left": [[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]],
        "right": [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
        "gapped": [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]],
    }
    mask = torch.tensor(masks[padding], dtype=torch.int64)
    embeddings = torch.arange(60, dtype=dtype).reshape(3, 4, 5).requires_grad_()
    reference_embeddings = embeddings.detach().clone().requires_grad_()
    original_mask = mask.clone()
    original_embeddings = embeddings.detach().clone()
    features = {"token_embeddings": embeddings, "attention_mask": mask, "prompt_length": prompt_length}
    result = Pooling(5, pooling_mode="lasttoken", include_prompt=False)(features)["sentence_embedding"]
    expected = reference_last(reference_embeddings, mask, prompt_length)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    weights = torch.arange(1, 16, dtype=dtype).reshape(3, 5)
    (result * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(embeddings.grad, reference_embeddings.grad, rtol=0, atol=0)
    torch.testing.assert_close(embeddings.detach(), original_embeddings, rtol=0, atol=0)
    torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)


def test_lasttoken_preserves_float_mask_gradients():
    embeddings = torch.randn(3, 7, 5, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0], [0] * 7], dtype=torch.float64, requires_grad=True
    )
    reference_embeddings = embeddings.detach().clone().requires_grad_()
    reference_mask = mask.detach().clone().requires_grad_()
    pooling = Pooling(5, pooling_mode="lasttoken")
    result = pooling({"token_embeddings": embeddings, "attention_mask": mask})["sentence_embedding"]
    # Original full-tensor operation order, including the empty-row selection.
    values, reverse_indices = reference_mask.flip(1).max(1)
    indices = 6 - torch.where(values == 0, 6, reverse_indices)
    expected = (reference_embeddings * reference_mask.unsqueeze(-1))[torch.arange(3), indices]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(embeddings.grad, reference_embeddings.grad, rtol=0, atol=0)
    torch.testing.assert_close(mask.grad, reference_mask.grad, rtol=0, atol=0)


def test_lasttoken_trace_accepts_different_padded_lengths():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.pooling = Pooling(5, pooling_mode="lasttoken")

        def forward(self, values, mask):
            return self.pooling({"token_embeddings": values, "attention_mask": mask})["sentence_embedding"]

    model = Wrapper()
    first = (torch.randn(2, 4, 5), torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]]))
    second = (torch.randn(3, 7, 5), torch.tensor([[1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1], [0] * 7]))
    traced = torch.jit.trace(model, first, check_inputs=[second])
    torch.testing.assert_close(traced(*second), model(*second), rtol=0, atol=0)


def test_pretrained_lasttoken_encoding_matches_selected_hidden_states(stsb_bert_tiny_model):
    transformer = stsb_bert_tiny_model[0]
    pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode="lasttoken")
    model = SentenceTransformer(modules=[transformer, pooling], device="cpu").eval()
    texts = ["Paris.", "Python is a programming language.", "A little snake."]
    features = model.preprocess(texts)
    with torch.no_grad():
        hidden = transformer(copy.deepcopy(features))
        expected = reference_last(hidden["token_embeddings"], hidden["attention_mask"], 0)
        actual = model.encode(texts, convert_to_tensor=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
