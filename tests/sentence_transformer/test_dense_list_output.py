from __future__ import annotations

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.util.quantization import quantize_embeddings


@pytest.fixture
def model() -> SentenceTransformer:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "short": 1, "longer": 2, "sentence": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    weights = torch.randn(4, 32, generator=torch.Generator().manual_seed(17))
    return SentenceTransformer(modules=[StaticEmbedding(tokenizer, embedding_weights=weights)], device="cpu")


TEXTS = ["short", "longer sentence", "short longer sentence", "sentence", "short short", "longer", ""]


class ScaleWithStackKeyword(torch.nn.Module):
    forward_kwargs = {"stack_embeddings"}

    def forward(self, features, stack_embeddings):
        features["sentence_embedding"] = features["sentence_embedding"] * stack_embeddings
        return features


@pytest.mark.parametrize("output_options", [{}, {"convert_to_numpy": False}, {"convert_to_tensor": True}])
def test_dense_list_internal_control_does_not_consume_module_kwarg(model, output_options):
    expected = model.encode(TEXTS, convert_to_tensor=True) * 2
    model.add_module("scale", ScaleWithStackKeyword())
    outputs = model.encode(TEXTS, stack_embeddings=2, **output_options)
    actual = torch.stack(outputs) if isinstance(outputs, list) else torch.as_tensor(outputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("truncate_dim", [None, 8])
@pytest.mark.parametrize("normalize_embeddings", [False, True])
def test_dense_list_output_keeps_compact_batch_storage(model, dtype, truncate_dim, normalize_embeddings):
    model.to(dtype=dtype)
    # A nonempty text avoids the existing FP16 zero-normalization behavior.
    texts = TEXTS[:-1]
    options = {"batch_size": 2, "truncate_dim": truncate_dim, "normalize_embeddings": normalize_embeddings}
    expected = model.encode(texts, convert_to_tensor=True, **options)
    outputs = model.encode(texts, convert_to_numpy=False, **options)

    assert isinstance(outputs, list)
    assert len(outputs) == len(texts)
    assert all(output.dtype == dtype and output.device == model.device for output in outputs)
    torch.testing.assert_close(torch.stack(outputs), expected, rtol=0, atol=0)

    storages = {output.untyped_storage().data_ptr(): output.untyped_storage().nbytes() for output in outputs}
    row_bytes = expected.shape[1] * expected.element_size()
    # A retained result must not hold the whole corpus, or the untruncated dimensions, alive.
    assert max(storages.values()) <= options["batch_size"] * row_bytes
    assert sum(storages.values()) == len(texts) * row_bytes


@pytest.mark.parametrize("precision", [None, "float32"])
@pytest.mark.parametrize("inputs", ["longer sentence", ["longer sentence"], []])
def test_dense_list_output_preserves_single_and_empty_shapes(model, precision, inputs):
    expected = model.encode(inputs, convert_to_tensor=True, precision=precision)
    outputs = model.encode(inputs, convert_to_numpy=False, precision=precision)
    if isinstance(inputs, str):
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (32,)
        torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
    else:
        assert isinstance(outputs, list)
        assert len(outputs) == len(inputs)
        if inputs:
            torch.testing.assert_close(outputs[0], expected[0], rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["int8", "uint8", "binary", "ubinary"])
def test_dense_list_quantization_still_calibrates_across_all_batches(model, precision):
    embeddings = model.encode(TEXTS, convert_to_tensor=True, batch_size=2)
    expected = torch.from_numpy(quantize_embeddings(embeddings, precision=precision))
    outputs = model.encode(TEXTS, convert_to_numpy=False, precision=precision, batch_size=2)
    assert isinstance(outputs, list)
    torch.testing.assert_close(torch.stack(outputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("convert_to_tensor", [False, True])
def test_dense_list_change_preserves_matrix_output(model, convert_to_tensor):
    outputs = model.encode(TEXTS, convert_to_tensor=convert_to_tensor, batch_size=2)
    assert isinstance(outputs, torch.Tensor if convert_to_tensor else np.ndarray)
    assert outputs.shape == (len(TEXTS), 32)


@pytest.mark.parametrize("precision", ["float32", "int8"])
@pytest.mark.parametrize("parallel_path", ["multi_process", "distributed"])
def test_dense_list_parallel_boundary_keeps_stacked_chunks(model, precision, parallel_path, monkeypatch):
    def infer_chunks(inputs, **kwargs):
        # Inspect the actual arguments crossing the process boundary, without requiring worker IPC.
        assert "_stack_embeddings" not in kwargs
        for key in ("device", "pool", "chunk_size"):
            kwargs.pop(key, None)
        chunks = [
            model._inference(inputs[start : start + 3], device="cpu", **kwargs) for start in range(0, len(inputs), 3)
        ]
        assert all(isinstance(chunk, torch.Tensor) for chunk in chunks)
        return torch.cat(chunks)

    options = {"convert_to_numpy": False, "precision": precision, "batch_size": 2, "show_progress_bar": False}
    expected = model.encode(TEXTS, **options)
    if parallel_path == "multi_process":
        monkeypatch.setattr(model, "_multi_process", infer_chunks)
        outputs = model.encode(TEXTS, device=["cpu"], chunk_size=3, **options)
    else:
        monkeypatch.setattr(model, "_distributed_inference", infer_chunks)
        outputs = model.encode(TEXTS, **options)
    assert isinstance(outputs, list)
    assert all(output.device.type == "cpu" for output in outputs)
    torch.testing.assert_close(torch.stack(outputs), torch.stack(expected), rtol=0, atol=0)
