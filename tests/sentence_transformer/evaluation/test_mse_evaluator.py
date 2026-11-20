from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sentence_transformer.evaluation import MSEEvaluator
from sentence_transformers.sparse_encoder.evaluation import SparseMSEEvaluator
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding


@pytest.mark.parametrize("evaluator_cls", [MSEEvaluator, SparseMSEEvaluator])
@pytest.mark.parametrize(("source_count", "target_count"), [(1, 2), (2, 1), (2, 3)])
def test_mse_rejects_unequal_sentence_counts(evaluator_cls, source_count, target_count):
    teacher = Mock()
    teacher.sparsity.return_value = {"active_dims": 1.0, "sparsity_ratio": 0.5}

    with pytest.raises(ValueError, match="source_sentences.*target_sentences.*same length") as exc_info:
        evaluator_cls(
            source_sentences=["source"] * source_count,
            target_sentences=["target"] * target_count,
            teacher_model=teacher,
        )

    assert str(exc_info.value) == (
        f"source_sentences ({source_count}) and target_sentences ({target_count}) must have the same length."
    )
    teacher.encode.assert_not_called()


@pytest.mark.parametrize("evaluator_cls", [MSEEvaluator, SparseMSEEvaluator])
@pytest.mark.parametrize("sentence_count", [1, 2])
@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_mse_equal_sentence_counts_preserve_score(evaluator_cls, sentence_count, offset):
    source_embeddings = np.eye(2, dtype=np.float32)[:sentence_count]
    target_embeddings = source_embeddings + offset
    if evaluator_cls is SparseMSEEvaluator:
        source_embeddings = torch.from_numpy(source_embeddings)
        target_embeddings = torch.from_numpy(target_embeddings)

    teacher = Mock()
    teacher.encode.return_value = source_embeddings
    teacher.sparsity.return_value = {"active_dims": 1.0, "sparsity_ratio": 0.5}
    student = Mock()
    student.encode.return_value = target_embeddings
    student.sparsity.return_value = {"active_dims": 1.0, "sparsity_ratio": 0.5}
    sources = [f"source {index}" for index in range(sentence_count)]
    targets = [f"target {index}" for index in range(sentence_count)]

    evaluator = evaluator_cls(sources, targets, teacher_model=teacher, name="paired")
    metrics = evaluator(student)

    assert metrics["paired_negative_mse"] == pytest.approx(-100 * offset**2)
    assert evaluator.primary_metric == "paired_negative_mse"
    teacher.encode.assert_called_once()
    student.encode.assert_called_once()
    assert teacher.encode.call_args.args[0] == sources
    assert student.encode.call_args.args[0] == targets


def _static_sparse_model(weights: torch.Tensor) -> SparseEncoder:
    backend = Tokenizer(
        models.WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3, "gamma": 4, "delta": 5}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", model_max_length=16
    )
    return SparseEncoder(modules=[SparseStaticEmbedding(tokenizer, weights)], device="cpu")


@pytest.mark.parametrize(
    ("teacher_dtype", "student_dtype"),
    [(dtype, dtype) for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]]
    + [(torch.float16, torch.float64), (torch.float64, torch.float32)],
)
def test_sparse_mse_keeps_embeddings_sparse_and_matches_dense_reference(teacher_dtype, student_dtype, tmp_path: Path):
    teacher = _static_sparse_model(torch.tensor([0, 0, 1, -0.5, 0.75, 0], dtype=teacher_dtype))
    student = _static_sparse_model(torch.tensor([0, 0, 0.75, 0.25, 0, 1.25], dtype=student_dtype))
    sources = ["alpha alpha beta", "gamma", "", "delta beta"]
    targets = ["alpha gamma", "beta beta", "", "delta"]
    source_reference = teacher.encode(sources, convert_to_sparse_tensor=False).double()
    target_reference = student.encode(targets, convert_to_sparse_tensor=False).double()
    expected_mse = ((source_reference - target_reference) ** 2).mean().item() * 100
    expected_active = target_reference.count_nonzero(dim=1).double().mean().item()

    evaluator = SparseMSEEvaluator(sources, targets, teacher_model=teacher, name="sparse", batch_size=2)
    cached = evaluator.source_embeddings
    assert cached.is_sparse
    assert cached.dtype == teacher_dtype
    torch.testing.assert_close(cached.to_dense().double(), source_reference)
    original_indices, original_values = cached._indices().clone(), cached._values().clone()

    for step in [1, 2]:
        metrics = evaluator(student, output_path=str(tmp_path), epoch=0, steps=step)
        assert metrics["sparse_negative_mse"] == pytest.approx(-expected_mse)
        assert metrics["sparse_active_dims"] == expected_active
        assert metrics["sparse_sparsity_ratio"] == pytest.approx(1 - expected_active / source_reference.shape[1])
        assert evaluator.source_embeddings is cached
        torch.testing.assert_close(cached._indices(), original_indices)
        torch.testing.assert_close(cached._values(), original_values)

    with (tmp_path / evaluator.csv_file).open(newline="", encoding="utf8") as csv_file:
        rows = list(csv.reader(csv_file))
    assert rows[0] == ["epoch", "steps", "MSE", "active_dims", "sparsity_ratio"]
    assert len(rows) == 3
    for step, row in enumerate(rows[1:], start=1):
        assert len(row) == len(rows[0])
        assert [int(row[0]), int(row[1])] == [0, step]
        assert float(row[2]) == pytest.approx(expected_mse)
        assert float(row[3]) == expected_active
        assert float(row[4]) == pytest.approx(metrics["sparse_sparsity_ratio"])


def test_sparse_mse_with_no_stored_values():
    model = _static_sparse_model(torch.zeros(6))
    evaluator = SparseMSEEvaluator(["alpha", ""], ["beta", ""], teacher_model=model, write_csv=False)
    assert evaluator.source_embeddings.is_sparse
    assert evaluator.source_embeddings._nnz() == 0
    assert evaluator(model) == {"negative_mse": 0.0, "active_dims": 0.0, "sparsity_ratio": 1.0}


def test_sparse_mse_preserves_float64_precision():
    teacher = _static_sparse_model(torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float64))
    student = _static_sparse_model(torch.tensor([0, 0, 1 + 2e-10, 0, 0, 0], dtype=torch.float64))
    evaluator = SparseMSEEvaluator(["alpha"], ["alpha"], teacher_model=teacher, write_csv=False)
    expected = ((teacher[0].weight - student[0].weight) ** 2).mean().item() * 100
    assert evaluator(student)["negative_mse"] == pytest.approx(-expected, rel=1e-12, abs=0)


def test_sparse_mse_coalesces_duplicate_coordinates():
    source = torch.sparse_coo_tensor([[0, 0, 1, 1], [1, 1, 2, 3]], [2.0, -1.0, -0.5, 0.0], (2, 6))
    target = torch.sparse_coo_tensor([[0, 0, 1, 1], [1, 2, 2, 2]], [1.0, 0.75, 0.5, -1.0], (2, 6))
    assert not source.is_coalesced() and not target.is_coalesced()
    expected = ((source.to_dense() - target.to_dense()) ** 2).mean().item() * 100
    teacher, student = Mock(), Mock()
    for model, embedding in [(teacher, source), (student, target)]:
        model.encode.return_value = embedding
        model.sparsity.side_effect = SparseEncoder.sparsity
    evaluator = SparseMSEEvaluator(["a", "b"], ["c", "d"], teacher_model=teacher, write_csv=False)
    assert evaluator(student)["negative_mse"] == pytest.approx(-expected)
    assert not source.is_coalesced() and not target.is_coalesced()
