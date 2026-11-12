from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding
from sentence_transformers.util import compute_count_vector


@pytest.mark.parametrize("layout", ["dense", "coo", "csr"])
@pytest.mark.parametrize("cancel_duplicates", [False, True])
def test_sparse_statistics_count_values_instead_of_stored_entries(layout: str, cancel_duplicates: bool) -> None:
    indices = [[0, 0, 1, 1], [0, 1, 1, 2]]
    values = [1.0, 0.0, 0.0, 2.0]
    if cancel_duplicates:
        indices[0].extend([0, 0])
        indices[1].extend([3, 3])
        values.extend([3.0, -3.0])
    embeddings = torch.sparse_coo_tensor(indices, values, (3, 4))
    dense = embeddings.to_dense()
    if layout == "dense":
        embeddings = dense
    elif layout == "csr":
        embeddings = embeddings.to_sparse_csr()

    stats = SparseEncoder.sparsity(embeddings)
    assert stats["active_dims"] == pytest.approx(2 / 3)
    assert stats["sparsity_ratio"] == pytest.approx(5 / 6)
    torch.testing.assert_close(compute_count_vector(embeddings), (dense != 0).sum(dim=0).int())


@pytest.mark.parametrize("values", [[2.0, 0.0, -2.0], [0.0, 0.0, 0.0]])
def test_sparse_vector_statistics_ignore_explicit_zeros(values: list[float]) -> None:
    embeddings = torch.sparse_coo_tensor([[0, 1, 2]], values, (4,))
    dense = embeddings.to_dense()

    assert SparseEncoder.sparsity(embeddings) == SparseEncoder.sparsity(dense)
    torch.testing.assert_close(compute_count_vector(embeddings), (dense != 0).int())


def test_sparse_retrieval_flops_match_dense_corpus_with_stored_zeros() -> None:
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    )
    model = SparseEncoder(modules=[SparseStaticEmbedding(tokenizer, weight=torch.ones(4))], device="cpu")
    model.model_card_data.generate_widget_examples = False
    evaluator = SparseInformationRetrievalEvaluator(
        queries={"q": "alpha beta"},
        corpus={"alpha": "alpha", "beta": "beta"},
        relevant_docs={"q": {"alpha", "beta"}},
        write_csv=False,
    )
    corpus = torch.sparse_coo_tensor([[0, 0, 1, 1], [2, 3, 2, 3]], [1.0, 0.0, 0.0, 1.0], (2, 4))

    sparse_metrics = evaluator(model, corpus_embeddings=corpus)
    dense_metrics = evaluator(model, corpus_embeddings=corpus.to_dense())

    assert sparse_metrics["corpus_active_dims"] == 1.0
    assert sparse_metrics["avg_flops"] == 1.0
    assert sparse_metrics == dense_metrics
