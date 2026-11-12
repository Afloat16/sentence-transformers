from __future__ import annotations

import csv

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseRerankingEvaluator
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding
from sentence_transformers.util import cos_sim


@pytest.mark.parametrize("use_batched_encoding", [True, False])
def test_reranking_sparsity_weights_each_embedding(tmp_path, use_batched_encoding):
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3, "c": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]")
    model = SparseEncoder(
        modules=[SparseStaticEmbedding(tokenizer, weight=torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))], device="cpu"
    )
    model.model_card_data.generate_widget_examples = False
    samples = [
        {"query": "a", "positive": ["a"], "negative": ["b"]},
        {"query": "b c", "positive": ["a b c"], "negative": ["a c", "b c", "a b"]},
    ]
    expected = {}
    for prefix, texts in [
        ("query", [sample["query"] for sample in samples]),
        ("corpus", [text for sample in samples for text in sample["positive"] + sample["negative"]]),
    ]:
        embeddings = model.encode(texts, convert_to_sparse_tensor=True)
        expected.update({f"weighted_{prefix}_{key}": value for key, value in model.sparsity(embeddings).items()})

    evaluator = SparseRerankingEvaluator(samples, name="weighted", use_batched_encoding=use_batched_encoding)
    # Reusing an evaluator must reset its counts as well as its accumulated statistics.
    for _ in range(2):
        metrics = evaluator(model, output_path=str(tmp_path))
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value)

    with (tmp_path / evaluator.csv_file).open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    for row in rows:
        assert float(row["corpus_active_dims"]) == pytest.approx(11 / 6)


@pytest.mark.parametrize("use_batched_encoding", [True, False])
@pytest.mark.parametrize(
    "similarity_fn_name, similarity_fct, expected_score",
    [("dot", None, 1.0), ("cosine", None, 0.0), ("dot", cos_sim, 0.0)],
)
def test_reranking_evaluator_similarity(
    use_batched_encoding: bool, similarity_fn_name: str, similarity_fct, expected_score: float
) -> None:
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3, "gamma": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    )
    model = SparseEncoder(
        modules=[SparseStaticEmbedding(tokenizer, weight=torch.tensor([0.0, 0.0, 2.0, 1.0, 2.0]))],
        similarity_fn_name=similarity_fn_name,
    )
    model.model_card_data.generate_widget_examples = False

    evaluator = SparseRerankingEvaluator(
        samples=[{"query": "alpha beta", "positive": ["alpha beta gamma"], "negative": ["alpha"]}],
        at_k=1,
        write_csv=False,
        use_batched_encoding=use_batched_encoding,
        similarity_fct=similarity_fct,
    )

    metrics = evaluator(model)
    assert metrics["mrr@1"] == expected_score
    assert metrics["ndcg@1"] == expected_score
