from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, ndcg_score
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sentence_transformers import MultiVectorEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorRerankingEvaluator
from sentence_transformers.sentence_transformer.evaluation import RerankingEvaluator
from sentence_transformers.sentence_transformer.modules import StaticEmbedding, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.sparse_encoder.evaluation import SparseRerankingEvaluator
from sentence_transformers.util.metrics import _reciprocal_rank


@pytest.fixture
def model():
    vocabulary = {"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3, "queryprefix": 4, "docprefix": 5}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    weights = torch.randn(6, 8, generator=torch.Generator().manual_seed(5))
    model = SentenceTransformer(modules=[StaticEmbedding(tokenizer, embedding_weights=weights)], device="cpu")
    model.prompts = {"query": "queryprefix ", "document": "docprefix "}
    return model


def make_evaluator(**kwargs):
    return RerankingEvaluator(
        [{"query": "alpha", "positive": ["beta"], "negative": ["gamma"]}], write_csv=False, batch_size=2, **kwargs
    )


@pytest.mark.parametrize("encode_fn_name", [None, "query", "document"])
@pytest.mark.parametrize("truncate_dim", [None, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reranking_encodes_unique_texts_and_restores_occurrences(
    model, monkeypatch, encode_fn_name, truncate_dim, dtype
):
    model.to(dtype=dtype)
    inputs = ["alpha beta", "gamma", "alpha beta", "alpha", "gamma"]
    evaluator = make_evaluator(truncate_dim=truncate_dim)
    method = f"encode_{encode_fn_name}" if encode_fn_name else "encode"
    encode = getattr(model, method)
    expected = encode(inputs, convert_to_tensor=True, batch_size=2, truncate_dim=truncate_dim)
    seen = []

    def record(sentences, **kwargs):
        seen.append(sentences)
        return encode(sentences, **kwargs)

    monkeypatch.setattr(model, method, record)
    actual = evaluator.embed_inputs(model, inputs, encode_fn_name=encode_fn_name)
    assert seen == [["alpha beta", "gamma", "alpha"]]
    assert actual.dtype == dtype
    assert actual.device == model.device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("use_batched_encoding", [False, True])
def test_reranking_reuse_preserves_candidate_scores_and_metrics(model, use_batched_encoding):
    samples = [
        {"query": "alpha", "positive": ["beta", "beta"], "negative": ["gamma", "beta", "gamma"]},
        {"query": "beta", "positive": ["gamma"], "negative": ["alpha", "beta", "alpha"]},
    ]
    expected_scores = []
    expected_metrics = {"map": [], "mrr@3": [], "ndcg@3": []}
    for sample in samples:
        query = model.encode_query([sample["query"]], convert_to_tensor=True)
        documents = model.encode_document(sample["positive"] + sample["negative"], convert_to_tensor=True)
        scores = model.similarity(query, documents)[0]
        expected_scores.append(scores)
        labels = [1] * len(sample["positive"]) + [0] * len(sample["negative"])
        expected_metrics["map"].append(average_precision_score(labels, scores.tolist()))
        expected_metrics["mrr@3"].append(_reciprocal_rank(labels, scores.tolist(), k=3))
        expected_metrics["ndcg@3"].append(ndcg_score([labels], [scores.tolist()], k=3))

    actual_scores = []

    def record_similarity(query, documents):
        scores = model.similarity(query, documents)
        actual_scores.append(scores[0])
        return scores

    evaluator = RerankingEvaluator(
        samples,
        at_k=3,
        batch_size=2,
        write_csv=False,
        similarity_fct=record_similarity,
        use_batched_encoding=use_batched_encoding,
    )
    metrics = evaluator(model)
    for actual, expected in zip(actual_scores, expected_scores):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metrics == pytest.approx({key: np.mean(values) for key, values in expected_metrics.items()})


def test_reranking_reuse_does_not_cache_across_training_steps(model, monkeypatch):
    evaluator = make_evaluator()
    inputs = ["alpha", "beta", "alpha"]
    original_encode = model.encode_document
    seen = []

    def record(sentences, **kwargs):
        seen.append(sentences)
        return original_encode(sentences, **kwargs)

    monkeypatch.setattr(model, "encode_document", record)
    before = evaluator.embed_inputs(model, inputs, encode_fn_name="document")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    output = model(model.preprocess(["alpha"]))["sentence_embedding"]
    output.square().sum().backward()
    optimizer.step()
    after = evaluator.embed_inputs(model, inputs, encode_fn_name="document")
    expected = original_encode(inputs, convert_to_tensor=True, batch_size=2)
    assert seen == [["alpha", "beta"], ["alpha", "beta"]]
    assert not torch.equal(before, after)
    torch.testing.assert_close(after, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "inputs",
    [
        ["alpha", "beta"],
        [],
        "alpha beta",
        ("alpha", "alpha"),
        [{"text": "alpha"}, {"text": "alpha"}],
        np.array(["alpha", "alpha"]),
    ],
)
def test_reranking_keeps_nonduplicate_and_other_input_forms(stsb_bert_tiny_model, monkeypatch, inputs):
    model = stsb_bert_tiny_model
    encode = model.encode_document
    expected = encode(inputs, convert_to_tensor=True, batch_size=2)
    seen = []

    def record(sentences, **kwargs):
        seen.append(sentences)
        return encode(sentences, **kwargs)

    monkeypatch.setattr(model, "encode_document", record)
    actual = make_evaluator().embed_inputs(model, inputs, encode_fn_name="document")
    assert len(seen) == 1
    assert seen[0] is inputs
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reranking_preserves_unsupported_message_error(stsb_bert_tiny_model, monkeypatch):
    model = stsb_bert_tiny_model
    inputs = [{"role": "user", "content": "alpha"}]
    encode = model.encode_document
    with pytest.raises(ValueError, match="does not support chat-style") as expected:
        encode(inputs, convert_to_tensor=True, batch_size=2)

    def record(sentences, **kwargs):
        assert sentences is inputs
        return encode(sentences, **kwargs)

    monkeypatch.setattr(model, "encode_document", record)
    with pytest.raises(ValueError) as actual:
        make_evaluator().embed_inputs(model, inputs, encode_fn_name="document")
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("model_type", ["sparse", "multi_vector"])
def test_reranking_subclass_embedding_overrides_are_unchanged(model, monkeypatch, model_type):
    if model_type == "sparse":
        encoder = SparseEncoder(modules=list(model.children()), device="cpu")
        evaluator_type = SparseRerankingEvaluator
    else:
        words = WordEmbeddings(
            WhitespaceTokenizer(["[UNK]", "alpha", "beta", "gamma"]),
            torch.rand(4, 8, generator=torch.Generator().manual_seed(5)),
        )
        encoder = MultiVectorEncoder(modules=[words], device="cpu")
        evaluator_type = MultiVectorRerankingEvaluator
    samples = [{"query": "alpha", "positive": ["beta", "beta"], "negative": ["gamma", "gamma"]}]
    encode = encoder.encode_document
    seen = []

    def record(sentences, **kwargs):
        seen.append(sentences)
        return encode(sentences, **kwargs)

    monkeypatch.setattr(encoder, "encode_document", record)
    metrics = evaluator_type(samples, write_csv=False)(encoder)
    assert seen == [["beta", "beta", "gamma", "gamma"]]
    assert 0 <= metrics["ndcg@10"] <= 1
