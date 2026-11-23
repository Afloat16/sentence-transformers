from __future__ import annotations

import json

import pytest
import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorInformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator
from sentence_transformers.util import dot_score, maxsim


class _TableModel:
    prompts = {}
    sparsity = staticmethod(SparseEncoder.sparsity)

    def __init__(self, layout, vectors):
        self.layout = layout
        self.vectors = vectors
        self.query_calls = 0
        self.document_calls = 0

    def encode(self, sentences, **kwargs):
        rows = [self.vectors[sentence] for sentence in sentences]
        if self.layout == "multi":
            return rows
        embeddings = torch.stack(rows)
        return embeddings.to_sparse() if self.layout == "sparse" else embeddings

    def encode_query(self, sentences, **kwargs):
        self.query_calls += 1
        return self.encode(sentences, **kwargs)

    def encode_document(self, sentences, **kwargs):
        self.document_calls += 1
        return self.encode(sentences, **kwargs)


@pytest.mark.parametrize("layout", ["dense", "sparse", "multi"])
@pytest.mark.parametrize("query_chunk_size", [1, 4, 64])
@pytest.mark.parametrize("corpus_chunk_size", [2, 20])
@pytest.mark.parametrize("precomputed", [False, True])
def test_query_chunks_preserve_rankings_and_encode_once(
    layout, query_chunk_size, corpus_chunk_size, precomputed, tmp_path
):
    generator = torch.Generator().manual_seed(13)
    # Reversed IDs, exact duplicates, zero vectors and negative scores exercise cutoff ties.
    queries = {f"q{i}": f"query-{i}" for i in range(7)}
    corpus = {f"d{i:02d}": f"document-{i}" for i in reversed(range(11))}
    vectors = {
        text: torch.randint(-2, 3, (5,), generator=generator).float() for text in [*queries.values(), *corpus.values()]
    }
    vectors["query-0"] = torch.zeros(5)
    vectors["document-0"] = vectors["document-1"].clone()
    if layout == "multi":
        vectors = {text: value.repeat(index % 3 + 1, 1) for index, (text, value) in enumerate(vectors.items())}
    model = _TableModel(layout, vectors)
    score_fn = maxsim if layout == "multi" else dot_score
    full_queries = model.encode(list(queries.values()))
    full_corpus = model.encode(list(corpus.values()))
    expected_scores = score_fn(full_queries, full_corpus).tolist()
    qrels = {query: {list(corpus)[index]} for index, query in enumerate(queries)}
    calls = []

    def bounded_score(query_embeddings, corpus_embeddings):
        calls.append((len(query_embeddings), len(corpus_embeddings)))
        assert len(query_embeddings) <= query_chunk_size
        return score_fn(query_embeddings, corpus_embeddings)

    evaluator_cls = {
        "dense": InformationRetrievalEvaluator,
        "sparse": SparseInformationRetrievalEvaluator,
        "multi": MultiVectorInformationRetrievalEvaluator,
    }[layout]
    evaluator = evaluator_cls(
        # A query without qrels is filtered out before chunk offsets are assigned.
        queries={"ignored": "not encoded", **queries},
        corpus=corpus,
        relevant_docs=qrels,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
        score_functions={"score": bounded_score, "reverse": lambda a, b: -bounded_score(a, b)},
        accuracy_at_k=[1, 3],
        precision_recall_at_k=[1, 3],
        mrr_at_k=[3],
        ndcg_at_k=[3],
        map_at_k=[3],
        write_csv=False,
        write_predictions=True,
    )
    metrics = evaluator.evaluate(
        model, output_path=str(tmp_path), corpus_embeddings=full_corpus if precomputed else None
    )
    assert model.query_calls == 1
    assert model.document_calls == (0 if precomputed else (len(corpus) + corpus_chunk_size - 1) // corpus_chunk_size)
    assert evaluator.get_config_dict()["query_chunk_size"] == query_chunk_size
    assert sum(rows * cols for rows, cols in calls) == 2 * len(queries) * len(corpus)
    for name, sign in [("score", 1), ("reverse", -1)]:
        expected_hits = [
            [
                {"corpus_id": corpus_id, "score": score}
                for score, corpus_id in sorted(
                    [(sign * score, corpus_id) for score, corpus_id in zip(row, corpus)],
                    key=lambda hit: (-hit[0], hit[1]),
                )[:3]
            ]
            for row in expected_scores
        ]
        expected_metrics = evaluator.compute_metrics(expected_hits)
        for metric, values in expected_metrics.items():
            for k, value in values.items():
                assert metrics[f"{name}_{metric.replace('@k', '@' + str(k))}"] == value
        prediction_path = tmp_path / evaluator.predictions_file.replace(".jsonl", f"_{name}.jsonl")
        predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
        assert [item["query_id"] for item in predictions] == list(queries)
        assert [item["results"] for item in predictions] == expected_hits


def test_default_query_chunks_bound_scoring():
    queries = {f"q{i}": f"q{i}" for i in range(205)}
    corpus = {"d": "d"}
    model = _TableModel("dense", {text: torch.ones(2) for text in [*queries, *corpus]})
    calls = []

    def score(a, b):
        calls.append(len(a))
        assert len(a) <= 100
        return dot_score(a, b)

    evaluator = InformationRetrievalEvaluator(
        queries, corpus, {qid: {"d"} for qid in queries}, score_functions={"dot": score}, write_csv=False
    )
    metrics = evaluator.evaluate(model)
    assert calls == [100, 100, 5]
    assert metrics["dot_accuracy@1"] == 1.0
    assert model.query_calls == model.document_calls == 1


def test_query_chunks_preserve_pretrained_model_metrics(stsb_bert_tiny_model):
    queries = {
        f"q{i}": text
        for i, text in enumerate(
            [
                "What is the capital of France?",
                "Who painted the Mona Lisa?",
                "What do pandas eat?",
            ]
            * 4
        )
    }
    corpus = {
        "paris": "Paris is the capital of France.",
        "berlin": "Berlin is the capital of Germany.",
        "leonardo": "Leonardo da Vinci painted the Mona Lisa.",
        "bamboo": "Pandas eat bamboo.",
        "paris-duplicate": "Paris is the capital of France.",
    }
    qrels = {query: [{"paris", "paris-duplicate"}, {"leonardo"}, {"bamboo"}][i % 3] for i, query in enumerate(queries)}
    kwargs = dict(queries=queries, corpus=corpus, relevant_docs=qrels, corpus_chunk_size=3, write_csv=False)
    expected = InformationRetrievalEvaluator(**kwargs, query_chunk_size=len(queries)).evaluate(stsb_bert_tiny_model)
    actual = InformationRetrievalEvaluator(**kwargs, query_chunk_size=1).evaluate(stsb_bert_tiny_model)
    assert actual == expected


@pytest.mark.parametrize("query_chunk_size", [0, -1])
def test_invalid_query_chunk_size(query_chunk_size):
    with pytest.raises(ValueError, match="query_chunk_size must be positive"):
        InformationRetrievalEvaluator(
            {"q": "query"}, {"d": "document"}, {"q": {"d"}}, query_chunk_size=query_chunk_size
        )
