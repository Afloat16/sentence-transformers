from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MarginMSELoss
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.util import mine_hard_negatives

pytest.importorskip("datasets", exc_type=ModuleNotFoundError)

from datasets import Dataset


def _model(similarity: str = "dot") -> SentenceTransformer:
    vocabulary = {"[UNK]": 0, "query": 1, "positive": 2, "hard": 3, "easy": 4}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    weights = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [2.0, 2.0], [0.1, 0.0]])
    return SentenceTransformer(
        modules=[StaticEmbedding(tokenizer, embedding_weights=weights)],
        device="cpu",
        similarity_fn_name=similarity,
    )


def _mine(model: SentenceTransformer, use_faiss: bool = False, **kwargs) -> Dataset:
    if use_faiss:
        pytest.importorskip("faiss", exc_type=ModuleNotFoundError)
    return mine_hard_negatives(
        Dataset.from_dict({"query": ["query"], "positive": ["positive"]}),
        model,
        corpus=["hard", "easy"],
        range_max=3,
        num_negatives=1,
        output_scores=True,
        verbose=False,
        use_faiss=use_faiss,
        **kwargs,
    )


@pytest.mark.parametrize("use_faiss", [False, True])
@pytest.mark.parametrize("similarity", ["dot", "cosine"])
def test_mining_uses_the_models_dot_or_cosine_ranking(similarity: str, use_faiss: bool) -> None:
    model = _model(similarity)
    row = _mine(model, use_faiss)[0]

    # Dot product ranks the longer [2, 2] vector above [0.1, 0], whereas cosine does the reverse.
    assert row["negative"] == ("hard" if similarity == "dot" else "easy")
    assert row["scores"] == pytest.approx([3.0, 2.0] if similarity == "dot" else [1.0, 1.0])

    # Check the reported scores against normal inference, without imposing any extra normalization.
    query = model.encode_query([row["query"]])
    documents = model.encode_document([row["positive"], row["negative"]])
    assert row["scores"] == pytest.approx(model.similarity(query, documents)[0].tolist())


@pytest.mark.parametrize("use_faiss", [False, True])
@pytest.mark.parametrize(
    ("filters", "negative", "score"),
    [
        ({"min_score": 1.5}, "hard", 2.0),
        ({"absolute_margin": 1.5}, "easy", 0.1),
        ({"relative_margin": 0.5}, "easy", 0.1),
    ],
)
def test_dot_mining_filters_use_the_models_score_scale(filters, negative: str, score: float, use_faiss: bool) -> None:
    mined = _mine(_model(), use_faiss, **filters)

    assert len(mined) == 1
    assert mined[0]["negative"] == negative
    assert mined[0]["scores"] == pytest.approx([3.0, score])


@pytest.mark.parametrize("similarities", [("dot", "cosine"), ("cosine", "dot")])
def test_dot_and_cosine_mining_do_not_share_normalized_cache_entries(tmp_path, similarities) -> None:
    model = _model()
    for similarity in similarities:
        model.similarity_fn_name = similarity
        for _ in range(2):
            row = _mine(model, cache_folder=str(tmp_path))[0]
            assert row["negative"] == ("hard" if similarity == "dot" else "easy")
            assert row["scores"] == pytest.approx([3.0, 2.0] if similarity == "dot" else [1.0, 1.0])


def test_dot_mining_scores_are_valid_margin_distillation_targets() -> None:
    model = _model()
    row = _mine(model)[0]
    features = [model.preprocess([row[column]]) for column in ("query", "positive", "negative")]
    loss = MarginMSELoss(model)(features, torch.tensor([row["scores"]]))

    # A student identical to the mining teacher should already reproduce the teacher's margin.
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    torch.testing.assert_close(model[0].embedding.weight.grad, torch.zeros_like(model[0].embedding.weight))
