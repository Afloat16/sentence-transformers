from __future__ import annotations

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import BoW


@pytest.mark.parametrize("cumulative", [True, False])
def test_bow_weighted_token_frequencies(cumulative):
    bow = BoW(
        ["alpha", "beta", "zero", "gamma"],
        word_weights={"alpha": 0.25, "beta": -0.5, "zero": 0.0, "gamma": 1.25},
        cumulative_term_frequency=cumulative,
    )
    features = bow.get_sentence_features([[2, 0, 2, 1, 0], [], [3] * 4096])
    expected = torch.tensor(
        [[0.5 if cumulative else 0.25, -0.5, 0.0, 0.0], [0.0] * 4, [0.0, 0.0, 0.0, 5120.0 if cumulative else 1.25]]
    )
    torch.testing.assert_close(features["sentence_embedding"], expected, rtol=0, atol=0)


@pytest.mark.parametrize("cumulative", [True, False])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("reload", [False, True])
def test_bow_encode_weighted_and_empty_sentences(tmp_path, cumulative, batch_size, reload):
    bow = BoW(
        ["Alpha", "beta", "zero", "Gamma", "unweighted", "Alpha"],
        word_weights={"alpha": 0.25, "beta": -0.5, "zero": 0.0, "Gamma": 2.0},
        unknown_word_weight=1.5,
        cumulative_term_frequency=cumulative,
    )
    model = SentenceTransformer(modules=[bow], device="cpu")
    if reload:
        model.save(str(tmp_path))
        model = SentenceTransformer(str(tmp_path), device="cpu")
    texts = ["Alpha Alpha beta zero Gamma unweighted", "beta beta Alpha", "", "outside"]
    expected = torch.tensor(
        [
            [0.5 if cumulative else 0.25, -0.5, 0.0, 2.0, 1.5],
            [0.25, -1.0 if cumulative else -0.5, 0.0, 0.0, 0.0],
            [0.0] * 5,
            [0.0] * 5,
        ]
    )
    actual = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
