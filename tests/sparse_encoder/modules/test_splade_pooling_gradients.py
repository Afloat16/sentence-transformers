from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.losses import SparseMSELoss
from sentence_transformers.sparse_encoder.modules import SpladePooling


def _reference(logits, mask, activation):
    transformed = (logits * mask.unsqueeze(-1)).relu().log1p()
    if activation == "log1p_relu":
        transformed = transformed.log1p()
    return transformed.max(dim=1).values


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", ["relu", "log1p_relu"])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 20, None])
def test_chunked_max_keeps_unchunked_values_and_gradients(dtype, activation, chunk_size):
    # Positive maxima repeat across chunk boundaries. Masked maxima must not win,
    # and a column containing only negative logits must retain a zero gradient.
    logits = torch.tensor(
        [
            [[3, 1, -1, 2], [9, 9, 9, 9], [3, 4, -2, 1], [1, 4, -3, 5], [3, 4, -1, 5], [9, 9, 9, 9], [3, 0, -2, 5]],
            [[1, 3, -2, 5], [3, 3, -1, 1], [9, 9, 9, 9], [3, 3, -2, 5], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    mask = torch.tensor([[1, 0, 1, 1, 1, 0, 1], [1, 1, 0, 1, 0, 0, 0]])
    expected_input = logits.detach().clone().requires_grad_()
    weights = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=dtype)
    expected = _reference(expected_input, mask, activation)
    (expected * weights).sum().backward()

    pooling = SpladePooling(activation_function=activation, chunk_size=chunk_size)
    actual = pooling({"token_embeddings": logits, "attention_mask": mask})["sentence_embedding"]
    (actual * weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, expected_input.grad, rtol=0, atol=0)


@pytest.mark.parametrize("activation", ["relu", "log1p_relu"])
@pytest.mark.parametrize("pooling_strategy", ["max", "sum"])
def test_flattened_pooling_keeps_padded_gradients(activation, pooling_strategy):
    segments = [torch.tensor([[3.0, 1.0], [3.0, 2.0], [3.0, 2.0]]), torch.tensor([[2.0, 4.0], [2.0, 4.0]])]
    padded = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True).requires_grad_()
    flat = torch.cat(segments).unsqueeze(0).requires_grad_()
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    pooling = SpladePooling(pooling_strategy=pooling_strategy, activation_function=activation)

    expected = pooling({"token_embeddings": padded, "attention_mask": mask})["sentence_embedding"]
    actual = pooling({"token_embeddings": flat, "cu_seq_lens_q": torch.tensor([0, 3, 5])})["sentence_embedding"]
    expected.sum().backward()
    actual.sum().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(flat.grad[0], padded.grad[mask.bool()], rtol=0, atol=0)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, None])
@pytest.mark.parametrize(
    "values", [[1.0, float("nan"), 3.0], [float("nan"), 2.0, float("nan")], [float("inf"), 1.0, float("inf")]]
)
def test_chunked_max_preserves_nonfinite_propagation(values, chunk_size):
    logits = torch.tensor(values).reshape(1, -1, 1).requires_grad_()
    expected_input = logits.detach().clone().requires_grad_()
    mask = torch.ones(1, len(values))
    expected = _reference(expected_input, mask, "relu")
    actual = SpladePooling(chunk_size=chunk_size)({"token_embeddings": logits, "attention_mask": mask})[
        "sentence_embedding"
    ]
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, equal_nan=True, rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, expected_input.grad, equal_nan=True, rtol=0, atol=0)


def _bert_sparse_model(path):
    torch.manual_seed(42)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    vocab.update({f"word{index}": index for index in range(5, 97)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        model_max_length=128,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    transformer = BertForMaskedLM(
        BertConfig(
            vocab_size=97,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=48,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )
    token_ids = torch.randint(5, 97, (2, 64))
    texts = [" ".join(f"word{index}" for index in row.tolist()) for row in token_ids]
    transformer.save_pretrained(path)
    tokenizer.save_pretrained(path)
    model = SparseEncoder(str(path), device="cpu", local_files_only=True).to(torch.bfloat16)
    return model, texts


@pytest.mark.parametrize("chunk_size", [1, 8, 16, 64])
def test_bfloat16_sparse_encoder_update_matches_unchunked_training(tmp_path, chunk_size):
    # Low-precision MLM logits produce positive ties in ordinary initialized
    # BERT weights. Changing the pooling memory budget must preserve the update.
    expected_model, texts = _bert_sparse_model(tmp_path / "bert")
    actual_model = deepcopy(expected_model)
    actual_model.splade_pooling_chunk_size = chunk_size
    gradients = []
    losses = []
    initial = {name: parameter.detach().clone() for name, parameter in expected_model.named_parameters()}
    for model in (expected_model, actual_model):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        loss = SparseMSELoss(model)([model.preprocess(texts)], torch.zeros(2, 97))
        loss.backward()
        losses.append(loss.detach())
        gradients.append(
            {
                name: parameter.grad.clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        )
        optimizer.step()

    torch.testing.assert_close(losses[1], losses[0], rtol=0, atol=0)
    torch.testing.assert_close(gradients[1], gradients[0], rtol=0, atol=0)
    for (name, expected), (actual_name, actual) in zip(
        expected_model.named_parameters(), actual_model.named_parameters()
    ):
        assert actual_name == name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert any(not torch.equal(parameter, initial[name]) for name, parameter in expected_model.named_parameters())
