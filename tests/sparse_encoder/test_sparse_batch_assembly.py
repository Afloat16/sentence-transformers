from __future__ import annotations

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding, SpladePooling, Transformer
from sentence_transformers.util import dot_score, semantic_search


def make_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {"[PAD]": 0, "[UNK]": 1, **{f"t{i}": i for i in range(2, 16)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", model_max_length=32)


@pytest.fixture
def static_encoder() -> SparseEncoder:
    return SparseEncoder(
        modules=[SparseStaticEmbedding(make_tokenizer(), weight=torch.arange(16, dtype=torch.float32) - 4)],
        device="cpu",
    )


def expected_static(texts: list[str], weights: torch.Tensor) -> torch.Tensor:
    expected = torch.zeros(len(texts), len(weights), dtype=weights.dtype)
    for row, text in enumerate(texts):
        for token in set(text.split()):
            index = int(token[1:]) if token.startswith("t") else 1
            expected[row, index] = weights[index]
    return expected


def as_dense(output: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(output, list):
        return torch.stack([row.to_dense() for row in output])
    return output.to_dense()


@pytest.mark.parametrize("batch_size", [1, 3, 16])
@pytest.mark.parametrize("sparse,tensor", [(True, True), (True, False), (False, True), (False, False)])
def test_output_order_values_and_format(static_encoder: SparseEncoder, batch_size: int, sparse: bool, tensor: bool):
    texts = ["t9", "t2 t7 t6 t9", "", "t3 t8", "t9", "unknown t4 t12", "t15 t10 t8"]
    output = static_encoder.encode(
        np.array(texts),
        batch_size=batch_size,
        convert_to_sparse_tensor=sparse,
        convert_to_tensor=tensor,
        show_progress_bar=False,
    )
    assert isinstance(output, torch.Tensor) if tensor else isinstance(output, list)
    assert all(row.is_sparse == sparse for row in (list(output) if tensor else output))
    expected = expected_static(texts, static_encoder[0].weight.detach())
    torch.testing.assert_close(as_dense(output), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("save_to_cpu", [False, True])
def test_dtype_and_output_device(static_encoder: SparseEncoder, dtype: torch.dtype, save_to_cpu: bool):
    model = static_encoder.to(dtype=dtype)
    texts = ["t15 t3", "t2", "t8 t10 t12", ""]
    output = model.encode(texts, batch_size=3, save_to_cpu=save_to_cpu, show_progress_bar=False)
    assert output.dtype == dtype
    assert output.device.type == "cpu"
    torch.testing.assert_close(output.to_dense(), expected_static(texts, model[0].weight.detach()), rtol=0, atol=0)


@pytest.mark.parametrize("sparse,tensor", [(True, True), (True, False), (False, True), (False, False)])
def test_single_input_format(static_encoder: SparseEncoder, sparse: bool, tensor: bool):
    output = static_encoder.encode(
        "t11 t3 t9", convert_to_sparse_tensor=sparse, convert_to_tensor=tensor, show_progress_bar=False
    )
    assert isinstance(output, torch.Tensor)
    assert output.is_sparse == sparse
    torch.testing.assert_close(
        output.to_dense(), expected_static(["t11 t3 t9"], static_encoder[0].weight.detach())[0], rtol=0, atol=0
    )


@pytest.mark.parametrize("batch_size", [1, 3])
def test_all_zero_sparse_batches(static_encoder: SparseEncoder, batch_size: int):
    with torch.no_grad():
        static_encoder[0].weight.zero_()
    output = static_encoder.encode(["t6", "t7 t8", "", "t9 t10 t11"], batch_size=batch_size, show_progress_bar=False)
    assert output.shape == (4, 16)
    assert output.is_sparse
    assert output._nnz() == 0


def test_prompt_active_dimensions_and_retrieval(static_encoder: SparseEncoder):
    documents = ["t2", "t6 t7 t8", "t10 t11", "t6"]
    corpus = static_encoder.encode_document(
        documents, prompt="t15 ", max_active_dims=2, batch_size=3, show_progress_bar=False
    )
    expected = torch.zeros(4, 16)
    expected[:, 15] = 11
    expected[0, 2] = -2
    expected[1, 8] = 4
    expected[2, 11] = 7
    expected[3, 6] = 2
    torch.testing.assert_close(corpus.to_dense(), expected, rtol=0, atol=0)
    query = static_encoder.encode_query(["t11"], show_progress_bar=False)
    hits = semantic_search(query, corpus, top_k=1, score_function=dot_score)
    assert hits == [[{"corpus_id": 2, "score": 49.0}]]


def test_sparse_list_rows_remain_independent(static_encoder: SparseEncoder):
    output = static_encoder.encode(
        ["t8 t10", "t8 t10", "t6"], batch_size=2, convert_to_tensor=False, show_progress_bar=False
    )
    second = output[1].to_dense().clone()
    output[0]._values().zero_()
    torch.testing.assert_close(output[1].to_dense(), second, rtol=0, atol=0)


@pytest.fixture
def local_splade(tmp_path) -> SparseEncoder:
    tokenizer = make_tokenizer()
    tokenizer.save_pretrained(tmp_path)
    with torch.random.fork_rng():
        torch.manual_seed(1729)
        BertForMaskedLM(
            BertConfig(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0,
            )
        ).save_pretrained(tmp_path)
    return SparseEncoder(
        modules=[Transformer(str(tmp_path), transformer_task="fill-mask"), SpladePooling()], device="cpu"
    ).eval()


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_splade_matches_masked_language_model_pooling(local_splade: SparseEncoder, batch_size: int):
    texts = ["t6 t7", "t3", "t8 t12 t9 t5", "t13 t10 t4", "t3"]
    tokens = local_splade.tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        logits = local_splade[0].auto_model(**tokens).logits
        weights = torch.log1p(torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1)
        expected = weights.max(dim=1).values
    output = local_splade.encode(texts, batch_size=batch_size, show_progress_bar=False)
    assert output.is_sparse
    torch.testing.assert_close(output.to_dense(), expected, rtol=1e-5, atol=1e-7)
