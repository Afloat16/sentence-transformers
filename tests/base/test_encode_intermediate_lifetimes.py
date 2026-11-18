from __future__ import annotations

import weakref

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from sentence_transformers import MultiVectorEncoder, SentenceTransformer
from sentence_transformers.base.modules import Module, Normalize, Transformer
from sentence_transformers.multi_vector_encoder.modules import LambdaTokenPooling, MultiVectorMask
from sentence_transformers.sentence_transformer.modules import Pooling


@pytest.fixture(scope="module")
def local_bert_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("encode-lifetimes-bert")
    vocab = {word: i for i, word in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "one", "two", "three"])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=32,
    ).save_pretrained(path)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        BertModel(
            BertConfig(
                vocab_size=len(vocab),
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=48,
                max_position_embeddings=32,
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0,
            )
        ).save_pretrained(path)
    return str(path)


def make_model(path, kind):
    transformer = Transformer(path, model_kwargs={"attn_implementation": "eager"})
    if kind == "multi_vector":
        return MultiVectorEncoder(
            modules=[transformer, Normalize(module_input_name="token_embeddings"), MultiVectorMask()], device="cpu"
        )
    return SentenceTransformer(modules=[transformer, Pooling(32)], device="cpu")


TEXTS = ["one", "one two three two one", "three two", "two three one", "two"]
CASES = [
    pytest.param("sentence", {}, id="sentence-numpy"),
    pytest.param("sentence", {"convert_to_tensor": True}, id="sentence-tensor"),
    pytest.param("sentence", {"convert_to_numpy": False}, id="sentence-list"),
    pytest.param("sentence", {"truncate_dim": 16, "normalize_embeddings": True}, id="sentence-normalized-truncated"),
    pytest.param("token", {"output_value": "token_embeddings"}, id="sentence-token-output"),
    pytest.param("multi_vector", {}, id="multi-vector-tensor"),
    pytest.param("multi_vector", {"convert_to_numpy": True}, id="multi-vector-numpy"),
    pytest.param("multi_vector", {"truncate_dim": 16, "normalize_embeddings": True}, id="multi-vector-truncated"),
    pytest.param(
        "multi_vector",
        {"token_pooling": LambdaTokenPooling(lambda embeddings: embeddings[:2])},
        id="multi-vector-pooling-view",
    ),
]


@pytest.mark.parametrize(("kind", "options"), CASES)
@pytest.mark.parametrize("count", [1, 5], ids=["single-batch", "multiple-batches"])
def test_encode_releases_previous_token_batch(local_bert_path, kind, options, count):
    model = make_model(local_bert_path, kind)
    texts = TEXTS[:count]
    with torch.inference_mode():
        inputs = model.tokenizer(texts, padding=True, return_tensors="pt")
        hidden = model[0].model(**inputs).last_hidden_state
        tokens = [row[mask.bool()] for row, mask in zip(hidden, inputs["attention_mask"])]
        if kind == "sentence":
            expected = [row.mean(dim=0) for row in tokens]
        elif kind == "multi_vector":
            expected = [torch.nn.functional.normalize(row, dim=-1) for row in tokens]
        else:
            expected = tokens
        if "token_pooling" in options:
            expected = [row[:2] for row in expected]
        if "truncate_dim" in options:
            expected = [row[..., : options["truncate_dim"]] for row in expected]
        if options.get("normalize_embeddings") or (kind == "multi_vector" and "truncate_dim" in options):
            expected = [torch.nn.functional.normalize(row, dim=-1) for row in expected]

    storage_refs = []
    previous_batch_alive = []

    def before_forward(module, args):
        if storage_refs:
            previous_batch_alive.append(storage_refs[-1]() is not None)

    def after_forward(module, args, features):
        # A last-row view can keep storage alive even after its parent Tensor is gone.
        storage_refs.append(weakref.ref(features["token_embeddings"].untyped_storage()))

    before = model.register_forward_pre_hook(before_forward)
    after = model.register_forward_hook(after_forward)
    try:
        actual = model.encode(texts, batch_size=2, show_progress_bar=False, **options)
    finally:
        before.remove()
        after.remove()

    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected):
        value = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        torch.testing.assert_close(value, reference)
    assert previous_batch_alive == [False] * ((count + 1) // 2 - 1)
    assert all(ref() is None for ref in storage_refs)


@pytest.mark.parametrize("kind", ["sentence", "multi_vector"])
@pytest.mark.parametrize("singular", [False, True])
def test_raw_encode_outputs_remain_valid(local_bert_path, kind, singular):
    model = make_model(local_bert_path, kind)
    texts = TEXTS[0] if singular else TEXTS
    outputs = model.encode(texts, output_value=None, batch_size=2, show_progress_bar=False)
    rows = [outputs] if singular else outputs
    snapshots = [{key: value.clone() for key, value in row.items() if isinstance(value, torch.Tensor)} for row in rows]
    model.encode(TEXTS[::-1], batch_size=2, show_progress_bar=False)
    for row, snapshot in zip(rows, snapshots):
        for key, value in snapshot.items():
            torch.testing.assert_close(row[key], value, rtol=0, atol=0)
    assert len(rows) == (1 if singular else len(TEXTS))


class FirstTokenView(Module):
    def forward(self, features):
        features["sentence_embedding"] = features["token_embeddings"][:, 0]
        return features

    def save(self, output_path, **kwargs):
        self.save_config(output_path)


def test_sentence_output_can_share_token_storage(local_bert_path):
    model = SentenceTransformer(modules=[Transformer(local_bert_path), FirstTokenView()], device="cpu")
    with torch.inference_mode():
        inputs = model.tokenizer(TEXTS, padding=True, return_tensors="pt")
        expected = model[0].model(**inputs).last_hidden_state[:, 0]
    actual = model.encode(TEXTS, batch_size=2, convert_to_tensor=True, show_progress_bar=False)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["sentence", "multi_vector"])
def test_empty_encode_does_not_need_batch_locals(local_bert_path, kind):
    model = make_model(local_bert_path, kind)
    assert len(model.encode([], show_progress_bar=False)) == 0
