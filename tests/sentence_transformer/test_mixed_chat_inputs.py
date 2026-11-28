from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from sentence_transformers import SentenceTransformer


@pytest.fixture(params=["flat", "structured"])
def local_chat_model(tmp_path, request):
    vocab = {
        word: index
        for index, word in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "system",
                "user",
                "assistant",
                "query",
                "document",
                "plain",
                "question",
                "answer",
                "instruction",
                ":",
            ]
        )
    }
    path = tmp_path / "model"
    BertModel(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
    ).save_pretrained(path)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    content = (
        "{{ message.content }}"
        if request.param == "flat"
        else "{% for item in message.content %}{% if item.type == 'text' %}{{ item.text }}{% endif %}{% endfor %}"
    )
    template = "{% for message in messages %}{{ message.role }} : " + content + " {% endfor %}"
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        model_max_length=128,
        chat_template=template,
    ).save_pretrained(path)
    return SentenceTransformer(str(path), local_files_only=True)


@pytest.mark.parametrize("single_message", [False, True])
@pytest.mark.parametrize("output", ["tokens", "embeddings"])
@pytest.mark.parametrize("text_input", ["plain", ("question", "answer")])
def test_mixed_chat_inputs_preserve_tokens_and_embeddings(local_chat_model, single_message, output, text_input):
    model = local_chat_model

    def message(role, content):
        if model[0].input_formatter.message_format == "structured":
            content = [{"type": "text", "text": content}]
        return {"role": role, "content": content}

    conversation = (
        message("assistant", "answer")
        if single_message
        else [
            message("system", "instruction"),
            message("user", "question"),
            message("assistant", "answer"),
        ]
    )
    inputs = [text_input, conversation, "question"]
    original = deepcopy(inputs)

    if output == "tokens":
        mixed = model.preprocess(inputs)
        for index, value in enumerate(inputs):
            separate = model.preprocess([value])
            actual_ids = mixed["input_ids"][index][mixed["attention_mask"][index].bool()].tolist()
            expected_ids = separate["input_ids"][0][separate["attention_mask"][0].bool()].tolist()
            assert actual_ids == expected_ids
    else:
        np.testing.assert_allclose(
            model.encode(inputs, batch_size=3), model.encode(inputs, batch_size=1), rtol=1e-5, atol=1e-6
        )
    assert inputs == original
