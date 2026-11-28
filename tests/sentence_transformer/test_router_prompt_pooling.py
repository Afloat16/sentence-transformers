from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from sentence_transformers import SentenceTransformer, SparseEncoder
from sentence_transformers.base.modules import Router, Transformer
from sentence_transformers.sentence_transformer.modules import Pooling
from sentence_transformers.sparse_encoder.modules import SparseAutoEncoder


@pytest.fixture(params=["plain", "router", "nested_router"])
def local_routed_model(tmp_path, request, model_class):
    vocab = {
        token: index
        for index, token in enumerate(
            ["[PAD]", "[UNK]", "query", "document", "instruction", "short", "longer", "sentence"]
        )
    }
    checkpoint = tmp_path / "backbone"
    torch.manual_seed(42)
    BertModel(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
    ).save_pretrained(checkpoint)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", model_max_length=128
    ).save_pretrained(checkpoint)

    autoencoder = SparseAutoEncoder(8, hidden_dim=16, k=8, k_aux=8) if model_class is SparseEncoder else None

    def route():
        modules = [Transformer(str(checkpoint)), Pooling(8)]
        if autoencoder is not None:
            modules.append(autoencoder)
        if request.param == "nested_router":
            modules = [Router({"text": modules})]
        return modules

    modules = route() if request.param == "plain" else [Router.for_query_document(route(), route())]
    model = model_class(
        modules=modules,
        prompts={"query": "query instruction ", "document": "document instruction "},
        device="cpu",
    )
    return model, autoencoder


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("task", ["query", "document"])
@pytest.mark.parametrize("model_class", [SentenceTransformer, SparseEncoder])
def test_prompt_pooling_setting_reaches_all_routes(local_routed_model, padding_side, task, model_class, tmp_path):
    model, autoencoder = local_routed_model
    for module in model.modules():
        if isinstance(module, Transformer):
            module.tokenizer.padding_side = padding_side
    sentences = ["short", "longer sentence"]
    encode = getattr(model, f"encode_{task}")
    encode_kwargs = {"convert_to_sparse_tensor": False} if autoencoder is not None else {"convert_to_tensor": True}
    including_prompt = encode(sentences, **encode_kwargs)

    # The two prompt tokens still condition the backbone. Only their contribution
    # to the final mean is excluded, using the actual contextual token embeddings.
    with torch.inference_mode():
        features = model.preprocess(sentences, prompt=model.prompts[task], task=task)
        output = model(features, task=task)
        expected = torch.stack(
            [
                tokens[mask.bool()][2:].mean(dim=0)
                for tokens, mask in zip(output["token_embeddings"], output["attention_mask"])
            ]
        )
        if autoencoder is not None:
            expected = autoencoder({"sentence_embedding": expected})["sentence_embedding"]
    assert not torch.allclose(including_prompt, expected)

    model.set_pooling_include_prompt(False)
    excluding_prompt = encode(sentences, **encode_kwargs)
    torch.testing.assert_close(excluding_prompt, expected)

    # Persist the actual route configuration and verify the public loading path.
    saved_model = tmp_path / "saved"
    model.save_pretrained(str(saved_model), create_model_card=False)
    reloaded = model_class(str(saved_model), local_files_only=True)
    torch.testing.assert_close(getattr(reloaded, f"encode_{task}")(sentences, **encode_kwargs), expected)

    model.set_pooling_include_prompt(True)
    torch.testing.assert_close(encode(sentences, **encode_kwargs), including_prompt)
