from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules import Transformer
from sentence_transformers.sentence_transformer.losses import CachedGISTEmbedLoss, GISTEmbedLoss
from sentence_transformers.sentence_transformer.modules import Pooling


@pytest.fixture
def local_model(tmp_path):
    vocabulary = {
        word: index
        for index, word in enumerate(
            ["[PAD]", "[UNK]", "anchor", "positive", "different", "first", "second", "third", "fourth", "one", "two"]
        )
    }
    checkpoint = tmp_path / "backbone"
    torch.manual_seed(42)
    BertModel(
        BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            hidden_dropout_prob=0.5,
            attention_probs_dropout_prob=0.5,
        )
    ).save_pretrained(checkpoint)
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", model_max_length=32
    ).save_pretrained(checkpoint)
    model = SentenceTransformer(modules=[Transformer(str(checkpoint)), Pooling(16)], device="cpu")
    model.model_card_data.generate_widget_examples = False
    return model


@pytest.fixture(params=[GISTEmbedLoss, CachedGISTEmbedLoss])
def make_loss(request):
    def create(model, guide):
        kwargs = {"mini_batch_size": 2} if request.param is CachedGISTEmbedLoss else {}
        return request.param(model, guide, **kwargs)

    return create


def _features(model):
    return [
        model.preprocess(texts)
        for texts in (
            ["anchor first", "anchor second", "anchor third", "anchor fourth"],
            ["positive one", "different two", "positive third", "positive fourth"],
        )
    ]


def _modes(model):
    return {module: module.training for module in model.modules()}


def test_frozen_guide_keeps_filtering_deterministic(local_model, make_loss):
    model = local_model
    guide = deepcopy(model)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    loss_fn = make_loss(model, guide).train()
    original_modes = _modes(guide)
    assert not any(parameter.requires_grad for parameter in guide.parameters())
    guide_outputs, guide_modes, masks = [], [], []

    def capture_guide(module, args, output):
        guide_outputs.append(output["sentence_embedding"].detach().clone())
        guide_modes.append(module.training)

    guide.register_forward_hook(capture_guide)
    loss_fn.cross_entropy_loss.register_forward_pre_hook(
        lambda module, args: masks.append(torch.isfinite(args[0]).detach().clone())
    )

    values = []
    for _ in range(2):
        value = loss_fn(_features(model), labels=None)
        values.append(value.detach())
        value.backward()
        assert any(parameter.grad is not None and parameter.grad.count_nonzero() for parameter in model.parameters())
        assert all(parameter.grad is None for parameter in guide.parameters())
        assert _modes(guide) == original_modes
        model.zero_grad()

    half = len(guide_outputs) // 2
    torch.testing.assert_close(torch.cat(guide_outputs[:half]), torch.cat(guide_outputs[half:]), rtol=0, atol=0)
    half = len(masks) // 2
    assert torch.equal(torch.cat(masks[:half]), torch.cat(masks[half:]))
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    assert not any(guide_modes)


@pytest.mark.parametrize("sharing", ["same_model", "shared_backbone"])
def test_guide_restores_shared_student_modes_and_gradients(local_model, make_loss, sharing):
    model = local_model.train()
    guide = model if sharing == "same_model" else SentenceTransformer(modules=[model[0], Pooling(16)])
    loss_fn = make_loss(model, guide).train()
    # Preserve a deliberately mixed train/eval configuration, not only the root flag.
    model[0].auto_model.encoder.layer[0].attention.output.dropout.eval()
    original_modes = _modes(model) | _modes(guide)

    loss = loss_fn(_features(model), labels=None)
    assert _modes(model) | _modes(guide) == original_modes
    loss.backward()
    assert _modes(model) | _modes(guide) == original_modes
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.count_nonzero() for gradient in gradients)


def test_guide_restores_mixed_modes_on_error(local_model, make_loss):
    guide = deepcopy(local_model)
    loss_fn = make_loss(local_model, guide).train()
    guide[0].auto_model.encoder.layer[0].attention.output.dropout.eval()
    original_modes = _modes(guide)

    def fail_in_guide(module, args):
        assert not module.training
        raise RuntimeError("guide forward failed")

    guide.register_forward_pre_hook(fail_in_guide)
    with pytest.raises(RuntimeError, match="guide forward failed"):
        loss_fn(_features(local_model), labels=None)
    assert _modes(guide) == original_modes


@pytest.mark.parametrize("training_api", ["trainer", "old_fit"])
def test_training_updates_student_with_guide_in_eval(local_model, make_loss, training_api, tmp_path):
    from datasets import Dataset
    from torch.utils.data import DataLoader

    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.readers import InputExample

    model = local_model
    guide = deepcopy(model)
    loss_fn = make_loss(model, guide)
    guide.train()
    guide_modes = []
    guide.register_forward_pre_hook(lambda module, args: guide_modes.append(module.training))
    initial_weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    guide_weights = {name: parameter.detach().clone() for name, parameter in guide.named_parameters()}
    columns = {
        "anchor": ["anchor first", "anchor second", "anchor third", "anchor fourth"],
        "positive": ["positive one", "different two", "positive third", "positive fourth"],
    }
    if training_api == "old_fit":
        loader = DataLoader([InputExample(texts=list(pair)) for pair in zip(*columns.values())], batch_size=4)
        model.old_fit([(loader, loss_fn)], epochs=1, warmup_steps=0, show_progress_bar=False)
    else:
        trainer = SentenceTransformerTrainer(
            model=model,
            loss=loss_fn,
            train_dataset=Dataset.from_dict(columns),
            args=SentenceTransformerTrainingArguments(
                output_dir=str(tmp_path / "train"),
                use_cpu=True,
                max_steps=1,
                per_device_train_batch_size=4,
                save_strategy="no",
                report_to=[],
                disable_tqdm=True,
            ),
        )
        trainer.train()
    assert guide_modes and not any(guide_modes)
    assert any(not torch.equal(parameter, initial_weights[name]) for name, parameter in model.named_parameters())
    for name, parameter in guide.named_parameters():
        torch.testing.assert_close(parameter, guide_weights[name], rtol=0, atol=0)
