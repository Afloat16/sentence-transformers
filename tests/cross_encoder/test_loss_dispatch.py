from __future__ import annotations

from pathlib import Path

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss


@pytest.fixture
def local_cross_encoder(tmp_path: Path) -> CrossEncoder:
    path = tmp_path / "model"
    vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
    BertForSequenceClassification(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_labels=1,
        )
    ).save_pretrained(path)
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        model_max_length=16,
    ).save_pretrained(path)
    return CrossEncoder(str(path))


class MetadataLoss(BinaryCrossEntropyLoss):
    def __init__(self, model, fail=False):
        super().__init__(model)
        self.fail = fail
        self.calls = []
        self.failure = TypeError("custom loss postprocessing failed")

    def forward(self, inputs, labels, prompt=None, task=None):
        self.calls.append((prompt, task))
        loss = super().forward(inputs, labels, prompt=prompt, task=task)
        if self.fail and prompt is not None:
            raise self.failure
        return loss


class LegacyLoss(BinaryCrossEntropyLoss):
    def __init__(self, model):
        super().__init__(model)
        self.calls = 0

    def forward(self, inputs, labels):
        self.calls += 1
        return super().forward(inputs, labels)


class KeywordLoss(MetadataLoss):
    def forward(self, inputs, labels, **kwargs):
        return super().forward(inputs, labels, **kwargs)


def make_trainer(model, loss, tmp_path):
    return CrossEncoderTrainer(
        model=model,
        loss=loss,
        train_dataset=Dataset.from_dict(
            {
                "query": ["hello", "world"],
                "document": ["world", "hello"],
                "label": [1.0, 0.0],
            }
        ),
        args=CrossEncoderTrainingArguments(
            output_dir=str(tmp_path / "output"),
            max_steps=1,
            per_device_train_batch_size=2,
            save_strategy="no",
            report_to=[],
            use_cpu=True,
            prompts="Search: ",
        ),
    )


@pytest.mark.parametrize("loss_type", [MetadataLoss, KeywordLoss])
def test_loss_body_type_error_is_not_retried_without_metadata(local_cross_encoder, tmp_path, loss_type):
    loss = loss_type(local_cross_encoder, fail=True)
    trainer = make_trainer(local_cross_encoder, loss, tmp_path)
    with pytest.raises(TypeError, match="custom loss postprocessing failed") as raised:
        trainer.train()
    assert raised.value is loss.failure
    assert loss.calls == [("Search: ", None)]
    assert trainer.state.global_step == 0


@pytest.mark.parametrize("loss_type", [MetadataLoss, KeywordLoss, LegacyLoss])
def test_supported_loss_signatures_train_once(local_cross_encoder, tmp_path, loss_type):
    loss = loss_type(local_cross_encoder)
    trainer = make_trainer(local_cross_encoder, loss, tmp_path)
    weights = next(local_cross_encoder.parameters())
    before = weights.detach().clone()
    result = trainer.train()
    assert result.global_step == 1
    assert not torch.equal(weights, before)
    assert torch.isfinite(weights).all()
    assert loss.calls == (1 if loss_type is LegacyLoss else [("Search: ", None)])
