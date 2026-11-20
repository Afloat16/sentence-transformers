from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.modules import Transformer
from sentence_transformers.sentence_transformer.data_collator import SentenceTransformerDataCollator
from sentence_transformers.sentence_transformer.losses import MSELoss
from sentence_transformers.sentence_transformer.modules import Pooling, StaticEmbedding
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(reason="Sentence Transformers requires the train extra for these tests.", allow_module_level=True)

from datasets import Dataset


def _tokenizer(hello_id: int) -> Tokenizer:
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": hello_id, "world": 3 - hello_id, "[PAD]": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def _processor(hello_id: int) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=_tokenizer(hello_id), unk_token="[UNK]", pad_token="[PAD]", model_max_length=32
    )


def _initial_weights() -> torch.Tensor:
    return torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32


def _static_model(hello_id: int) -> SentenceTransformer:
    model = SentenceTransformer(
        modules=[StaticEmbedding(_tokenizer(hello_id), embedding_weights=_initial_weights())], device="cpu"
    )
    model.model_card_data.generate_widget_examples = False
    return model


def _training_args(tmp_path: Path, **kwargs) -> SentenceTransformerTrainingArguments:
    return SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        per_device_train_batch_size=1,
        learning_rate=0.1,
        optim="sgd",
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
        **kwargs,
    )


class CustomDefaultCollatorTrainer(SentenceTransformerTrainer):
    def get_data_collator(self, model, args, processing_class=None):
        return SentenceTransformerDataCollator(preprocess_fn=model.preprocess, valid_label_columns=["target"])


@pytest.mark.parametrize("trainer_class", [SentenceTransformerTrainer, CustomDefaultCollatorTrainer])
def test_model_init_refreshes_collator_before_training(tmp_path: Path, trainer_class) -> None:
    config = {"hello_id": 1}
    label_column = "target" if trainer_class is CustomDefaultCollatorTrainer else "label"
    trainer = trainer_class(
        model_init=lambda: _static_model(config["hello_id"]),
        args=_training_args(tmp_path),
        train_dataset=Dataset.from_dict({"sentence": ["hello"], label_column: [[1.0] * 8]}),
        loss=MSELoss,
    )

    for hello_id in (2, 1):
        config["hello_id"] = hello_id
        trainer.train()
        changed_rows = (next(trainer.model.parameters()).detach() != _initial_weights()).any(dim=1).nonzero().flatten()
        assert changed_rows.tolist() == [hello_id]
        batch = trainer.data_collator([{"sentence": "hello"}])
        assert batch["sentence_input_ids"].tolist() == [hello_id]


def test_model_init_constructor_uses_the_final_model_for_preprocessing(tmp_path: Path, transformer_factory) -> None:
    calls = []

    def model_init():
        calls.append(None)
        return transformer_factory(1 if len(calls) == 1 else 2)

    trainer = SentenceTransformerTrainer(model_init=model_init, args=_training_args(tmp_path), loss=MSELoss)
    expected = trainer.model.preprocess(["hello"])["input_ids"]
    actual = trainer.data_collator([{"sentence": "hello"}])["sentence_input_ids"]
    assert torch.equal(actual, expected)
    assert trainer.processing_class("hello")["input_ids"] == expected.flatten().tolist()
    assert trainer.callback_handler.processing_class("hello")["input_ids"] == expected.flatten().tolist()


@pytest.mark.parametrize("replace_after_init", [False, True])
def test_model_init_preserves_custom_collator(tmp_path: Path, replace_after_init: bool) -> None:
    custom_collator = SentenceTransformerDataCollator(preprocess_fn=_static_model(1).preprocess)
    trainer = SentenceTransformerTrainer(
        model_init=lambda: _static_model(2),
        args=_training_args(tmp_path),
        train_dataset=Dataset.from_dict({"sentence": ["hello"], "label": [[1.0] * 8]}),
        loss=MSELoss,
        data_collator=None if replace_after_init else custom_collator,
    )
    if replace_after_init:
        trainer.data_collator = custom_collator
    trainer.train()
    assert trainer.data_collator is custom_collator
    assert trainer.data_collator([{"sentence": "hello"}])["sentence_input_ids"].tolist() == [1]
    changed_rows = (next(trainer.model.parameters()).detach() != _initial_weights()).any(dim=1).nonzero().flatten()
    assert changed_rows.tolist() == [1]


@pytest.fixture
def transformer_factory(tmp_path: Path):
    backbone = tmp_path / "backbone"
    BertModel(
        BertConfig(vocab_size=4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16)
    ).save_pretrained(backbone)
    _processor(1).save_pretrained(backbone)

    def factory(hello_id):
        transformer = Transformer(str(backbone))
        transformer.processor = _processor(hello_id)
        model = SentenceTransformer(modules=[transformer, Pooling(8)], device="cpu")
        model.model_card_data.generate_widget_examples = False
        return model

    return factory


@pytest.mark.parametrize("initial_static,final_static", [(False, False), (True, False), (False, True)])
def test_model_init_checkpoint_keeps_current_processor(
    tmp_path: Path, transformer_factory, initial_static: bool, final_static: bool
) -> None:
    config = {"hello_id": 1, "static": initial_static}

    def model_init():
        return (_static_model if config["static"] else transformer_factory)(config["hello_id"])

    trainer = SentenceTransformerTrainer(
        model_init=model_init,
        args=_training_args(tmp_path),
        train_dataset=Dataset.from_dict({"sentence": ["hello"], "label": [[1.0] * 8]}),
        loss=MSELoss,
    )
    config.update(hello_id=2, static=final_static)
    trainer.train()
    output = tmp_path / "checkpoint"
    trainer.save_model(str(output))
    reloaded = SentenceTransformer(str(output), device="cpu")
    assert reloaded.preprocess(["hello"])["input_ids"].flatten().tolist() == [2]
    torch.testing.assert_close(reloaded.encode(["hello"]), trainer.model.encode(["hello"]))
    assert trainer.processing_class is (None if final_static else trainer.model.processor)
    assert trainer.callback_handler.processing_class is trainer.processing_class


@pytest.mark.parametrize("replace_after_init", [False, True])
def test_model_init_preserves_custom_processor(tmp_path: Path, transformer_factory, replace_after_init: bool) -> None:
    custom_processor = _processor(1)
    trainer = SentenceTransformerTrainer(
        model_init=lambda: transformer_factory(2),
        args=_training_args(tmp_path),
        loss=MSELoss,
        processing_class=None if replace_after_init else custom_processor,
    )
    if replace_after_init:
        trainer.processing_class = custom_processor
    trainer.model = trainer.call_model_init()
    assert trainer.processing_class is custom_processor
    assert trainer.processing_class("hello")["input_ids"] == [1]


def test_model_init_refreshes_cached_eval_collation(tmp_path: Path) -> None:
    config = {"hello_id": 1}
    dataset = Dataset.from_dict({"sentence": ["hello"], "label": [[1.0] * 8]})
    trainer = SentenceTransformerTrainer(
        model_init=lambda: _static_model(config["hello_id"]),
        args=_training_args(tmp_path, dataloader_num_workers=1, dataloader_persistent_workers=True),
        train_dataset=dataset,
        eval_dataset=dataset,
        loss=MSELoss,
    )
    original_loader = trainer.get_eval_dataloader()
    assert original_loader.collate_fn([{"sentence": "hello"}])["sentence_input_ids"].tolist() == [1]
    assert trainer.get_eval_dataloader() is original_loader
    config["hello_id"] = 2
    trainer.model = trainer.call_model_init()
    current_loader = trainer.get_eval_dataloader()
    assert current_loader.collate_fn([{"sentence": "hello"}])["sentence_input_ids"].tolist() == [2]
    assert current_loader is not original_loader
    assert trainer.get_eval_dataloader() is current_loader
