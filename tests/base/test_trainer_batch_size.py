from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import TrainerState

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MSELoss
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(reason="Sentence Transformers requires the train extra for these tests.", allow_module_level=True)

from datasets import Dataset, DatasetDict


@pytest.fixture
def model() -> SentenceTransformer:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, **{f"word{i}": i + 1 for i in range(8)}}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    weights = torch.arange(72, dtype=torch.float32).reshape(9, 8) / 72
    return SentenceTransformer(modules=[StaticEmbedding(tokenizer, embedding_weights=weights)], device="cpu")


def _dataset(kind: str):
    dataset = Dataset.from_dict({"sentence": [f"word{i}" for i in range(8)], "label": [[1.0] * 8] * 8})
    if kind == "dict":
        return DatasetDict({"first": dataset, "second": dataset})
    if kind == "iterable":
        return dataset.to_iterable_dataset()
    return dataset


def _args(tmp_path: Path, **kwargs) -> SentenceTransformerTrainingArguments:
    return SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=5,
        optim="sgd",
        learning_rate=0.1,
        use_cpu=True,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        **kwargs,
    )


class MemoryLimitedMSELoss(MSELoss):
    def __init__(self, model: SentenceTransformer, max_batch_size: int | None = 2, error: str = "CUDA out of memory."):
        super().__init__(model)
        self.max_batch_size = max_batch_size
        self.error = error
        self.batch_sizes: list[int] = []

    def forward(self, sentence_features, labels):
        self.batch_sizes.append(len(labels))
        if self.max_batch_size is not None and len(labels) > self.max_batch_size:
            # Exercise the real Accelerate retry loop without exhausting CPU or GPU memory.
            raise RuntimeError(self.error)
        return super().forward(sentence_features, labels)


@pytest.mark.parametrize("dataset_kind", ["map", "dict", "iterable"])
def test_auto_find_batch_size_rebuilds_loader_with_reduced_batches(model, tmp_path, dataset_kind):
    original = next(model.parameters()).detach().clone()
    loss = MemoryLimitedMSELoss(model)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=_args(tmp_path, auto_find_batch_size=True),
        train_dataset=_dataset(dataset_kind),
        loss=loss,
    )

    result = trainer.train()

    assert loss.batch_sizes[0] == 4
    assert loss.batch_sizes[-1] <= 2
    assert loss.batch_sizes == sorted(loss.batch_sizes, reverse=True)
    assert trainer._train_batch_size == trainer.state.train_batch_size == loss.batch_sizes[-1]
    assert trainer.args.train_batch_size == 4
    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert not torch.equal(original, next(model.parameters()).detach())


@pytest.mark.parametrize("dataset_kind", ["map", "dict", "iterable"])
def test_manual_train_dataloader_uses_initialized_batch_size(model, tmp_path, dataset_kind):
    trainer = SentenceTransformerTrainer(
        model=model,
        args=_args(tmp_path),
        train_dataset=_dataset(dataset_kind),
        loss=MSELoss(model),
    )

    assert trainer._train_batch_size == trainer.args.train_batch_size == 4
    assert len(next(iter(trainer.get_train_dataloader()))["label"]) == 4


@pytest.mark.parametrize(
    "auto_find_batch_size,error",
    [(False, "CUDA out of memory."), (True, "Unrelated forward failure")],
)
def test_batch_size_retries_only_when_enabled_for_memory_errors(model, tmp_path, auto_find_batch_size, error):
    loss = MemoryLimitedMSELoss(model, error=error)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=_args(tmp_path, auto_find_batch_size=auto_find_batch_size),
        train_dataset=_dataset("map"),
        loss=loss,
    )

    with pytest.raises(RuntimeError, match=error):
        trainer.train()

    assert loss.batch_sizes == [4]


def test_resumed_train_batch_size_does_not_change_evaluation_batches(model, tmp_path):
    checkpoint = tmp_path / "checkpoint-0"
    model.save_pretrained(str(checkpoint))
    # A local model/state checkpoint exercises the saved batch-size contract without optimizer or RNG files.
    TrainerState(train_batch_size=2).save_to_json(str(checkpoint / "trainer_state.json"))
    loss = MemoryLimitedMSELoss(model, max_batch_size=None)
    dataset = _dataset("map")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=_args(tmp_path, auto_find_batch_size=True),
        train_dataset=dataset,
        eval_dataset=dataset,
        loss=loss,
    )
    original = next(model.parameters()).detach().clone()

    result = trainer.train(resume_from_checkpoint=str(checkpoint))

    assert loss.batch_sizes == [2]
    assert trainer._train_batch_size == trainer.state.train_batch_size == 2
    assert trainer.args.train_batch_size == 4
    assert result.global_step == 1
    assert not torch.equal(original, next(model.parameters()).detach())
    assert len(next(iter(trainer.get_eval_dataloader()))["label"]) == 5
    assert len(next(iter(trainer.get_test_dataloader(dataset)))["label"]) == 5
