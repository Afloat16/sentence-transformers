from __future__ import annotations

import pytest
import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from sentence_transformers.base.sampler import BatchSamplers, MultiDatasetBatchSamplers
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments


@pytest.fixture
def local_model_path(tmp_path):
    path = tmp_path / "model"
    vocab = {"[PAD]": 0, "[UNK]": 1, "query": 2, "document": 3}
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
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    ).save_pretrained(path)
    return path


def make_trainer(path, seed, data_seed, sampler, multi=None):
    def dataset(prefix):
        return Dataset.from_dict(
            {
                "query": [f"query {prefix}{i}" for i in range(16)],
                "document": [f"document {prefix}{i}" for i in range(16)],
                "label": [i % 2 for i in range(16)],
            }
        )

    data = dataset("a") if multi is None else DatasetDict(a=dataset("a"), b=dataset("b"))
    return CrossEncoderTrainer(
        model=CrossEncoder(str(path)),
        train_dataset=data,
        args=CrossEncoderTrainingArguments(
            output_dir=str(path.parent / "output"),
            use_cpu=True,
            report_to=[],
            save_strategy="no",
            max_steps=1,
            per_device_train_batch_size=4,
            seed=seed,
            data_seed=data_seed,
            batch_sampler=sampler,
            multi_dataset_batch_sampler=multi or MultiDatasetBatchSamplers.PROPORTIONAL,
        ),
    )


def order(path, seed, data_seed, sampler, multi=None):
    loader = make_trainer(path, seed, data_seed, sampler, multi).get_train_dataloader()
    return [query for batch in loader for query in batch["query_input"]]


@pytest.mark.parametrize(
    "sampler", [BatchSamplers.BATCH_SAMPLER, BatchSamplers.NO_DUPLICATES, BatchSamplers.GROUP_BY_LABEL]
)
@pytest.mark.parametrize(
    "multi", [None, MultiDatasetBatchSamplers.PROPORTIONAL, MultiDatasetBatchSamplers.ROUND_ROBIN]
)
def test_data_seed_is_independent_of_model_seed(local_model_path, sampler, multi):
    first = order(local_model_path, 11, 37, sampler, multi)
    assert first == order(local_model_path, 29, 37, sampler, multi)
    assert first == order(local_model_path, 11, 37, sampler, multi)
    assert first != order(local_model_path, 11, 41, sampler, multi)
    expected = {f"query {prefix}{i}" for prefix in ("a", "b") if multi is not None or prefix == "a" for i in range(16)}
    assert len(first) == len(expected)
    assert set(first) == expected


@pytest.mark.parametrize(
    "sampler", [BatchSamplers.BATCH_SAMPLER, BatchSamplers.NO_DUPLICATES, BatchSamplers.GROUP_BY_LABEL]
)
def test_unset_data_seed_falls_back_and_zero_is_explicit(local_model_path, sampler):
    assert order(local_model_path, 11, None, sampler) == order(local_model_path, 11, 11, sampler)
    assert order(local_model_path, 11, 0, sampler) == order(local_model_path, 29, 0, sampler)


def test_seeded_group_sampler_runs_a_real_training_step(local_model_path):
    trainer = make_trainer(local_model_path, 11, 37, BatchSamplers.GROUP_BY_LABEL)
    weights = next(trainer.model.parameters())
    before = weights.detach().clone()
    result = trainer.train()
    assert result.global_step == 1
    assert not torch.equal(weights, before)
    assert torch.isfinite(weights).all()
