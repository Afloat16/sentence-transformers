from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Trainer

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MSELoss
from sentence_transformers.sentence_transformer.modules import Dense, StaticEmbedding
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )

from datasets import Dataset


def tiny_model() -> SentenceTransformer:
    torch.manual_seed(42)
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3, "delta": 4, "epsilon": 5},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    model = SentenceTransformer(
        modules=[StaticEmbedding(tokenizer, embedding_dim=8), Dense(8, 4, activation_function=torch.nn.Identity())],
        device="cpu",
    )
    model.model_card_data.generate_widget_examples = False
    return model


def group_for(optimizer, parameter):
    return next(group for group in optimizer.param_groups if any(p is parameter for p in group["params"]))


@pytest.mark.parametrize("group_key", ["params", "optimizer_dict"])
def test_learning_rate_mapping_preserves_group_options_and_sgd_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, group_key: str
) -> None:
    model = tiny_model()
    embedding, weight, bias = model[0].embedding.weight, model[1].linear.weight, model[1].linear.bias
    args = SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path),
        learning_rate=0.01,
        weight_decay=0.5,
        learning_rate_mapping={r"model\.1\.linear": 0.02, r"model\.1\.linear\.weight": 0.04},
        report_to=[],
        use_cpu=True,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, loss=MSELoss(model))
    # Optimizers may supply groups with settings beyond lr and weight_decay. Exercise that superclass
    # contract with real SGD, whose first update has an independent closed-form expectation.
    groups = [
        {"params": iter([embedding]), "weight_decay": 0.25, "momentum": 0.3},
        {"params": [bias], "weight_decay": 0.25, "momentum": 0.3},
        {"params": [weight], "weight_decay": 0.125, "momentum": 0.8, "nesterov": True},
    ]
    monkeypatch.setattr(
        Trainer,
        "get_optimizer_cls_and_kwargs",
        staticmethod(lambda *args, **kwargs: (torch.optim.SGD, {"lr": 0.01, group_key: groups})),
    )
    optimizer = trainer.create_optimizer()
    expected_options = [
        (embedding, 0.01, 0.25, 0.3, False),
        (weight, 0.04, 0.125, 0.8, True),
        (bias, 0.02, 0.25, 0.3, False),
    ]
    expected_updates = []
    for parameter, lr, decay, momentum, nesterov in expected_options:
        before = parameter.detach().clone()
        parameter.grad = torch.full_like(parameter, 1.25)
        direction = parameter.grad + decay * before
        expected_updates.append(before - lr * direction * (1 + momentum if nesterov else 1))
    optimizer.step()
    for (parameter, lr, decay, momentum, nesterov), expected in zip(expected_options, expected_updates):
        torch.testing.assert_close(parameter, expected)
        group = group_for(optimizer, parameter)
        assert group["lr"] == lr
        assert group["weight_decay"] == decay
        assert group["momentum"] == momentum
        assert group["nesterov"] == nesterov
    assert len({id(p) for group in optimizer.param_groups for p in group["params"]}) == 3
    assert sum(len(group["params"]) for group in optimizer.param_groups) == 3


def test_learning_rate_mapping_keeps_default_decay_groups(tmp_path: Path) -> None:
    model = tiny_model()
    args = SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path),
        optim="adamw_torch",
        learning_rate=0.01,
        weight_decay=0.1,
        learning_rate_mapping={r"model\.1\.linear": 0.02},
        report_to=[],
        use_cpu=True,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, loss=MSELoss(model))
    optimizer = trainer.create_optimizer()
    for parameter, lr, decay in [
        (model[0].embedding.weight, 0.01, 0.1),
        (model[1].linear.weight, 0.02, 0.1),
        (model[1].linear.bias, 0.02, 0.0),
    ]:
        group = group_for(optimizer, parameter)
        assert group["lr"] == lr
        assert group["weight_decay"] == decay


@pytest.mark.parametrize("mapped_lr", [0.01, 0.02])
def test_galore_training_retains_low_rank_state_and_update(tmp_path: Path, mapped_lr: float) -> None:
    pytest.importorskip("galore_torch")
    original = tiny_model()
    dataset = Dataset.from_dict(
        {
            "sentence": ["alpha beta", "beta gamma", "delta alpha", "epsilon"],
            "label": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        }
    )
    trained_models = []
    for mapped in [False, True]:
        model = deepcopy(original)
        args = SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path / str(mapped)),
            max_steps=2,
            per_device_train_batch_size=2,
            learning_rate=0.01,
            lr_scheduler_type="constant",
            optim="galore_adamw",
            optim_target_modules=["linear"],
            optim_args="rank=2,update_proj_gap=10,scale=0.25",
            learning_rate_mapping={r"model\.1\.linear\.weight": mapped_lr} if mapped else {},
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        )
        trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=MSELoss(model))
        optimizer = trainer.create_optimizer()
        parameter = model[1].linear.weight
        if not mapped:
            # Independent control: set the same lr on GaLore's existing group without rebuilding it.
            group_for(optimizer, parameter)["lr"] = mapped_lr
        trainer.train()
        group = group_for(trainer.optimizer, parameter)
        assert group["lr"] == mapped_lr
        assert group.get("rank") == 2
        assert group.get("update_proj_gap") == 10
        assert group.get("scale") == 0.25
        assert group.get("proj_type") == "std"
        state = trainer.optimizer.state[parameter]
        assert "projector" in state
        assert state["exp_avg"].numel() == 16 < parameter.numel()
        assert not torch.equal(parameter, original[1].linear.weight)
        trained_models.append(model)
    for control, mapped in zip(trained_models[0].parameters(), trained_models[1].parameters()):
        torch.testing.assert_close(mapped, control, rtol=0, atol=0)
