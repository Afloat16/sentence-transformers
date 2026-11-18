from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sentence_transformer.losses import EmbedDistillLoss
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.sparse_encoder.losses import CachedSpladeLoss, CSRLoss, SparseMSELoss, SpladeLoss
from sentence_transformers.sparse_encoder.modules import SparseAutoEncoder, SparseStaticEmbedding
from sentence_transformers.util import is_training_available

TEXTS = ["alpha", "beta", "gamma"]


def _model(csr: bool = False) -> SparseEncoder:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "gamma": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    if csr:
        backbone = StaticEmbedding(
            tokenizer,
            embedding_weights=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.1, 0.2], [0.2, 1.0, 0.3], [0.1, 0.2, 1.0]]
            ),
        )
        autoencoder = SparseAutoEncoder(input_dim=3, hidden_dim=12, k=2, k_aux=0)
        with torch.no_grad():
            autoencoder.encoder.weight.copy_(torch.rand(12, 3, generator=torch.Generator().manual_seed(7)))
        modules = [backbone, autoencoder]
    else:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", model_max_length=16
        )
        modules = [SparseStaticEmbedding(tokenizer, weight=torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))]
    model = SparseEncoder(modules=modules, device="cpu")
    model.model_card_data.generate_widget_examples = False
    return model


def _wrapped_loss(model, wrapper):
    base = SparseMSELoss(model)
    if wrapper is CSRLoss:
        return wrapper(model, loss=base)
    kwargs = {"mini_batch_size": 2} if wrapper is CachedSpladeLoss else {}
    return wrapper(model, loss=base, document_regularizer_weight=0.0, use_document_regularizer_only=True, **kwargs)


def _gradients(model):
    return {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}


@pytest.mark.parametrize("wrapper", [SpladeLoss, CachedSpladeLoss, CSRLoss])
@pytest.mark.parametrize("per_column", [False, True])
@pytest.mark.parametrize("teacher_matches_student", [False, True])
def test_wrapped_sparse_mse_keeps_each_rows_teacher_target(wrapper, per_column, teacher_matches_student):
    model = _model(csr=wrapper is CSRLoss)
    columns = [TEXTS, list(reversed(TEXTS)) if per_column else [f"{text} {text}" for text in TEXTS]]
    with torch.no_grad():
        targets = [model(model.preprocess(column))["sentence_embedding"].clone() for column in columns]
        if not teacher_matches_student:
            targets = [target * 0.5 for target in targets]
    labels = (torch.stack(targets, dim=1) if per_column else targets[0].clone()).requires_grad_()

    # Direct elementwise MSE over the complete batch is the oracle, independent of the loss's target splitting.
    predictions = [model(model.preprocess(column))["sentence_embedding"] for column in columns]
    expected = sum((prediction - target).square().mean() for prediction, target in zip(predictions, targets)) / 2
    expected.backward()
    expected_gradients = _gradients(model)
    model.zero_grad()

    wrapped = _wrapped_loss(model, wrapper)
    components = wrapped([model.preprocess(column) for column in columns], labels)
    actual = components["base_loss"]
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(_gradients(model), expected_gradients, atol=1e-6, rtol=1e-5)
    assert labels.grad is None, "teacher targets must remain detached, as in SparseMSELoss.forward"
    if teacher_matches_student:
        assert actual.item() == 0
        assert all(torch.count_nonzero(gradient) == 0 for gradient in _gradients(model).values())
    else:
        assert actual.item() > 0
        assert any(torch.count_nonzero(gradient) > 0 for gradient in _gradients(model).values())


@pytest.mark.parametrize("wrapper", [SpladeLoss, CachedSpladeLoss])
@pytest.mark.parametrize("per_column", [False, True])
def test_wrapped_sparse_mse_evaluation_keeps_full_batch_targets(wrapper, per_column):
    model = _model()
    columns = [TEXTS, list(reversed(TEXTS)) if per_column else TEXTS]
    with torch.no_grad():
        targets = [model(model.preprocess(column))["sentence_embedding"].clone() for column in columns]
        labels = torch.stack(targets, dim=1) if per_column else targets[0]
        output = _wrapped_loss(model, wrapper)([model.preprocess(column) for column in columns], labels)
    assert output["base_loss"].item() == 0


@pytest.mark.parametrize("metric", ["mse", "l2", "cosine"])
def test_embedding_distillation_keeps_explicit_teacher_column_list(metric):
    model = _model()
    loss = EmbedDistillLoss(model, distance_metric=metric)
    students = [torch.tensor([[1.0, 0.0], [0.0, 2.0]]), torch.tensor([[2.0, 1.0], [1.0, 3.0]])]
    teachers = [torch.tensor([[0.8, 0.2], [0.1, 1.8]]), torch.tensor([[1.5, 0.7], [0.9, 2.7]])]
    from_list = loss.compute_loss_from_embeddings(students, teachers)
    from_tensor = loss.compute_loss_from_embeddings(students, torch.stack(teachers, dim=1))
    torch.testing.assert_close(from_tensor, from_list)


@pytest.mark.parametrize("per_column", [False, True])
def test_embedding_distillation_forward_preserves_subclass_column_list(per_column):
    class ListTeacherLoss(EmbedDistillLoss):
        def compute_loss_from_embeddings(self, embeddings, teacher_embeddings):
            assert isinstance(teacher_embeddings, list)
            assert len(teacher_embeddings) == len(embeddings) == 2
            return super().compute_loss_from_embeddings(embeddings, teacher_embeddings)

    model = _model()
    columns = [TEXTS, list(reversed(TEXTS)) if per_column else TEXTS]
    with torch.no_grad():
        targets = [model(model.preprocess(column))["sentence_embedding"].clone() for column in columns]
    labels = torch.stack(targets, dim=1) if per_column else targets[0]
    loss = ListTeacherLoss(model)([model.preprocess(column) for column in columns], labels)
    assert loss.item() == 0


@pytest.mark.parametrize("wrapper", [SpladeLoss, CachedSpladeLoss])
@pytest.mark.parametrize("teacher_matches_student", [False, True])
@pytest.mark.skipif(not is_training_available(), reason="The [train] extra is required")
def test_sparse_mse_trainer_update_matches_rowwise_mse(tmp_path, wrapper, teacher_matches_student):
    from datasets import Dataset

    model = _model()
    initial = model[0].weight.detach().clone()
    teacher_factor = 1.0 if teacher_matches_student else 0.5
    labels = torch.diag(initial)[2:] * teacher_factor
    dataset = Dataset.from_dict({"text": TEXTS, "label": labels.tolist()})
    learning_rate = 0.2
    trainer = SparseEncoderTrainer(
        model=model,
        loss=_wrapped_loss(model, wrapper),
        train_dataset=dataset,
        optimizers=(torch.optim.SGD(model.parameters(), lr=learning_rate), None),
        args=SparseEncoderTrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            max_steps=1,
            per_device_train_batch_size=3,
            learning_rate=learning_rate,
            lr_scheduler_type="constant",
            max_grad_norm=0.0,
            weight_decay=0.0,
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            dataloader_pin_memory=False,
        ),
    )
    result = trainer.train()

    # Each active weight occurs once in a 3 x 5 batch: dMSE/dw = 2 * (w - teacher) / 15.
    expected = initial - learning_rate * 2 * (1 - teacher_factor) * initial / 15
    torch.testing.assert_close(model[0].weight, expected, atol=1e-7, rtol=1e-7)
    assert result.global_step == 1
    assert result.training_loss == pytest.approx(float(((1 - teacher_factor) * initial).square().sum() / 15))
    assert torch.equal(model[0].weight, initial) == teacher_matches_student
