from __future__ import annotations

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
from sentence_transformers.sentence_transformer.losses import (
    ContrastiveTensionLoss,
    ContrastiveTensionLossInBatchNegatives,
)
from sentence_transformers.sentence_transformer.modules import Dropout, Pooling, StaticEmbedding, Transformer
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip('Sentence Transformers was not installed with the ["train"] extra.', allow_module_level=True)

from datasets import Dataset


def _static_model(dropout: Dropout) -> SentenceTransformer:
    vocabulary = {"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3, "delta": 4}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    weights = torch.randn(5, 16, generator=torch.Generator().manual_seed(12)) / 4
    model = SentenceTransformer(modules=[StaticEmbedding(tokenizer, embedding_weights=weights), dropout], device="cpu")
    model.model_card_data.generate_widget_examples = False
    return model


@pytest.mark.parametrize("loss_class", [ContrastiveTensionLoss, ContrastiveTensionLossInBatchNegatives])
@pytest.mark.parametrize("initial_training", [False, True])
def test_contrastive_tension_encoder_modes_follow_trainer(tmp_path, loss_class, initial_training: bool) -> None:
    model = _static_model(Dropout(0.5))
    model.train(initial_training)
    loss = loss_class(model)
    secondary_model = loss.model1

    columns = {"text": ["alpha", "beta", "gamma", "delta"]}
    if loss_class is ContrastiveTensionLoss:
        columns.update({"pair": ["alpha", "gamma", "beta", "delta"], "label": [1.0, 0.0, 0.0, 1.0]})
    dataset = Dataset.from_dict(columns)
    trainer = SentenceTransformerTrainer(
        model=model,
        loss=loss,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            max_steps=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=0.01,
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            dataloader_pin_memory=False,
        ),
    )
    observed_modes = []
    handle = secondary_model[1].dropout_layer.register_forward_pre_hook(
        lambda module, inputs: observed_modes.append(module.training)
    )
    first_eval = trainer.evaluate()["eval_loss"]
    second_eval = trainer.evaluate()["eval_loss"]
    eval_modes = observed_modes[:]
    observed_modes.clear()
    primary_before = model[0].embedding.weight.detach().clone()
    secondary_before = secondary_model[0].embedding.weight.detach().clone()

    result = trainer.train()
    handle.remove()

    assert eval_modes == [False, False]
    assert first_eval == second_eval
    assert observed_modes == [True]
    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert not torch.equal(model[0].embedding.weight, primary_before)
    assert not torch.equal(secondary_model[0].embedding.weight, secondary_before)
    assert loss.model1 is secondary_model
    assert secondary_model[0].embedding.weight.data_ptr() != model[0].embedding.weight.data_ptr()


@pytest.mark.parametrize("loss_class", [ContrastiveTensionLoss, ContrastiveTensionLossInBatchNegatives])
def test_contrastive_tension_preserves_selective_module_modes(loss_class) -> None:
    class TrackingDropout(Dropout):
        def __init__(self):
            super().__init__(0.5)
            self.mode_updates = []

        def train(self, mode: bool = True):
            self.mode_updates.append(mode)
            return super().train(mode)

    model = _static_model(TrackingDropout())
    model[1].eval()
    loss = loss_class(model)
    copied_dropout = loss.model1[1]

    def forward():
        features = [model.preprocess(["alpha", "beta"])]
        if loss_class is ContrastiveTensionLoss:
            features.append(model.preprocess(["alpha", "gamma"]))
        return loss(features, torch.tensor([1.0, 0.0]))

    forward()
    assert model.training and loss.model1.training
    assert not model[1].training and not copied_dropout.training
    assert copied_dropout.mode_updates == [False]

    model.train()
    forward()
    assert copied_dropout.training
    assert copied_dropout.mode_updates == [False, True]

    model[1].eval()
    forward()
    assert not copied_dropout.training
    assert copied_dropout.mode_updates == [False, True, False]


@pytest.mark.parametrize("loss_class", [ContrastiveTensionLoss, ContrastiveTensionLossInBatchNegatives])
def test_contrastive_tension_tracks_loaded_transformer_modes(tmp_path, loss_class) -> None:
    model_path = tmp_path / "encoder"
    config = BertConfig(vocab_size=5, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    BertModel(config).save_pretrained(model_path)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "gamma": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(
        model_path
    )
    model = SentenceTransformer(modules=[Transformer(str(model_path)), Pooling(16)], device="cpu")
    loss = loss_class(model)
    # from_pretrained() loads the backbone in eval mode, inside a training-mode SentenceTransformer.
    assert model.training and loss.model1.training
    assert not loss.model1[0].auto_model.training

    model.train()
    features = [model.preprocess(["alpha", "beta"])]
    if loss_class is ContrastiveTensionLoss:
        features.append(model.preprocess(["alpha", "gamma"]))
    value = loss(features, torch.tensor([1.0, 0.0]))
    value.backward()

    assert loss.model1[0].auto_model.training
    assert loss.model1[0].auto_model.embeddings.dropout.training
    assert torch.isfinite(value)
    assert loss.model1[0].auto_model.embeddings.word_embeddings.weight.grad is not None
