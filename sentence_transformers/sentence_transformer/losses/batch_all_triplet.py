from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers.sentence_transformer.model import SentenceTransformer

from .batch_hard_triplet import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction


class _BatchAllTripletReduction(torch.autograd.Function):
    @staticmethod
    def forward(distances, labels, anchor_indices, positive_indices, margin):
        # Each active hinge contributes +1 to its anchor-positive distance and -1 to
        # its anchor-negative distance. Accumulate those coefficients rather than
        # retaining an autograd graph for every triplet in a large label group.
        reduction_dtype = torch.promote_types(torch.result_type(distances, margin), torch.float32)
        coefficients = torch.zeros_like(distances, dtype=reduction_dtype)
        numerator = distances.new_zeros((), dtype=reduction_dtype)
        num_positive = distances.new_zeros((), dtype=torch.long)
        batch_size = len(distances)
        for start in range(0, len(anchor_indices), max(1, batch_size)):
            anchors = anchor_indices[start : start + batch_size]
            positives = positive_indices[start : start + batch_size]
            # Preserve the subtraction order: (positive + margin) - negative can
            # cross the zero-hinge boundary after floating-point rounding.
            losses = distances[anchors, positives, None] - distances[anchors] + margin
            valid = labels[anchors, None] != labels[None, :]
            losses = valid.float() * losses
            negative = losses < 0
            # The original strict '< 0' mask retains the derivative at a zero hinge.
            active = valid & ~negative
            losses[negative] = 0
            numerator += losses.sum()
            num_positive += (losses > 1e-16).sum()

            coefficients[anchors, positives] = active.sum(dim=1).to(reduction_dtype)
            coefficients.index_add_(0, anchors, -active.to(reduction_dtype))

        denominator = num_positive.to(reduction_dtype) + 1e-16
        return numerator / denominator, coefficients, denominator

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, coefficients, denominator = output
        ctx.distance_dtype = inputs[0].dtype
        ctx.margin_shape = inputs[4].shape if isinstance(inputs[4], Tensor) else None
        ctx.margin_dtype = inputs[4].dtype if isinstance(inputs[4], Tensor) else None
        ctx.save_for_backward(coefficients, denominator)
        ctx.save_for_forward(coefficients, denominator)
        ctx.mark_non_differentiable(coefficients, denominator)

    @staticmethod
    def backward(ctx, grad_output, _grad_coefficients, _grad_denominator):
        coefficients, denominator = ctx.saved_tensors
        # Match the cast from the float32 mask multiplication back to a custom
        # low-precision distance metric before accumulating its contributions.
        scale = (grad_output / denominator).to(ctx.distance_dtype)
        gradient = coefficients * scale.to(coefficients.dtype)
        # Inactive entries remain zero when the reciprocal overflows a custom fp16
        # distance dtype (no positive triplets); multiplying zero by infinity does not.
        gradient.masked_fill_(coefficients == 0, 0)
        margin_gradient = None
        if ctx.needs_input_grad[4]:
            margin_gradient = (
                (-coefficients.clamp(max=0).sum(dim=0) * (grad_output / denominator))
                .sum_to_size(ctx.margin_shape)
                .to(ctx.margin_dtype)
            )
        return gradient.to(ctx.distance_dtype), None, None, None, margin_gradient

    @staticmethod
    def jvp(ctx, distance_tangent, _labels_tangent, _anchors_tangent, _positives_tangent, margin_tangent):
        coefficients, denominator = ctx.saved_tensors
        tangent = (coefficients * distance_tangent).sum()
        if margin_tangent is not None:
            tangent -= (coefficients.clamp(max=0).sum(dim=0) * margin_tangent).sum()
        return tangent / denominator, None, None


class BatchAllTripletLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric=BatchHardTripletLossDistanceFunction.euclidean_distance,
        margin: float = 5,
    ) -> None:
        """
        BatchAllTripletLoss takes a batch with (input, label) pairs and computes the loss for all possible, valid
        triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. The labels
        must be integers, with same label indicating inputs from the same class. Your train dataset
        must contain at least 2 examples per label class.

        Triplet reduction uses quadratic intermediate memory, including when many examples share
        a label. It still computes every valid triplet; the number of comparisons can be cubic in
        the batch size.

        Args:
            model: SentenceTransformer model
            distance_metric: Function that returns a distance between
                two embeddings. The class SiameseDistanceMetric contains
                pre-defined metrics that can be used.
            margin: Negative samples should be at least margin further
                apart from the anchor than the positive.

        References:
            * Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
            * Paper: In Defense of the Triplet Loss for Person Re-Identification, https://huggingface.co/papers/1703.07737
            * Blog post: https://omoindrot.github.io/triplet-loss

        Requirements:
            1. Each input must be labeled with a class.
            2. Your dataset must contain at least 2 examples per labels class.

        Inputs:
            +------------------+--------+
            | Inputs           | Labels |
            +==================+========+
            | single inputs    | class  |
            +------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.GROUP_BY_LABEL`` (:class:`docs <sentence_transformers.base.sampler.BatchSamplers>`)
              to ensure that each batch contains at least 2 distinct labels with at least 2 samples per label.

        Relations:
            * :class:`BatchHardTripletLoss` uses only the hardest positive and negative samples, rather than all possible, valid triplets.
            * :class:`BatchHardSoftMarginTripletLoss` uses only the hardest positive and negative samples, rather than all possible, valid triplets.
              Also, it does not require setting a margin.
            * :class:`BatchSemiHardTripletLoss` uses only semi-hard triplets, valid triplets, rather than all possible, valid triplets.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                # E.g. 0: sports, 1: economy, 2: politics
                train_dataset = Dataset.from_dict({
                    "sentence": [
                        "He played a great game.",
                        "The stock is up 20%",
                        "They won 2-1.",
                        "The last goal was amazing.",
                        "They all voted against the bill.",
                    ],
                    "label": [0, 1, 0, 0, 2],
                })
                loss = losses.BatchAllTripletLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

        """
        super().__init__()
        self.sentence_embedder = model
        self.triplet_margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        rep = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        return self.compute_loss_from_embeddings([rep], labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        return self.batch_all_triplet_loss(labels, embeddings[0])

    def batch_all_triplet_loss(self, labels: Tensor, embeddings: Tensor) -> Tensor:
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        batch_size = embeddings.shape[0]
        if labels.numel() != batch_size:
            raise ValueError(
                f"{self.__class__.__name__} expects {batch_size} labels (one per example), "
                f"but got {labels.numel()} labels with shape {tuple(labels.shape)}."
            )
        labels = labels.view(-1)

        pairwise_dist = self.distance_metric(embeddings)

        # Only valid anchor-positive pairs need to be compared with the negatives.
        anchor_indices, positive_indices = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).nonzero(
            as_tuple=True
        )
        return _BatchAllTripletReduction.apply(
            pairwise_dist, labels, anchor_indices, positive_indices, self.triplet_margin
        )[0]

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "distance_metric": getattr(self.distance_metric, "__name__", str(self.distance_metric)),
            "margin": self.triplet_margin,
        }

    @property
    def citation(self) -> str:
        return """
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
"""
