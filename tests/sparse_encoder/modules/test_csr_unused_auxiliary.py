from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from sentence_transformers.sparse_encoder.losses import CSRReconstructionLoss
from sentence_transformers.sparse_encoder.modules import SparseAutoEncoder


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("k_aux", [0, 4])
@pytest.mark.parametrize("mode", ["train", "eval", "no_grad"])
def test_skipping_discarded_auxiliary_latents_preserves_training(normalize, k_aux, mode):
    torch.manual_seed(13)
    actual = SparseAutoEncoder(input_dim=4, hidden_dim=16, k=2, k_aux=k_aux, normalize=normalize)
    actual.stats_last_nonzero.copy_(torch.arange(16) + 30)
    reference = deepcopy(actual)
    original_top_k = reference.top_k

    def compute_every_auxiliary(*args, **kwargs):
        # Reproduce the old path, which computed and discarded the 4k pass's auxiliary values.
        kwargs["compute_aux"] = True
        return original_top_k(*args, **kwargs)

    reference.top_k = compute_every_auxiliary
    if mode == "eval":
        actual.eval()
        reference.eval()
    loss_fn = CSRReconstructionLoss(model=None)
    for _ in range(3):
        inputs = torch.randn(3, 4)
        with torch.set_grad_enabled(mode != "no_grad"):
            actual_output = actual({"sentence_embedding": inputs.clone()})
            reference_output = reference({"sentence_embedding": inputs.clone()})
            for name in actual_output:
                torch.testing.assert_close(actual_output[name], reference_output[name], rtol=0, atol=0)
            actual_losses = loss_fn.compute_loss_from_embeddings([actual_output])
            reference_losses = loss_fn.compute_loss_from_embeddings([reference_output])
            for name in actual_losses:
                torch.testing.assert_close(actual_losses[name], reference_losses[name], rtol=0, atol=0)
            if mode != "no_grad":
                sum(actual_losses.values()).backward()
                sum(reference_losses.values()).backward()
                for actual_parameter, reference_parameter in zip(actual.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad, rtol=0, atol=0)
                actual.zero_grad()
                reference.zero_grad()
        torch.testing.assert_close(actual.stats_last_nonzero, reference.stats_last_nonzero, rtol=0, atol=0)
