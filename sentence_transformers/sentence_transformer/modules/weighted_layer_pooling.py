from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
from torch import Tensor, nn

from sentence_transformers.base.modules.module import Module
from sentence_transformers.util.decorators import deprecated_kwargs


class WeightedLayerPooling(Module):
    """Token embeddings are weighted mean of their different hidden layer representations"""

    config_keys: list[str] = ["embedding_dimension", "layer_start", "num_hidden_layers"]
    config_key_renames = {"word_embedding_dimension": "embedding_dimension"}

    @deprecated_kwargs(**config_key_renames)
    def __init__(self, embedding_dimension, num_hidden_layers: int = 12, layer_start: int = 4, layer_weights=None):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float))
        )

    def forward(self, features: dict[str, Tensor]):
        ft_all_layers = features.get("all_layer_embeddings")
        if ft_all_layers is None:
            return features

        selected_layers = ft_all_layers[self.layer_start :]
        if (
            isinstance(ft_all_layers, (list, tuple))
            and selected_layers
            and selected_layers[0].ndim == 3
            and all(
                layer.shape == selected_layers[0].shape
                and layer.dtype == selected_layers[0].dtype
                and layer.device == selected_layers[0].device
                and layer.layout == torch.strided
                for layer in ft_all_layers
            )
        ):
            all_layer_embedding = torch.stack(selected_layers)
            # Preserve the zero-gradient connection that stacking all layers provided.
            # Empty views do not read discarded NaN/Inf values or copy their storage.
            discarded_zero = sum(
                layer[..., :0].sum() for layer in ft_all_layers[: len(ft_all_layers) - len(selected_layers)]
            )
        else:
            # Mixed layer dtypes and unusual shapes retain the original stack semantics.
            all_layer_embedding = torch.stack(ft_all_layers)[self.layer_start :, :, :, :]
            discarded_zero = 0

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_sum = (weight_factor * all_layer_embedding).sum(dim=0)
        weighted_average = (weighted_sum + discarded_zero) / self.layer_weights.sum()

        features.update({"token_embeddings": weighted_average})
        return features

    def get_embedding_dimension(self):
        return self.embedding_dimension

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        model = cls(**config)
        model = cls.load_torch_weights(model_name_or_path=model_name_or_path, model=model, **hub_kwargs)
        return model
