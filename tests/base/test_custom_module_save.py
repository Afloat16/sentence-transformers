from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import BoW


def _load_custom_projection(source: Path, factor: int):
    source.mkdir()
    (source / "projection_utils.py").write_text(
        f"def scale_embedding(features):\n"
        f"    features['sentence_embedding'] = features['sentence_embedding'] * {factor}\n"
        f"    return features\n",
        encoding="utf8",
    )
    (source / "modeling_projection.py").write_text(
        "from sentence_transformers.base.modules import Dense\n"
        "from .projection_utils import scale_embedding\n\n"
        "class CustomProjection(Dense):\n"
        "    def forward(self, features):\n"
        "        return scale_embedding(super().forward(features))\n",
        encoding="utf8",
    )
    return get_class_from_dynamic_module("modeling_projection.CustomProjection", str(source))


@pytest.mark.parametrize("num_projections", [1, 2])
def test_dynamic_projection_save_roundtrip(tmp_path: Path, num_projections: int) -> None:
    modules = [BoW(["red", "blue", "green"])]
    for index in range(num_projections):
        # The two source packages deliberately have the same filenames, but different helper code.
        projection = _load_custom_projection(tmp_path / f"source_{tmp_path.name}_{index}", factor=index + 2)
        modules.append(projection(3, 3, init_weight=torch.eye(3), init_bias=torch.zeros(3)))
    model = SentenceTransformer(modules=modules, device="cpu")
    sentences = ["red red blue", "green blue", "red green"]
    expected = model.encode(sentences)

    for generation in range(2):
        checkpoint = tmp_path / f"checkpoint_{tmp_path.name}_{generation}"
        model.save(str(checkpoint), create_model_card=False)
        with pytest.raises(ValueError, match="trust_remote_code"):
            SentenceTransformer(str(checkpoint), local_files_only=True, device="cpu")
        model = SentenceTransformer(str(checkpoint), trust_remote_code=True, local_files_only=True, device="cpu")
        np.testing.assert_array_equal(model.encode(sentences), expected)
        assert all(type(module).__name__ == "CustomProjection" for module in list(model)[1:])


@pytest.mark.parametrize("save_in_root", [False, True])
def test_dynamic_input_module_save_roundtrip(tmp_path: Path, save_in_root: bool) -> None:
    source = tmp_path / f"input_source_{tmp_path.name}"
    source.mkdir()
    (source / "modeling_bow.py").write_text(
        "from sentence_transformers.sentence_transformer.modules import BoW\n\n"
        "class CustomBoW(BoW):\n"
        f"    save_in_root = {save_in_root}\n",
        encoding="utf8",
    )
    input_module = get_class_from_dynamic_module("modeling_bow.CustomBoW", str(source))
    model = SentenceTransformer(modules=[input_module(["red", "blue", "green"])], device="cpu")
    sentences = ["red blue", "green green"]
    expected = model.encode(sentences)

    for generation in range(2):
        checkpoint = tmp_path / f"input_checkpoint_{tmp_path.name}_{generation}"
        model.save(str(checkpoint), create_model_card=False)
        model = SentenceTransformer(str(checkpoint), trust_remote_code=True, local_files_only=True, device="cpu")
        np.testing.assert_array_equal(model.encode(sentences), expected)
        assert type(model[0]).__name__ == "CustomBoW"


def test_root_custom_code_with_subfolder_weights_still_loads(tmp_path: Path) -> None:
    projection = _load_custom_projection(tmp_path / f"legacy_source_{tmp_path.name}", factor=2)
    model = SentenceTransformer(modules=[BoW(["red", "blue"]), projection(2, 2)], device="cpu")
    sentences = ["red red blue", "blue"]
    expected = model.encode(sentences)
    checkpoint = tmp_path / f"legacy_checkpoint_{tmp_path.name}"
    model.save(str(checkpoint), create_model_card=False)

    # Existing Hub models may keep their custom code at the root while module weights live in a subfolder.
    config = json.loads((checkpoint / "modules.json").read_text(encoding="utf8"))
    config[1]["type"] = "modeling_projection.CustomProjection"
    (checkpoint / "modules.json").write_text(json.dumps(config), encoding="utf8")
    for code_file in (checkpoint / config[1]["path"]).glob("*.py"):
        shutil.move(str(code_file), checkpoint / code_file.name)

    model = SentenceTransformer(str(checkpoint), trust_remote_code=True, local_files_only=True, device="cpu")
    np.testing.assert_array_equal(model.encode(sentences), expected)
    resaved = tmp_path / f"resaved_checkpoint_{tmp_path.name}"
    model.save(str(resaved), create_model_card=False)
    model = SentenceTransformer(str(resaved), trust_remote_code=True, local_files_only=True, device="cpu")
    np.testing.assert_array_equal(model.encode(sentences), expected)
