from pathlib import Path

import pytest

from contextrag.experiments.eval_config import load_eval_config


def test_load_eval_config_defaults(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("dataset: data/sample\n", encoding="utf-8")
    config = load_eval_config(path)
    assert config.dataset == "data/sample"
    assert config.baseline == "router"
    assert config.k == 5


def test_load_eval_config_requires_dataset(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("baseline: uniform\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset"):
        load_eval_config(path)


def test_load_eval_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("dataset: data/sample\nextra: value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_eval_config(path)


def test_load_eval_config_validates_schema_version(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("dataset: data/sample\nschema_version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_eval_config(path)


def test_load_eval_config_validates_embed_provider(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("dataset: data/sample\nembed_provider: nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="embed_provider"):
        load_eval_config(path)


def test_load_eval_config_validates_k(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("dataset: data/sample\nk: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="k"):
        load_eval_config(path)
