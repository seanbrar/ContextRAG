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
