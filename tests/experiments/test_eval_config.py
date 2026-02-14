
import pytest

from contextrag.experiments.eval_config import load_eval_config


def test_load_eval_config_defaults(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("dataset: data/sample\n", encoding="utf-8")
    config = load_eval_config(path)
    assert config.dataset == "data/sample"
    assert config.baseline == "uniform"
    assert config.k == 5


def test_load_eval_config_rejects_openai_provider(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("dataset: data/sample\nembed_provider: openai\n", encoding="utf-8")
    with pytest.raises(ValueError, match="embed_provider"):
        load_eval_config(path)


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


def test_load_eval_config_validates_types_and_values(tmp_path):
    # schema_version
    path = tmp_path / "config_schema.yml"
    path.write_text("dataset: d\nschema_version: bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an integer"):
        load_eval_config(path)
        
    path.write_text("dataset: d\nschema_version: 999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be 1"):
        load_eval_config(path)

    # dataset
    path = tmp_path / "config_dataset.yml"
    path.write_text("dataset: ''\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty string"):
        load_eval_config(path)

    # baseline
    path = tmp_path / "config_baseline.yml"
    path.write_text("dataset: d\nbaseline: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'baseline' must be a string"):
        load_eval_config(path)
        
    path.write_text("dataset: d\nbaseline: invalid_base\n", encoding="utf-8")
    with pytest.raises(ValueError, match="baseline.*must be one of"):
        load_eval_config(path)

    # k
    path = tmp_path / "config_k.yml"
    path.write_text("dataset: d\nk: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'k' must be an integer"):
        load_eval_config(path)
        
    path.write_text("dataset: d\nk: 'not_int'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'k' must be an integer"):
        load_eval_config(path)
        
    path.write_text("dataset: d\nk: -1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'k' must be an integer"):
        load_eval_config(path)

    # embed_provider
    path = tmp_path / "config_embed.yml"
    path.write_text("dataset: d\nembed_provider: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'embed_provider' must be a string"):
        load_eval_config(path)

    # embedding_model
    path = tmp_path / "config_model.yml"
    path.write_text("dataset: d\nembedding_model: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'embedding_model' must be a string"):
        load_eval_config(path)
        
    # misc paths
    path = tmp_path / "config_misc.yml"
    path.write_text("dataset: d\noutput: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'output' must be a string"):
        load_eval_config(path)
        
    path.write_text("dataset: d\npersist: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'persist' must be a string"):
        load_eval_config(path)
        
    path.write_text("dataset: d\nrun_dir: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'run_dir' must be a string"):
        load_eval_config(path)


def test_load_eval_config_not_mapping(tmp_path):
    path = tmp_path / "config_list.yml"
    path.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_eval_config(path)
