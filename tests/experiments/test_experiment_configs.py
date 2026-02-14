from pathlib import Path

from contextrag.experiments.eval_config import load_eval_config


EXPERIMENT_DIR = Path("experiments")


def test_all_experiment_configs_load_and_reference_existing_datasets():
    configs = sorted(EXPERIMENT_DIR.glob("*.yaml"))
    assert configs, "Expected at least one experiment YAML config."

    for config_path in configs:
        config = load_eval_config(config_path)
        dataset = Path(config.dataset)
        assert dataset.exists(), f"Missing dataset for {config_path}: {dataset}"
        assert (dataset / "documents").exists(), (
            f"Missing documents directory for {config_path}: {dataset / 'documents'}"
        )
        assert (dataset / "queries.jsonl").exists(), (
            f"Missing queries file for {config_path}: {dataset / 'queries.jsonl'}"
        )
