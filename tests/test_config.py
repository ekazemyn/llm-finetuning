from pathlib import Path

import pytest

from llm_finetuning.config import ExperimentConfig, load_config


def test_load_project_experiment_config():
    config = load_config(Path("configs/experiment.yaml"))
    assert config.seed == 42
    assert config.training_method == "lora"
    assert config.lora_rank == 16


def test_reject_unsupported_method_and_rank():
    raw = {"training": {"method": "full"}, "lora": {"rank": 16}}
    config = ExperimentConfig(raw=raw)
    with pytest.raises(ValueError, match="Unsupported training method"):
        _ = config.training_method
    raw["training"]["method"] = "lora"
    raw["lora"]["rank"] = 4
    with pytest.raises(ValueError, match="rank must be one of"):
        _ = config.lora_rank


def test_project_revisions_are_immutable_commit_shas(tmp_path):
    config_text = Path("configs/experiment.yaml").read_text(encoding="utf-8")
    invalid_config = tmp_path / "experiment.yaml"
    invalid_config.write_text(config_text.replace("revision: 989aa798", "revision: main", 1))

    with pytest.raises(ValueError, match="pinned 40-character commit SHA"):
        load_config(invalid_config)
