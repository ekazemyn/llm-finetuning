from pathlib import Path

import pytest

from llm_finetuning.config import ExperimentConfig, load_config


def test_load_project_experiment_config():
    config = load_config(Path("configs/experiment.yaml"))
    assert config.seed == 42
    assert config.training_method == "lora"
    assert config.lora_rank == 16
    assert config.raw["project"]["mlflow_tracking_uri"] == "sqlite:///mlflow.db"


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

    with pytest.raises(ValueError, match="model.revision"):
        load_config(invalid_config)


def _write_mutated_config(tmp_path, mutate):
    import yaml

    raw = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_reject_unknown_keys_at_root_and_within_sections(tmp_path):
    unknown_root = _write_mutated_config(tmp_path, lambda raw: raw.update(extra=True))
    with pytest.raises(ValueError, match="extra"):
        load_config(unknown_root)

    unknown_section = _write_mutated_config(
        tmp_path, lambda raw: raw["training"].update(unused_option=1)
    )
    with pytest.raises(ValueError, match="unused_option"):
        load_config(unknown_section)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("project", "seed", "42"),
        ("model", "max_length", 0),
        ("data", "max_train_samples", 0),
        ("training", "learning_rate", 0.0),
        ("training", "warmup_ratio", 1.0),
        ("training", "max_steps", 0),
        ("lora", "dropout", 1.0),
        ("inference", "batch_sizes", [1, 0]),
    ],
)
def test_reject_invalid_types_and_ranges(tmp_path, section, key, value):
    config_path = _write_mutated_config(tmp_path, lambda raw: raw[section].update({key: value}))

    with pytest.raises(ValueError, match="Invalid experiment config"):
        load_config(config_path)


def test_reject_mismatched_save_and_evaluation_strategies(tmp_path):
    config_path = _write_mutated_config(
        tmp_path, lambda raw: raw["training"].update(evaluation_strategy="steps")
    )

    with pytest.raises(ValueError, match="save_strategy must match evaluation_strategy"):
        load_config(config_path)
