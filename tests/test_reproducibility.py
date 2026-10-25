from copy import deepcopy
from pathlib import Path

from llm_finetuning.config import load_config
from llm_finetuning.reproducibility import build_provenance


def test_provenance_contains_config_hash_git_revisions_and_runtime():
    config = load_config(Path("configs/experiment.yaml")).raw

    provenance = build_provenance(config)

    assert len(provenance["config_sha256"]) == 64
    assert provenance["resolved_config"] == config
    assert provenance["source_revisions"] == {
        "model": config["model"]["revision"],
        "tokenizer": config["model"]["tokenizer_revision"],
        "dataset": config["data"]["dataset_revision"],
    }
    assert "commit" in provenance["git"]
    assert "dirty" in provenance["git"]
    assert "python_version" in provenance["runtime"]
    assert "torch" in provenance["packages"]


def test_provenance_hash_changes_when_config_changes():
    config = load_config(Path("configs/experiment.yaml")).raw
    changed_config = deepcopy(config)
    changed_config["project"]["seed"] += 1

    assert (
        build_provenance(config)["config_sha256"]
        != build_provenance(changed_config)["config_sha256"]
    )
