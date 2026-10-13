"""Typed experiment configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    @property
    def model_name(self) -> str:
        return str(self.raw["model"]["name_or_path"])

    @property
    def training_method(self) -> str:
        method = str(self.raw["training"]["method"]).lower()
        if method not in {"baseline", "lora", "qlora"}:
            raise ValueError(f"Unsupported training method: {method}")
        return method

    @property
    def lora_rank(self) -> int:
        rank = int(self.raw["lora"]["rank"])
        if rank not in {8, 16, 32}:
            raise ValueError("LoRA rank must be one of the controlled ablation values: 8, 16, 32")
        return rank


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment file and validate fields used by the pipelines."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    for section in ("project", "model", "data", "training", "lora", "inference"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"Missing or invalid configuration section: {section}")
    for section, key in (
        ("model", "revision"),
        ("model", "tokenizer_revision"),
        ("data", "dataset_revision"),
    ):
        revision = str(raw[section].get(key, ""))
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError(f"{section}.{key} must be a pinned 40-character commit SHA")
    config = ExperimentConfig(raw=raw)
    _ = config.seed, config.model_name, config.training_method, config.lora_rank
    return config
