"""Typed experiment configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _ProjectSettings(_StrictModel):
    name: StrictStr = Field(min_length=1)
    seed: StrictInt = Field(ge=0)
    output_dir: StrictStr = Field(min_length=1)
    mlflow_tracking_uri: StrictStr = Field(min_length=1)
    experiment_name: StrictStr = Field(min_length=1)


class _ModelSettings(_StrictModel):
    name_or_path: StrictStr = Field(min_length=1)
    revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    max_length: StrictInt = Field(gt=0)
    trust_remote_code: StrictBool = False


class _DataSettings(_StrictModel):
    dataset_name: StrictStr = Field(min_length=1)
    dataset_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    source_repository: StrictStr = Field(default="czyssrs/FinQA", min_length=1)
    train_split: StrictStr = Field(min_length=1)
    validation_split: StrictStr = Field(min_length=1)
    test_split: StrictStr = Field(min_length=1)
    max_train_samples: StrictInt | None = Field(default=None, gt=0)
    max_eval_samples: StrictInt | None = Field(default=None, gt=0)
    max_test_samples: StrictInt | None = Field(default=None, gt=0)
    prepared_dir: StrictStr = Field(default="data/finqa-prepared", min_length=1)


class _TrainingSettings(_StrictModel):
    method: Literal["baseline", "lora", "qlora"]
    output_dir: StrictStr = Field(min_length=1)
    learning_rate: StrictFloat = Field(gt=0)
    epochs: StrictFloat | StrictInt = Field(gt=0)
    per_device_train_batch_size: StrictInt = Field(gt=0)
    per_device_eval_batch_size: StrictInt = Field(gt=0)
    gradient_accumulation_steps: StrictInt = Field(gt=0)
    gradient_checkpointing: StrictBool
    warmup_ratio: StrictFloat = Field(ge=0, lt=1)
    weight_decay: StrictFloat = Field(ge=0)
    logging_steps: StrictInt = Field(gt=0)
    save_strategy: Literal["epoch", "steps", "no", "best"]
    evaluation_strategy: Literal["epoch", "steps", "no"]
    max_steps: StrictInt = -1
    bf16: StrictBool = False
    fp16: StrictBool = False
    disable_tqdm: StrictBool = False

    @model_validator(mode="after")
    def validate_training_schedule(self):
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive integer")
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled")
        if self.evaluation_strategy == "no":
            raise ValueError(
                "evaluation_strategy must enable evaluation for best-checkpoint selection"
            )
        if self.save_strategy != "best" and self.save_strategy != self.evaluation_strategy:
            raise ValueError("save_strategy must match evaluation_strategy")
        return self


class _LoraSettings(_StrictModel):
    rank: Literal[8, 16, 32]
    alpha: StrictInt = Field(gt=0)
    dropout: StrictFloat = Field(ge=0, lt=1)
    target_modules: list[StrictStr] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target_modules(self):
        if any(not module.strip() for module in self.target_modules):
            raise ValueError("target_modules cannot contain empty names")
        return self


class _InferenceSettings(_StrictModel):
    backend: Literal["transformers", "vllm"]
    quantization: StrictStr = Field(min_length=1)
    batch_sizes: list[StrictInt] = Field(min_length=1)
    warmup_requests: StrictInt = Field(ge=0)
    measured_requests: StrictInt = Field(gt=0)
    max_new_tokens: StrictInt = Field(gt=0, le=1024)

    @model_validator(mode="after")
    def validate_batch_sizes(self):
        if any(batch_size <= 0 for batch_size in self.batch_sizes):
            raise ValueError("inference batch_sizes must contain only positive integers")
        return self


class _ExperimentSettings(_StrictModel):
    project: _ProjectSettings
    model: _ModelSettings
    data: _DataSettings
    training: _TrainingSettings
    lora: _LoraSettings
    inference: _InferenceSettings


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
    """Load and strictly validate the complete YAML experiment configuration."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    try:
        validated = _ExperimentSettings.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid experiment config in {config_path}:\n{exc}") from exc
    config = ExperimentConfig(raw=validated.model_dump())
    _ = config.seed, config.model_name, config.training_method, config.lora_rank
    return config
