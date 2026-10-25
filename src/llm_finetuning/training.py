"""Reproducible supervised fine-tuning and experiment tracking."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from llm_finetuning.data import load_finqa
from llm_finetuning.modeling import load_model, load_tokenizer
from llm_finetuning.reproducibility import build_provenance, set_seed


def _warmup_steps(training_config: dict[str, Any]) -> float:
    warmup_ratio = float(training_config["warmup_ratio"])
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in the range [0, 1).")
    return warmup_ratio


def _chat_text(example: dict[str, str], tokenizer, max_length: int) -> dict[str, str]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = example["completion"].strip()
    if tokenizer.eos_token and not completion.endswith(tokenizer.eos_token):
        completion = f"{completion}{tokenizer.eos_token}"

    completion_budget = min(256, max(1, max_length // 3))
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if len(completion_ids) > completion_budget:
        marker = "Final answer:"
        marker_index = completion.rfind(marker)
        if marker_index >= 0:
            answer_ids = tokenizer.encode(completion[marker_index:], add_special_tokens=False)
            if len(answer_ids) >= completion_budget:
                completion_ids = answer_ids[-completion_budget:]
            else:
                reasoning_ids = tokenizer.encode(
                    completion[:marker_index], add_special_tokens=False
                )
                completion_ids = reasoning_ids[: completion_budget - len(answer_ids)] + answer_ids
        else:
            completion_ids = completion_ids[:completion_budget]
        completion = tokenizer.decode(
            completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_budget = max_length - len(completion_ids) - 8
    if prompt_budget < 1:
        raise ValueError("max_length leaves no room for the FinQA question and answer.")
    if len(prompt_ids) > prompt_budget:
        prompt = tokenizer.decode(
            prompt_ids[-prompt_budget:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    return {"prompt": prompt, "completion": completion}


def _mlflow_log(
    config: dict[str, Any], method: str, result: dict[str, Any], output_dir: Path
) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for tracked training runs; install the project dependencies."
        ) from exc

    project = config["project"]
    mlflow.set_tracking_uri(project["mlflow_tracking_uri"])
    mlflow.set_experiment(project["experiment_name"])
    run_name = f"{method}-rank{config['lora']['rank']}-seed{project['seed']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "method": method,
                "seed": project["seed"],
                "model": config["model"]["name_or_path"],
                "model_revision": config["model"].get("revision", "main"),
                "dataset": config["data"]["dataset_name"],
                "dataset_revision": config["data"].get("dataset_revision", "main"),
                "lora_rank": config["lora"]["rank"],
                "max_length": config["model"]["max_length"],
            }
        )
        for name, value in result.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(name, float(value))
        mlflow.log_dict(result["provenance"], "provenance.json")
        mlflow.log_artifact(str(output_dir / "run.json"))


def train(
    config: dict[str, Any],
    method: str | None = None,
    rank: int | None = None,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a configured method and persist a resumable model checkpoint."""
    method = (method or config["training"]["method"]).lower()
    if method not in {"lora", "qlora"}:
        raise ValueError("Training method must be 'lora' or 'qlora'; baseline is evaluation-only.")
    if rank is not None:
        if rank not in {8, 16, 32}:
            raise ValueError("Controlled LoRA ranks are 8, 16, and 32.")
        config = {**config, "lora": {**config["lora"], "rank": rank}}

    set_seed(int(config["project"]["seed"]))
    try:
        import torch
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and TRL are required to train; install the project dependencies."
        ) from exc

    output_dir = (
        Path(config["training"]["output_dir"])
        / f"{method}-r{config['lora']['rank']}-s{config['project']['seed']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(
        config["model"]["name_or_path"], config["model"].get("tokenizer_revision", "main")
    )
    splits = load_finqa(config)
    max_length = int(config["model"]["max_length"])
    train_data = splits["train"].map(
        lambda example: _chat_text(example, tokenizer, max_length),
        remove_columns=splits["train"].column_names,
    )
    eval_data = splits["validation"].map(
        lambda example: _chat_text(example, tokenizer, max_length),
        remove_columns=splits["validation"].column_names,
    )
    model = load_model(config, method=method)

    training_config = config["training"]
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=float(training_config["learning_rate"]),
        num_train_epochs=float(training_config["epochs"]),
        max_steps=int(training_config.get("max_steps", -1)),
        per_device_train_batch_size=int(training_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(training_config["gradient_checkpointing"]),
        warmup_steps=_warmup_steps(training_config),
        weight_decay=float(training_config["weight_decay"]),
        logging_strategy="steps",
        logging_steps=int(training_config["logging_steps"]),
        logging_first_step=True,
        disable_tqdm=bool(training_config.get("disable_tqdm", False)),
        save_strategy=training_config["save_strategy"],
        eval_strategy=training_config["evaluation_strategy"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=int(config["project"]["seed"]),
        data_seed=int(config["project"]["seed"]),
        fp16=bool(training_config["fp16"] and torch.cuda.is_available()),
        bf16=bool(training_config["bf16"] and torch.cuda.is_available()),
        max_length=int(config["model"]["max_length"]),
        completion_only_loss=True,
        eos_token=tokenizer.eos_token,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    peak_memory_allocated_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
    )
    peak_memory_reserved_mb = (
        torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else None
    )
    evaluation = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    provenance = build_provenance(config)

    result: dict[str, Any] = {
        "method": method,
        "seed": int(config["project"]["seed"]),
        "lora_rank": int(config["lora"]["rank"]),
        "train_runtime_seconds": elapsed_seconds,
        "train_loss": train_output.metrics.get("train_loss"),
        "eval_loss": evaluation.get("eval_loss"),
        "peak_gpu_memory_allocated_mb": peak_memory_allocated_mb,
        "peak_gpu_memory_reserved_mb": peak_memory_reserved_mb,
        "global_step": int(train_output.global_step),
        "runtime": provenance["runtime"],
        "provenance": provenance,
    }
    (output_dir / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _mlflow_log(config, method, result, output_dir)
    return result
