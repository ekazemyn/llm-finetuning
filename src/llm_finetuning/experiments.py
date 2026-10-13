"""End-to-end comparison and controlled adapter rank/seed ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_finetuning.evaluation import evaluate, write_comparison
from llm_finetuning.training import train


def _checkpoint_path(config: dict[str, Any], method: str, rank: int, seed: int) -> Path:
    return Path(config["training"]["output_dir"]) / f"{method}-r{rank}-s{seed}"


def _run_one(config: dict[str, Any], method: str, rank: int, seed: int) -> dict[str, Any]:
    run_config = {
        **config,
        "project": {**config["project"], "seed": seed},
        "lora": {**config["lora"], "rank": rank},
    }
    checkpoint = _checkpoint_path(run_config, method, rank, seed)
    training_metrics = train(run_config, method=method, rank=rank)
    evaluation = evaluate(run_config, method=method, model_path=checkpoint, split="test")
    return {
        "method": method,
        "lora_rank": rank,
        "seed": seed,
        "accuracy": evaluation["accuracy"],
        "eval_loss": evaluation["eval_loss"],
        "train_loss": training_metrics["train_loss"],
        "train_runtime_seconds": training_metrics["train_runtime_seconds"],
        "train_peak_gpu_memory_allocated_mb": training_metrics["peak_gpu_memory_allocated_mb"],
        "eval_peak_gpu_memory_allocated_mb": evaluation["peak_gpu_memory_allocated_mb"],
    }


def compare(config: dict[str, Any], methods: tuple[str, ...] = ("baseline", "lora", "qlora")):
    """Evaluate the zero-shot baseline and train/evaluate requested adapter methods."""
    results = []
    for method in methods:
        if method == "baseline":
            evaluation = evaluate(config, method="baseline", split="test")
            results.append(
                {
                    "method": "baseline",
                    "lora_rank": None,
                    "seed": config["project"]["seed"],
                    "accuracy": evaluation["accuracy"],
                    "eval_loss": evaluation["eval_loss"],
                    "train_loss": None,
                    "train_runtime_seconds": None,
                    "train_peak_gpu_memory_allocated_mb": None,
                    "eval_peak_gpu_memory_allocated_mb": evaluation["peak_gpu_memory_allocated_mb"],
                }
            )
        elif method in {"lora", "qlora"}:
            results.append(
                _run_one(
                    config, method, int(config["lora"]["rank"]), int(config["project"]["seed"])
                )
            )
        else:
            raise ValueError(f"Unknown comparison method: {method}")
    write_comparison(results, Path(config["project"]["output_dir"]) / "comparison.csv")
    return results


def ablate(
    config: dict[str, Any],
    methods: tuple[str, ...] = ("lora", "qlora"),
    ranks: tuple[int, ...] = (8, 16, 32),
    seeds: tuple[int, ...] | None = None,
):
    """Run the same rank grid with identical seeds and all other settings fixed."""
    seeds = seeds or (int(config["project"]["seed"]),)
    if set(ranks) - {8, 16, 32}:
        raise ValueError("Controlled LoRA ranks are 8, 16, and 32.")
    results = []
    for method in methods:
        if method not in {"lora", "qlora"}:
            raise ValueError("Rank ablations apply only to lora and qlora.")
        for rank in ranks:
            for seed in seeds:
                results.append(_run_one(config, method, rank, seed))
    write_comparison(results, Path(config["project"]["output_dir"]) / "ablations.csv")
    return results
