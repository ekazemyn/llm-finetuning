"""Generation-based FinQA evaluation and experiment comparison utilities."""

from __future__ import annotations

import csv
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from llm_finetuning.data import load_finqa
from llm_finetuning.modeling import format_user_prompt, load_model, load_tokenizer
from llm_finetuning.reproducibility import runtime_manifest, set_seed
from llm_finetuning.tracking import log_result

_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?%?")


def extract_numeric_answer(text: str) -> Decimal | None:
    """Extract the last numeric answer, preferring text after the final answer marker."""
    segments = re.split(r"final\s+answer\s*:", text, flags=re.IGNORECASE)
    candidate_text = segments[-1] if len(segments) > 1 else text
    matches = _NUMBER.findall(candidate_text)
    if not matches and len(segments) > 1:
        matches = _NUMBER.findall(text)
    if not matches:
        return None
    normalized = matches[-1].replace(",", "").rstrip("%")
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def answers_match(prediction: str, reference: str, tolerance: Decimal = Decimal("0.0001")) -> bool:
    predicted_value = extract_numeric_answer(prediction)
    reference_value = extract_numeric_answer(reference)
    if predicted_value is None or reference_value is None:
        return False
    scale = max(abs(reference_value), Decimal(1))
    return abs(predicted_value - reference_value) <= tolerance * scale


def _move_batch(batch: dict[str, Any], model):
    device = model.get_input_embeddings().weight.device
    return {key: value.to(device) for key, value in batch.items()}


def _batch_rows(dataset, start: int, batch_size: int) -> list[dict[str, Any]]:
    return [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]


def _mean_eval_loss(model, tokenizer, examples, batch_size: int, max_length: int) -> float | None:
    import torch

    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["completion"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for item in batch_examples
            ]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = _move_batch(encoded, model)
            labels = encoded["input_ids"].clone()
            labels[encoded["attention_mask"] == 0] = -100
            outputs = model(**encoded, labels=labels)
            tokens = int((labels[:, 1:] != -100).sum().item())
            if tokens:
                total_loss += float(outputs.loss.item()) * tokens
                total_tokens += tokens
    return total_loss / total_tokens if total_tokens else None


def evaluate(
    config: dict[str, Any],
    method: str = "baseline",
    model_path: str | Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Measure numeric answer accuracy and teacher-forced loss on a fixed split."""
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test.")
    set_seed(int(config["project"]["seed"]))
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model evaluation.") from exc

    examples = load_finqa(config)[split]
    tokenizer = load_tokenizer(
        config["model"]["name_or_path"], config["model"].get("tokenizer_revision", "main")
    )
    tokenizer.padding_side = "left"
    model = load_model(config, method=method, model_path=model_path, inference=True)
    batch_size = int(config["training"]["per_device_eval_batch_size"])
    max_length = int(config["model"]["max_length"])
    max_new_tokens = int(config["inference"]["max_new_tokens"])
    predictions: list[dict[str, str]] = []
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for start in range(0, len(examples), batch_size):
        batch = _batch_rows(examples, start, batch_size)
        encoded = tokenizer(
            [format_user_prompt(tokenizer, item["prompt"]) for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = _move_batch(encoded, model)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        predictions.extend(
            {"id": item["id"], "prediction": text, "reference": item["answer"]}
            for item, text in zip(batch, decoded, strict=True)
        )
    inference_seconds = time.perf_counter() - started
    accuracy = sum(answers_match(row["prediction"], row["reference"]) for row in predictions)
    accuracy = accuracy / len(predictions) if predictions else 0.0
    loss = _mean_eval_loss(model, tokenizer, list(examples), batch_size, max_length)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
    )

    result: dict[str, Any] = {
        "method": method,
        "split": split,
        "num_examples": len(predictions),
        "accuracy": accuracy,
        "eval_loss": loss,
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_allocated_mb": peak_memory_mb,
        "runtime": runtime_manifest(),
        "predictions": predictions,
    }
    output_dir = Path(config["project"]["output_dir"]) / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_label = Path(model_path).name if model_path else "base"
    result_path = output_dir / f"{method}-{checkpoint_label}-{split}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log_result(
        config,
        f"evaluation-{method}-{checkpoint_label}-{split}",
        {"method": method, "split": split, "model_path": model_path},
        {
            "accuracy": accuracy,
            "eval_loss": loss,
            "inference_seconds": inference_seconds,
            "peak_gpu_memory_allocated_mb": peak_memory_mb,
        },
        result_path,
    )
    return result


def write_comparison(results: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write a stable machine-readable summary; unavailable metrics remain blank."""
    fields = [
        "method",
        "lora_rank",
        "seed",
        "accuracy",
        "eval_loss",
        "train_loss",
        "train_runtime_seconds",
        "train_peak_gpu_memory_allocated_mb",
        "eval_peak_gpu_memory_allocated_mb",
        "backend",
        "quantization",
        "batch_size",
        "throughput_requests_per_second",
        "throughput_tokens_per_second",
        "p50_latency_ms",
        "p95_latency_ms",
        "latency_unit",
        "vram_peak_mb",
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
