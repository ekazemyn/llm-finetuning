"""Small, time-budgeted CPU/CUDA LoRA pilot; separate from the full experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from llm_finetuning.config import load_config
from llm_finetuning.data import _source_cache_directory, load_finqa
from llm_finetuning.modeling import load_model, load_tokenizer
from llm_finetuning.pilot_diagnostics import score_answer, summarize_predictions
from llm_finetuning.pilot_protocol import PROTOCOL, _encode_example
from llm_finetuning.reproducibility import build_provenance, set_seed


def _check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError(
            "Pilot compute budget exhausted; completed stages are saved in metrics.json"
        )


def _batch(example, device):
    import torch

    return {
        key: torch.tensor([example[key]], dtype=torch.long, device=device)
        for key in ("input_ids", "labels")
    }


def _evaluate(model, tokenizer, examples, max_new_tokens, deadline):
    import torch

    model.eval()
    device = model.get_input_embeddings().weight.device
    loss_sum = 0.0
    token_count = 0
    predictions = []
    started = time.perf_counter()
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos_ids = eos if isinstance(eos, list) else [eos]
    eos_ids = sorted({token for token in [*eos_ids, tokenizer.eos_token_id] if token is not None})
    if not eos_ids:
        raise ValueError("Evaluation requires an EOS token")
    with torch.inference_mode():
        for example in examples:
            _check_deadline(deadline)
            batch = _batch(example, device)
            count = sum(label != -100 for label in example["labels"][1:])
            loss_sum += float(model(**batch).loss) * count
            token_count += count
            _check_deadline(deadline)
            prompt = torch.tensor([example["prompt_ids"]], device=device)
            output = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )
            generated = output[0, prompt.shape[1] :].tolist()
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            stopped_on_eos = bool(generated and generated[-1] in eos_ids)
            hit_limit = len(generated) >= max_new_tokens and not stopped_on_eos
            predictions.append(
                {
                    "id": example["id"],
                    "reference": example["answer"],
                    "prediction": prediction,
                    **score_answer(prediction, example["answer"]),
                    "context": example.get("context", {}),
                    "stop_reason": "eos"
                    if stopped_on_eos
                    else ("token_limit" if hit_limit else "other"),
                    "generated_tokens": len(generated),
                    "hit_token_limit": hit_limit,
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    loss = loss_sum / token_count
    return {
        "completion_loss": loss,
        "completion_perplexity": math.exp(loss) if loss < 700 else None,
        **summarize_predictions(predictions),
        "examples": len(predictions),
        "answer_tokens": token_count,
        "evaluation_seconds": time.perf_counter() - started,
        "predictions": predictions,
    }


def run_pilot(config_path="configs/local-pilot.yaml", *, budget_minutes=25):
    """Run matched before/after evaluation, capped training, and save presentation artifacts.

    The budget is cooperative, checked between operations. Initial downloads and a
    single model operation can overrun it; it is not a hard process timeout.
    """
    if not 0 < budget_minutes <= 25:
        raise ValueError("budget_minutes must be greater than zero and at most 25")
    started = time.perf_counter()
    deadline = started + budget_minutes * 60
    config = load_config(config_path).raw
    output = Path(config["training"]["output_dir"]) / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "scope": "small-model FinQA pilot; not a full benchmark",
        "protocol": PROTOCOL,
        "schema_version": 2,
        "scoring": "pilot-strict-answer-v2; numeric/percent tolerance 0.0001; yes/no/text exact",
        "budget_minutes": budget_minutes,
        "config": config,
    }

    def save():
        result["total_seconds"] = time.perf_counter() - started
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    save()
    try:
        import torch

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        set_seed(config["project"]["seed"])
        result["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"Pilot on {result['device']}; artifacts: {output}", flush=True)
        tokenizer = load_tokenizer(
            config["model"]["name_or_path"], config["model"]["tokenizer_revision"]
        )
        splits = load_finqa(config)
        max_length = config["model"]["max_length"]
        max_new_tokens = config["inference"]["max_new_tokens"]
        training = [
            _encode_example(row, tokenizer, max_length, max_new_tokens) for row in splits["train"]
        ]
        # Seeded held-out dev selection, identical for baseline and adapter. Test is untouched.
        validation = splits["validation"].shuffle(seed=config["project"]["seed"])
        validation = validation.select(range(min(8, len(validation))))
        heldout = [
            _encode_example(row, tokenizer, max_length, max_new_tokens) for row in validation
        ]
        result["training_context"] = [row["context"] for row in training]
        result["train_ids"] = [row["id"] for row in training]
        result["heldout_ids"] = [row["id"] for row in heldout]
        result["source_manifest"] = json.loads(
            (_source_cache_directory(config) / "source_manifest.json").read_text(encoding="utf-8")
        )
        result["provenance"] = build_provenance(config)
        model = load_model(config, method="lora")
        model.to(result["device"])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        result["parameters"] = {
            "trainable": trainable,
            "total": total,
            "trainable_percent": 100 * trainable / total,
        }
        max_new_tokens = config["inference"]["max_new_tokens"]
        print("Evaluating baseline on held-out dev examples...", flush=True)
        with model.disable_adapter():
            result["baseline"] = _evaluate(model, tokenizer, heldout, max_new_tokens, deadline)
        save()
        # Reserve time for the same held-out evaluation, saving, and presentation output.
        reserve = max(60, result["baseline"]["evaluation_seconds"] * 1.5)
        train_deadline = min(time.perf_counter() + 600, deadline - reserve)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["training"]["learning_rate"],
        )
        model.train()
        history = []
        training_started = time.perf_counter()
        for step in range(config["training"]["max_steps"]):
            if time.perf_counter() >= train_deadline:
                break
            example = training[step % len(training)]
            optimizer.zero_grad(set_to_none=True)
            loss = model(**_batch(example, result["device"])).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; refusing to report a successful pilot")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            history.append({"step": step + 1, "loss": float(loss.detach())})
            print(f"Step {step + 1}: loss={history[-1]['loss']:.4f}", flush=True)
        result["training"] = {
            "steps": len(history),
            "requested_steps": config["training"]["max_steps"],
            "seconds": time.perf_counter() - training_started,
            "history": history,
        }
        model.save_pretrained(output / "adapter")
        tokenizer.save_pretrained(output / "adapter")
        save()
        if not history:
            raise TimeoutError("No training time remained after baseline evaluation")
        print("Evaluating adapted model on the same held-out examples...", flush=True)
        result["adapted"] = _evaluate(model, tokenizer, heldout, max_new_tokens, deadline)
        result["peak_gpu_allocated_mb"] = (
            torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
        )
        rows = [
            {
                "model": name,
                **{
                    key: result[name][key]
                    for key in (
                        "completion_loss",
                        "completion_perplexity",
                        "numeric_accuracy",
                        "numeric_examples",
                        "answer_accuracy",
                        "examples",
                        "evaluation_seconds",
                    )
                },
            }
            for name in ("baseline", "adapted")
        ]
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        diagnostics = {
            name: {
                "answer_type_metrics": result[name]["answer_type_metrics"],
                "error_analysis": result[name]["error_analysis"],
            }
            for name in ("baseline", "adapted")
        }
        (output / "error_analysis.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        result["status"] = (
            "completed"
            if len(history) == config["training"]["max_steps"]
            else ("completed_with_time_capped_training")
        )
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "incomplete"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    print(f"Finished in {result['total_seconds'] / 60:.1f} minutes. See {output}", flush=True)
    return output, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local-pilot.yaml")
    parser.add_argument("--budget-minutes", type=float, default=25)
    args = parser.parse_args()
    run_pilot(args.config, budget_minutes=args.budget_minutes)
