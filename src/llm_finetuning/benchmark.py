"""Controlled Transformers and vLLM generation benchmarks."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from llm_finetuning.data import load_finqa
from llm_finetuning.modeling import load_model, load_tokenizer
from llm_finetuning.reproducibility import runtime_manifest, set_seed
from llm_finetuning.tracking import log_result


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return ordered[index]


def _generate_transformers_batches(
    prompts: list[str], tokenizer, model, torch, config: dict[str, Any], batch_size: int
) -> tuple[list[float], int]:
    latencies: list[float] = []
    output_tokens = 0
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(config["model"]["max_length"]),
        )
        device = model.get_input_embeddings().weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(config["inference"]["max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)
        output_tokens += sum(
            [max(0, int(generated.shape[1]) - int(encoded["input_ids"].shape[1]))] * len(chunk)
        )
    return latencies, output_tokens


def _generate_vllm_batches(prompts, engine, sampling, lora_request, batch_size: int):
    latencies = []
    output_tokens = 0
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        started = time.perf_counter()
        outputs = engine.generate(chunk, sampling, lora_request=lora_request)
        latencies.append(time.perf_counter() - started)
        output_tokens += sum(len(output.outputs[0].token_ids) for output in outputs)
    return latencies, output_tokens


def _transformers_benchmark(
    prompts: list[str],
    config: dict[str, Any],
    method: str,
    model_path,
    batch_size: int,
    quantization: str,
    warmup_count: int,
) -> tuple[list[float], int, float | None]:
    import torch

    tokenizer = load_tokenizer(
        config["model"]["name_or_path"], config["model"].get("tokenizer_revision", "main")
    )
    tokenizer.padding_side = "left"
    model = load_model(
        config,
        method=method,
        model_path=model_path,
        inference=True,
        quantization=quantization,
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    warmup_count = max(0, min(warmup_count, len(prompts)))
    _generate_transformers_batches(
        prompts[:warmup_count], tokenizer, model, torch, config, batch_size
    )
    batch_latencies, output_tokens = _generate_transformers_batches(
        prompts[warmup_count:], tokenizer, model, torch, config, batch_size
    )
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
    del model
    return batch_latencies, output_tokens, peak_vram


def _vllm_benchmark(
    prompts: list[str],
    config: dict[str, Any],
    method: str,
    model_path,
    batch_size: int,
    quantization: str,
    warmup_count: int,
) -> tuple[list[float], int, float | None]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is Linux/CUDA-only; install it with pip install -e '.[vllm]'."
        ) from exc
    if method in {"lora", "qlora"} and (
        model_path is None or not (Path(model_path) / "adapter_config.json").is_file()
    ):
        raise ValueError(f"vLLM requires a trained {method.upper()} adapter checkpoint.")

    adapter_path = (
        model_path if model_path and (Path(model_path) / "adapter_config.json").is_file() else None
    )
    engine_kwargs: dict[str, Any] = {
        "model": str(
            config["model"]["name_or_path"]
            if adapter_path
            else model_path or config["model"]["name_or_path"]
        ),
        "trust_remote_code": bool(config["model"].get("trust_remote_code", False)),
        "max_model_len": int(config["model"]["max_length"]),
    }
    if quantization and quantization != "none":
        engine_kwargs["quantization"] = quantization
    if method in {"lora", "qlora"} and adapter_path:
        engine_kwargs.update(enable_lora=True, max_lora_rank=int(config["lora"]["rank"]))
    engine = LLM(**engine_kwargs)
    sampling = SamplingParams(
        temperature=0.0, max_tokens=int(config["inference"]["max_new_tokens"])
    )
    lora_request = None
    if method in {"lora", "qlora"} and adapter_path:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("finqa-adapter", 1, str(adapter_path))
    try:
        import pynvml

        pynvml.nvmlInit()
        device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        peak_vram = pynvml.nvmlDeviceGetMemoryInfo(device_handle).used / (1024**2)
    except Exception:
        pynvml = None
        device_handle = None
        peak_vram = None
    warmup_count = max(0, min(warmup_count, len(prompts)))
    _generate_vllm_batches(prompts[:warmup_count], engine, sampling, lora_request, batch_size)
    batch_latencies, output_tokens = _generate_vllm_batches(
        prompts[warmup_count:], engine, sampling, lora_request, batch_size
    )
    if pynvml is not None and device_handle is not None:
        peak_vram = max(
            peak_vram or 0, pynvml.nvmlDeviceGetMemoryInfo(device_handle).used / (1024**2)
        )
    del engine
    if pynvml is not None:
        pynvml.nvmlShutdown()
    return batch_latencies, output_tokens, peak_vram


def benchmark(
    config: dict[str, Any],
    backend: str,
    method: str = "baseline",
    model_path: str | Path | None = None,
    quantization: str = "none",
    batch_sizes: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Benchmark fixed prompts, warm up each batch size, and report call latency/throughput."""
    if backend not in {"transformers", "vllm"}:
        raise ValueError("Backend must be transformers or vllm.")
    set_seed(int(config["project"]["seed"]))
    examples = load_finqa(config)["test"]
    warmup_count = int(config["inference"]["warmup_requests"])
    measured_count = int(config["inference"]["measured_requests"])
    if len(examples) < warmup_count + measured_count:
        required_count = warmup_count + measured_count
        raise ValueError(
            f"Benchmark needs at least {required_count} test prompts; found {len(examples)}."
        )
    prompts = [examples[index]["prompt"] for index in range(warmup_count + measured_count)]
    sizes = batch_sizes or list(config["inference"]["batch_sizes"])
    results = []

    for batch_size in sizes:
        runner = _transformers_benchmark if backend == "transformers" else _vllm_benchmark
        selected_quantization = (
            "4bit" if method == "qlora" and quantization == "none" else quantization
        )
        if backend == "vllm" and selected_quantization in {"4bit", "8bit"}:
            selected_quantization = "bitsandbytes"
        if batch_size < 1:
            raise ValueError("Batch sizes must be positive integers.")
        latencies, output_tokens, peak_vram = runner(
            prompts, config, method, model_path, batch_size, selected_quantization, warmup_count
        )
        elapsed = sum(latencies)
        result = {
            "backend": backend,
            "method": method,
            "quantization": selected_quantization,
            "batch_size": batch_size,
            "num_requests": measured_count,
            "throughput_requests_per_second": measured_count / elapsed if elapsed else 0.0,
            "throughput_tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
            "p50_latency_ms": statistics.median(latencies) * 1000 if latencies else None,
            "p95_latency_ms": _percentile(latencies, 0.95) * 1000 if latencies else None,
            "latency_unit": "generation_batch_call",
            "vram_peak_mb": peak_vram,
            "runtime": runtime_manifest(),
        }
        results.append(result)

    output_dir = Path(config["project"]["output_dir"]) / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{backend}-{method}-{quantization}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        log_result(
            config,
            f"benchmark-{backend}-{method}-b{result['batch_size']}",
            {
                "backend": backend,
                "method": method,
                "quantization": result["quantization"],
                "batch_size": result["batch_size"],
            },
            {
                "throughput_requests_per_second": result["throughput_requests_per_second"],
                "throughput_tokens_per_second": result["throughput_tokens_per_second"],
                "p50_latency_ms": result["p50_latency_ms"],
                "p95_latency_ms": result["p95_latency_ms"],
                "vram_peak_mb": result["vram_peak_mb"],
            },
            output_path,
        )
    return results
