"""Shared loading for the pretrained baseline, LoRA, and QLoRA models."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install the project dependencies first.") from exc
    return torch


def load_tokenizer(model_name: str, revision: str = "main"):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required. Install the project dependencies first."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    config: dict[str, Any],
    method: str | None = None,
    model_path: str | Path | None = None,
    inference: bool = False,
    quantization: str | None = None,
):
    """Load a base model or attach a trainable adapter; QLoRA is Linux/CUDA-only."""
    torch = _require_torch()
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required. Install the project dependencies first."
        ) from exc

    model_config = config["model"]
    training_config = config["training"]
    method = (method or training_config["method"]).lower()
    if method not in {"baseline", "lora", "qlora"}:
        raise ValueError(f"Unsupported model method: {method}")
    if inference and method in {"lora", "qlora"} and model_path is None:
        raise ValueError(
            f"A trained {method.upper()} adapter checkpoint is required for inference."
        )
    source = str(model_path or model_config["name_or_path"])
    quantization = quantization or ("4bit" if method == "qlora" else "none")
    if quantization not in {"none", "4bit", "8bit"}:
        raise ValueError("Transformers quantization must be none, 4bit, or 8bit.")
    if method == "qlora" and quantization != "4bit":
        raise ValueError("QLoRA training requires 4-bit NF4 quantization.")
    is_adapter = bool(model_path and (Path(model_path) / "adapter_config.json").is_file())
    if model_path and method in {"lora", "qlora"} and not is_adapter:
        raise ValueError(f"Adapter checkpoint is missing adapter_config.json: {model_path}")
    if is_adapter:
        source = str(model_config["name_or_path"])
    revision = model_config.get("revision", "main")
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", False))
    }
    if revision:
        load_kwargs["revision"] = revision
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.float16

    if quantization != "none":
        if not torch.cuda.is_available():
            raise RuntimeError("4/8-bit bitsandbytes loading requires a CUDA GPU.")
        import platform

        if platform.system() != "Linux":
            raise RuntimeError(
                "The configured bitsandbytes quantization stack is supported on Linux only."
            )
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install the Linux QLoRA extra: pip install -e '.[qlora]'") from exc
        if quantization == "4bit":
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
    if not inference:
        model.config.use_cache = False
    if is_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load this adapter checkpoint.") from exc
        model = PeftModel.from_pretrained(model, str(model_path), is_trainable=not inference)
    elif method in {"lora", "qlora"} and model_path is None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError(
                "PEFT is required for LoRA/QLoRA. Install the project dependencies."
            ) from exc
        if method == "qlora":
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(
                    training_config.get("gradient_checkpointing", True)
                ),
            )
        adapter_config = config["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(adapter_config["rank"]),
                lora_alpha=int(adapter_config["alpha"]),
                lora_dropout=float(adapter_config["dropout"]),
                target_modules=list(adapter_config["target_modules"]),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    return model
