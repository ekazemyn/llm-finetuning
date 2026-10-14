"""Inference adapters shared by the HTTP API and deployment entrypoint."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from llm_finetuning.modeling import format_user_prompt, load_model, load_tokenizer


class TransformersGenerator:
    def __init__(
        self,
        config: dict[str, Any],
        method: str,
        model_path: str | Path | None,
        quantization: str = "none",
    ) -> None:
        import torch

        self.tokenizer = load_tokenizer(
            config["model"]["name_or_path"], config["model"].get("tokenizer_revision", "main")
        )
        self.tokenizer.padding_side = "left"
        self.model = load_model(
            config,
            method=method,
            model_path=model_path,
            inference=True,
            quantization=quantization,
        )
        self.model.eval()
        self.torch = torch
        self.max_length = int(config["model"]["max_length"])
        self.backend = "transformers"
        self._lock = threading.Lock()

    def generate(self, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
        with self._lock, self.torch.inference_mode():
            encoded = self.tokenizer(
                [format_user_prompt(self.tokenizer, prompt) for prompt in prompts],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            device = self.model.get_input_embeddings().weight.device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generation_args: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_args["temperature"] = temperature
            generated = self.model.generate(**encoded, **generation_args)
            prompt_width = encoded["input_ids"].shape[1]
            return self.tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )


class VLLMGenerator:
    def __init__(
        self,
        config: dict[str, Any],
        method: str,
        model_path: str | Path | None,
        quantization: str = "none",
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is Linux/CUDA-only; install the optional vLLM dependencies."
            ) from exc
        self.tokenizer = load_tokenizer(
            config["model"]["name_or_path"], config["model"].get("tokenizer_revision", "main")
        )
        adapter_path = (
            model_path
            if model_path and (Path(model_path) / "adapter_config.json").is_file()
            else None
        )
        self.engine_kwargs: dict[str, Any] = {
            "model": str(
                config["model"]["name_or_path"]
                if adapter_path
                else model_path or config["model"]["name_or_path"]
            ),
            "trust_remote_code": bool(config["model"].get("trust_remote_code", False)),
            "max_model_len": int(config["model"]["max_length"]),
        }
        if quantization != "none":
            self.engine_kwargs["quantization"] = quantization
        self.lora_request = None
        if method in {"lora", "qlora"} and adapter_path:
            from vllm.lora.request import LoRARequest

            self.engine_kwargs.update(enable_lora=True, max_lora_rank=int(config["lora"]["rank"]))
            self.lora_request = LoRARequest("finqa-adapter", 1, str(adapter_path))
        self.engine = LLM(**self.engine_kwargs)
        self.backend = "vllm"
        self._lock = threading.Lock()

    def generate(self, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
        from vllm import SamplingParams

        with self._lock:
            outputs = self.engine.generate(
                [format_user_prompt(self.tokenizer, prompt) for prompt in prompts],
                SamplingParams(temperature=temperature, max_tokens=max_tokens),
                lora_request=self.lora_request,
            )
        return [output.outputs[0].text for output in outputs]
