"""Check runtime and optional-backend readiness without loading model weights."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

PACKAGE_REQUIREMENTS = {
    "torch": ">=2.4,<3",
    "transformers": ">=5.19,<6",
    "datasets": ">=3,<5",
    "accelerate": ">=1,<2",
    "mlflow": ">=2.17,<4",
    "peft": ">=0.14,<1",
    "trl": ">=1.15,<2",
    "vllm": ">=0.7,<1",
}


def _cuda_device_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "error": "PyTorch is not installed"}

    if not torch.cuda.is_available():
        return {"available": False}
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    except (AttributeError, RuntimeError):
        free_bytes = None
        total_bytes = properties.total_memory
    return {
        "available": True,
        "name": properties.name,
        "index": device_index,
        "free_memory_gb": round(free_bytes / (1024**3), 2) if free_bytes is not None else None,
        "total_memory_gb": round(total_bytes / (1024**3), 2),
        "compute_capability": list(torch.cuda.get_device_capability(device_index)),
    }


def _installed_versions(package_names: set[str]) -> dict[str, str | None]:
    versions = {}
    for package_name in sorted(package_names):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def run_preflight(
    config: dict[str, Any],
    method: str | None = None,
    backend: str | None = None,
    quantization: str | None = None,
    *,
    installed_packages: dict[str, str | None] | None = None,
    python_version: tuple[int, int] | None = None,
    platform_name: str | None = None,
    accelerator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return actionable readiness errors/warnings without network/model access."""
    method = (method or config["training"]["method"]).lower()
    backend = (backend or config["inference"]["backend"]).lower()
    quantization = (quantization or config["inference"]["quantization"]).lower()
    platform_name = platform_name or platform.system()
    python_version = python_version or (sys.version_info.major, sys.version_info.minor)
    accelerator = accelerator if accelerator is not None else _cuda_device_info()

    required_packages = {"torch", "transformers", "datasets", "accelerate", "mlflow"}
    if method in {"lora", "qlora"}:
        required_packages.update({"peft", "trl"})
    if backend == "vllm":
        required_packages.add("vllm")
    needs_bitsandbytes = method == "qlora" or quantization in {"4bit", "8bit", "bitsandbytes"}
    if needs_bitsandbytes:
        required_packages.add("bitsandbytes")

    installed_packages = installed_packages or _installed_versions(required_packages)
    errors: list[str] = []
    warnings: list[str] = []

    if python_version < (3, 10):
        errors.append(f"Python 3.10+ is required; found {python_version[0]}.{python_version[1]}")
    for package_name in sorted(required_packages):
        version = installed_packages.get(package_name)
        if version is None:
            errors.append(f"Required package '{package_name}' is not installed")
            continue
        version_spec = PACKAGE_REQUIREMENTS.get(package_name)
        if version_spec:
            try:
                compatible = Version(version) in SpecifierSet(version_spec)
            except InvalidVersion:
                compatible = False
            if not compatible:
                errors.append(
                    f"{package_name} {version} is outside the supported range {version_spec}"
                )

    cuda_available = bool(accelerator.get("available", False))
    if not cuda_available:
        if method == "qlora":
            errors.append("QLoRA requires a CUDA-visible NVIDIA GPU")
        elif method == "lora":
            warnings.append(
                "CUDA is not available; LoRA training will run on CPU and may be very slow"
            )
        elif backend == "vllm":
            errors.append("vLLM requires a CUDA-visible NVIDIA GPU in this project")
        else:
            warnings.append("CUDA is not available; model evaluation will run on CPU")

    if (method == "qlora" or needs_bitsandbytes) and platform_name != "Linux":
        errors.append("bitsandbytes quantization is supported only on Linux in this project")
    if backend == "vllm" and platform_name != "Linux":
        errors.append("vLLM is supported only on Linux in this project")

    return {
        "ok": not errors,
        "method": method,
        "backend": backend,
        "quantization": quantization,
        "model": config["model"]["name_or_path"],
        "revisions": {
            "model": config["model"]["revision"],
            "tokenizer": config["model"]["tokenizer_revision"],
            "dataset": config["data"]["dataset_revision"],
        },
        "python_version": f"{python_version[0]}.{python_version[1]}",
        "platform": platform_name,
        "packages": {name: installed_packages.get(name) for name in sorted(required_packages)},
        "accelerator": accelerator,
        "errors": errors,
        "warnings": warnings,
    }
