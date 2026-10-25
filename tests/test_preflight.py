from pathlib import Path

import pytest

from llm_finetuning.config import load_config
from llm_finetuning.preflight import run_preflight


@pytest.fixture
def config():
    return load_config(Path("configs/experiment.yaml")).raw


@pytest.fixture
def installed_packages():
    return {
        "torch": "2.11.0+cu130",
        "transformers": "5.19.0",
        "datasets": "4.8.5",
        "accelerate": "1.15.0",
        "mlflow": "3.17.1",
        "peft": "0.21.1",
        "trl": "1.15.0",
        "bitsandbytes": "0.46.0",
        "vllm": "0.7.0",
    }


@pytest.fixture
def cuda_accelerator():
    return {
        "available": True,
        "name": "Tesla T4",
        "index": 0,
        "free_memory_gb": 14.0,
        "total_memory_gb": 15.0,
        "compute_capability": [7, 5],
    }


def test_preflight_reports_supported_runtime_without_loading_model(
    config, installed_packages, cuda_accelerator
):
    result = run_preflight(
        config,
        method="lora",
        backend="transformers",
        installed_packages=installed_packages,
        python_version=(3, 13),
        platform_name="Linux",
        accelerator=cuda_accelerator,
    )

    assert result["ok"] is True
    assert result["accelerator"]["name"] == "Tesla T4"
    assert result["revisions"]["dataset"] == config["data"]["dataset_revision"]
    assert result["errors"] == []


def test_qlora_preflight_requires_bitsandbytes(config, installed_packages, cuda_accelerator):
    installed_packages.pop("bitsandbytes")
    result = run_preflight(
        config,
        method="qlora",
        backend="transformers",
        quantization="4bit",
        installed_packages=installed_packages,
        platform_name="Linux",
        accelerator=cuda_accelerator,
    )

    assert result["ok"] is False
    assert any("bitsandbytes" in error for error in result["errors"])


def test_vllm_preflight_rejects_windows_and_missing_cuda(config, installed_packages):
    result = run_preflight(
        config,
        method="baseline",
        backend="vllm",
        installed_packages=installed_packages,
        platform_name="Windows",
        accelerator={"available": False},
    )

    assert result["ok"] is False
    assert any("vLLM is supported only on Linux" in error for error in result["errors"])
    assert any("CUDA-visible NVIDIA GPU" in error for error in result["errors"])


def test_preflight_warns_for_cpu_baseline_and_rejects_old_python(config, installed_packages):
    result = run_preflight(
        config,
        method="baseline",
        backend="transformers",
        installed_packages=installed_packages,
        python_version=(3, 9),
        platform_name="Windows",
        accelerator={"available": False},
    )

    assert result["ok"] is False
    assert any("Python 3.10+" in error for error in result["errors"])
    assert any("run on CPU" in warning for warning in result["warnings"])


def test_preflight_rejects_unsupported_core_package_version(
    config, installed_packages, cuda_accelerator
):
    installed_packages["transformers"] = "4.57.0"
    result = run_preflight(
        config,
        method="baseline",
        backend="transformers",
        installed_packages=installed_packages,
        platform_name="Linux",
        accelerator=cuda_accelerator,
    )

    assert result["ok"] is False
    assert any("transformers 4.57.0" in error for error in result["errors"])
