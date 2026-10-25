"""Reproducibility helpers shared by training, evaluation, and benchmarks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python and available ML libraries before creating data/model objects."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def runtime_manifest() -> dict[str, Any]:
    """Return machine/runtime metadata to attach to every recorded run."""
    import platform
    import sys

    manifest: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        manifest["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            manifest["gpu_name"] = torch.cuda.get_device_name(0)
            manifest["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        manifest["torch_version"] = None
        manifest["cuda_available"] = False
    return manifest


def build_provenance(
    config: dict[str, Any], project_root: str | Path | None = None
) -> dict[str, Any]:
    """Capture the resolved experiment inputs, source checkout, packages, and runtime."""
    resolved_config = json.loads(json.dumps(config, sort_keys=True, default=str))
    canonical_config = json.dumps(
        resolved_config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]

    git_commit = None
    git_branch = None
    git_dirty = None
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_branch = subprocess.run(
            ["git", "-C", str(root), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass

    core_packages = (
        "accelerate",
        "bitsandbytes",
        "datasets",
        "mlflow",
        "numpy",
        "peft",
        "torch",
        "transformers",
        "trl",
        "vllm",
    )
    packages = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    for package_name in core_packages:
        try:
            packages.setdefault(package_name, importlib.metadata.version(package_name))
        except importlib.metadata.PackageNotFoundError:
            packages.setdefault(package_name, None)

    return {
        "schema_version": 1,
        "config_sha256": hashlib.sha256(canonical_config.encode("utf-8")).hexdigest(),
        "resolved_config": resolved_config,
        "git": {"commit": git_commit, "branch": git_branch, "dirty": git_dirty},
        "source_revisions": {
            "model": config["model"]["revision"],
            "tokenizer": config["model"]["tokenizer_revision"],
            "dataset": config["data"]["dataset_revision"],
        },
        "packages": packages,
        "runtime": runtime_manifest(),
    }
