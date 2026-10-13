"""Reproducibility helpers shared by training, evaluation, and benchmarks."""

from __future__ import annotations

import os
import random
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
