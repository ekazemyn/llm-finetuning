"""FinQA loading and deterministic conversion to instruction-tuning examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREPROCESSING_VERSION = 1


def _cache_metadata(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config["data"]
    return {
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset_name": data_config["dataset_name"],
        "dataset_revision": data_config.get("dataset_revision"),
        "source_repository": data_config.get("source_repository", "czyssrs/FinQA"),
        "source_splits": {
            "train": data_config["train_split"],
            "validation": data_config["validation_split"],
            "test": data_config["test_split"],
        },
        "sample_limits": {
            "train": data_config.get("max_train_samples"),
            "validation": data_config.get("max_eval_samples"),
            "test": data_config.get("max_test_samples"),
        },
        "seed": int(config["project"]["seed"]),
    }


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(item) for item in value if item is not None)
    return str(value)


def _render_table(table: Any) -> str:
    if not table:
        return ""
    if isinstance(table, str):
        return table.strip()
    if isinstance(table, dict):
        rows = table.get("rows", [])
    else:
        rows = table
    rendered = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            rendered.append(" | ".join(_as_text(cell) for cell in row))
        else:
            rendered.append(_as_text(row))
    return "\n".join(rendered)


def normalize_example(row: dict[str, Any]) -> dict[str, str]:
    """Normalize the IBM FinQA record shape and common flattened variants."""
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    question = _as_text(qa.get("question", row.get("question")))
    answer = _as_text(qa.get("answer")) or _as_text(row.get("final_result", row.get("answer")))
    if not answer and qa.get("steps"):
        final_step = qa["steps"][-1]
        answer = _as_text(final_step.get("res") if isinstance(final_step, dict) else final_step)
    if not question or not answer:
        raise ValueError(
            f"FinQA record is missing a question or answer: {row.get('id', '<unknown>')}"
        )

    context_parts = [
        _as_text(row.get("pre_text")),
        _render_table(row.get("table")),
        _as_text(row.get("post_text")),
    ]
    context = "\n\n".join(part for part in context_parts if part)
    prompt = (
        "Solve the financial question using the supplied evidence. "
        "Show concise arithmetic when useful and end with `Final answer: <number>`.\n\n"
        f"Evidence:\n{context or '(no additional evidence provided)'}\n\n"
        f"Question: {question}"
    )
    rationale = _as_text(qa.get("rationale", row.get("rationale")))
    program = _as_text(qa.get("program", row.get("program_re")))
    reasoning = rationale or (f"Reference program: {program}" if program else "")
    completion = f"{reasoning}\nFinal answer: {answer}" if reasoning else f"Final answer: {answer}"
    return {
        "id": _as_text(row.get("id", "")),
        "prompt": prompt,
        "completion": completion,
        "answer": answer,
    }


def load_finqa(config: dict[str, Any]):
    """Load configured FinQA splits and map records without changing source data."""
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "Install the project dependencies to load FinQA (pip install -e .)"
        ) from exc

    data_config = config["data"]
    revision = data_config.get("dataset_revision")
    prepared_dir_value = data_config.get("prepared_dir", "data/finqa-prepared")
    prepared_dir = Path(prepared_dir_value) if prepared_dir_value else None
    if prepared_dir is not None and prepared_dir.exists():
        manifest_path = prepared_dir / "data_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Prepared dataset has no manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = _cache_metadata(config)
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            raise ValueError(f"Prepared dataset metadata mismatch for: {', '.join(mismatches)}")
        dataset = load_from_disk(str(prepared_dir))
        return {name: dataset[name] for name in ("train", "validation", "test")}

    split_names = {
        "train": data_config["train_split"],
        "validation": data_config["validation_split"],
        "test": data_config["test_split"],
    }
    source_revision = str(revision or "")
    if len(source_revision) != 40 or any(
        char not in "0123456789abcdef" for char in source_revision
    ):
        raise ValueError("FinQA dataset_revision must be a pinned 40-character source commit SHA.")
    source_repo = data_config.get("source_repository", "czyssrs/FinQA")
    source_files = {
        split_name: (
            f"https://raw.githubusercontent.com/{source_repo}/{source_revision}/dataset/"
            f"{split_name}.json"
        )
        for split_name in split_names.values()
    }
    dataset = load_dataset("json", data_files=source_files)
    missing = [name for name in split_names.values() if name not in dataset]
    if missing:
        raise ValueError(
            f"FinQA source is missing configured splits {missing}; available: {list(dataset)}"
        )

    normalized = {}
    for canonical_name, source_name in split_names.items():
        split = dataset[source_name].map(normalize_example)
        sample_limit_key = (
            "max_eval_samples"
            if canonical_name == "validation"
            else f"max_{canonical_name}_samples"
        )
        max_samples = data_config.get(sample_limit_key)
        if max_samples is not None:
            limit = min(int(max_samples), len(split))
            if canonical_name == "train":
                split = split.shuffle(seed=int(config["project"]["seed"]))
            split = split.select(range(limit))
        normalized[canonical_name] = split
    return normalized


def prepare_finqa(config: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    """Download, normalize, and persist the exact splits used by subsequent runs."""
    try:
        from datasets import DatasetDict
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to prepare FinQA.") from exc
    destination = Path(output_dir or config["data"].get("prepared_dir", "data/finqa-prepared"))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing prepared data at {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    splits = load_finqa({**config, "data": {**config["data"], "prepared_dir": ""}})
    DatasetDict(splits).save_to_disk(str(destination))
    manifest = {
        **_cache_metadata(config),
        "splits": {name: len(split) for name, split in splits.items()},
        "split_fingerprints": {
            name: getattr(split, "_fingerprint", None) for name, split in splits.items()
        },
    }
    (destination / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return destination
