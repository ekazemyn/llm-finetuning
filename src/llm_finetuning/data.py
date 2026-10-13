"""FinQA loading and deterministic conversion to instruction-tuning examples."""

from __future__ import annotations

from typing import Any


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
    answer = _as_text(qa.get("answer", row.get("answer")))
    if not question or not answer:
        raise ValueError(f"FinQA record is missing a question or answer: {row.get('id', '<unknown>')}")

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
    completion = f"{rationale}\nFinal answer: {answer}" if rationale else f"Final answer: {answer}"
    return {
        "id": _as_text(row.get("id", "")),
        "prompt": prompt,
        "completion": completion,
        "answer": answer,
    }


def load_finqa(config: dict[str, Any]):
    """Load configured FinQA splits and map records without changing source data."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to load FinQA (pip install -e .)") from exc

    data_config = config["data"]
    revision = data_config.get("dataset_revision")
    dataset = load_dataset(
        data_config["dataset_name"],
        revision=revision,
    )
    split_names = {
        "train": data_config["train_split"],
        "validation": data_config["validation_split"],
        "test": data_config["test_split"],
    }
    missing = [name for name in split_names.values() if name not in dataset]
    if missing:
        raise ValueError(f"FinQA source is missing configured splits {missing}; available: {list(dataset)}")

    normalized = {}
    for canonical_name, source_name in split_names.items():
        split = dataset[source_name].map(normalize_example)
        sample_limit_key = "max_eval_samples" if canonical_name == "validation" else f"max_{canonical_name}_samples"
        max_samples = data_config.get(sample_limit_key)
        if max_samples is not None:
            limit = min(int(max_samples), len(split))
            if canonical_name == "train":
                split = split.shuffle(seed=int(config["project"]["seed"]))
            split = split.select(range(limit))
        normalized[canonical_name] = split
    return normalized
