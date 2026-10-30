"""Conservative answer parsing and descriptive pilot failure counts."""

from __future__ import annotations

import re
from decimal import Decimal

_NUMBER = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?")


def _parse_answer(text):
    candidate = re.split(r"final\s+answer\s*:", text, flags=re.IGNORECASE)[-1].strip()
    candidate = candidate.rstrip(".!?").strip()
    if not candidate:
        return "empty", None
    if _NUMBER.fullmatch(candidate):
        return ("percent" if candidate.endswith("%") else "numeric"), Decimal(
            candidate.replace(",", "").rstrip("%")
        )
    if candidate.casefold() in {"yes", "no"}:
        return "yes_no", candidate.casefold()
    return "text", candidate.casefold()


def score_answer(prediction, reference):
    """Require a complete answer, not an incidental year or number in generated prose."""
    reference_type, expected = _parse_answer(reference)
    predicted_type, actual = _parse_answer(prediction)
    if reference_type == "empty":
        raise ValueError("Cannot score an empty reference")
    parse_failure = predicted_type == "empty" or (
        reference_type != "text" and predicted_type != reference_type
    )
    if reference_type in {"numeric", "percent"}:
        correct = not parse_failure and abs(actual - expected) <= Decimal("0.0001") * max(
            abs(expected), Decimal(1)
        )
    else:
        correct = not parse_failure and actual == expected
    return {
        "reference_type": reference_type,
        "predicted_type": predicted_type,
        "parsed_prediction": str(actual) if actual is not None else None,
        "parse_failure": parse_failure,
        "correct": bool(correct),
        "outcome": "correct"
        if correct
        else (
            "empty_output"
            if predicted_type == "empty"
            else ("parse_failure" if parse_failure else "incorrect_answer")
        ),
    }


def summarize_predictions(predictions):
    """Counts describe observed conditions, not inferred causes of wrong answers."""
    types = {}
    for name in ("numeric", "percent", "yes_no", "text"):
        rows = [row for row in predictions if row["reference_type"] == name]
        count = len(rows)
        correct = sum(row["correct"] for row in rows)
        types[name] = {
            "examples": count,
            "correct": correct,
            "accuracy": correct / count if count else None,
            "parse_failures": sum(row["parse_failure"] for row in rows),
        }
    count = len(predictions)
    correct = sum(row["correct"] for row in predictions)
    failures = sum(row["parse_failure"] for row in predictions)
    numeric = [row for row in predictions if row["reference_type"] in {"numeric", "percent"}]
    return {
        "answer_accuracy": correct / count if count else None,
        "numeric_accuracy": sum(row["correct"] for row in numeric) / len(numeric)
        if numeric
        else None,
        "numeric_examples": len(numeric),
        "answer_type_metrics": types,
        "error_analysis": {
            "examples": count,
            "correct": correct,
            "incorrect": count - correct,
            "parse_failures": failures,
            "parse_failure_rate": failures / count if count else None,
            "empty_outputs": sum(row["outcome"] == "empty_output" for row in predictions),
            "generations_at_token_limit": sum(row["hit_token_limit"] for row in predictions),
            "eos_stopped": sum(row["stop_reason"] == "eos" for row in predictions),
            "evidence_truncated": sum(
                row["context"].get("evidence_truncated", False) for row in predictions
            ),
            "no_source_evidence": sum(
                row["context"].get("no_source_evidence", False) for row in predictions
            ),
            "wrong_with_truncated_evidence": sum(
                not row["correct"] and row["context"].get("evidence_truncated", False)
                for row in predictions
            ),
        },
    }
