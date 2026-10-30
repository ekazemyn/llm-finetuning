import pytest

from llm_finetuning.pilot_diagnostics import score_answer, summarize_predictions


@pytest.mark.parametrize(
    "prediction,reference,kind,correct,parse_failure",
    [
        ("Final answer: 1,250.02", "1250", "numeric", True, False),
        ("Final answer: 9.9%", "9.9%", "percent", True, False),
        ("Final answer: -2.6%", "-2.6%", "percent", True, False),
        ("Final answer: yes.", "yes", "yes_no", True, False),
        ("Final answer: operating income", "Operating income", "text", True, False),
        ("Based on revenue in 2024", "2024", "numeric", False, True),
        ("Final answer:", "42", "numeric", False, True),
        ("", "42", "numeric", False, True),
        ("Final answer: 41", "42", "numeric", False, False),
        ("10", "10%", "percent", False, True),
        ("yes and no", "yes", "yes_no", False, True),
        ("Final answer: 42 or 43", "42", "numeric", False, True),
    ],
)
def test_answer_types_and_conservative_parsing(prediction, reference, kind, correct, parse_failure):
    row = score_answer(prediction, reference)
    assert row["reference_type"] == kind
    assert row["correct"] == correct
    assert row["parse_failure"] == parse_failure


def test_diagnostics_separate_types_and_overlapping_failure_conditions():
    pairs = [("42", "42"), ("unfinished in 2024", "5%"), ("no", "yes"), ("income", "income")]
    rows = [
        {
            **score_answer(prediction, reference),
            "hit_token_limit": index == 1,
            "stop_reason": "token_limit" if index == 1 else "eos",
            "context": {"evidence_truncated": index == 1, "no_source_evidence": False},
        }
        for index, (prediction, reference) in enumerate(pairs)
    ]
    metrics = summarize_predictions(rows)
    assert metrics["numeric_examples"] == 2
    assert metrics["numeric_accuracy"] == 0.5
    assert metrics["answer_accuracy"] == 0.5
    assert metrics["answer_type_metrics"]["yes_no"]["examples"] == 1
    assert metrics["error_analysis"]["parse_failures"] == 1
    assert metrics["error_analysis"]["generations_at_token_limit"] == 1
    assert metrics["error_analysis"]["wrong_with_truncated_evidence"] == 1
    assert metrics["error_analysis"]["incorrect"] == 2


def test_empty_denominators_are_not_reported_as_zero_accuracy():
    metrics = summarize_predictions([])
    assert metrics["numeric_accuracy"] is None
    assert metrics["answer_accuracy"] is None
    assert metrics["answer_type_metrics"]["percent"]["accuracy"] is None
