from decimal import Decimal

from llm_finetuning.evaluation import answers_match, extract_numeric_answer


def test_extract_final_numeric_answer():
    assert extract_numeric_answer("Compute 120 / 3. Final answer: 40.0") == Decimal("40.0")
    assert extract_numeric_answer("No numeric result") is None


def test_numeric_match_tolerance_and_percent():
    assert answers_match("Final answer: 1,250.02", "1250")
    assert answers_match("Final answer: 12.5%", "12.5")
    assert not answers_match("Final answer: 13", "12")
