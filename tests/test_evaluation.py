from decimal import Decimal

from llm_finetuning.evaluation import _batch_rows, answers_match, extract_numeric_answer


def test_extract_final_numeric_answer():
    assert extract_numeric_answer("Compute 120 / 3. Final answer: 40.0") == Decimal("40.0")
    assert extract_numeric_answer("No numeric result") is None


def test_numeric_match_tolerance_and_percent():
    assert answers_match("Final answer: 1,250.02", "1250")
    assert answers_match("Final answer: 12.5%", "12.5")
    assert not answers_match("Final answer: 13", "12")


def test_batch_rows_uses_integer_access_for_columnar_datasets():
    rows = [
        {"id": "1", "prompt": "first", "answer": "1"},
        {"id": "2", "prompt": "second", "answer": "2"},
        {"id": "3", "prompt": "third", "answer": "3"},
    ]

    class ColumnarDataset:
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            if isinstance(index, slice):
                selected = rows[index]
                return {key: [row[key] for row in selected] for key in rows[0]}
            return rows[index]

    assert _batch_rows(ColumnarDataset(), 1, 2) == rows[1:]
