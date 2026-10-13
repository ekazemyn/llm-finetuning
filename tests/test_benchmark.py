from llm_finetuning.benchmark import _percentile


def test_percentile_uses_nearest_rank():
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.50) == 2.0
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0
    assert _percentile([], 0.95) is None
