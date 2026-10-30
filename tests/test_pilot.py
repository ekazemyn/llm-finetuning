import json
import time
from pathlib import Path

import pytest

from llm_finetuning.config import load_config
from llm_finetuning.pilot import _check_deadline, _encode_example


class Tokenizer:
    eos_token = "~"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return messages[0]["content"] + " assistant:"

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token) for token in ids)


def test_pilot_masks_prompt_and_keeps_answer_under_truncation():
    example = {"id": "a", "prompt": "evidence " * 100 + "question?", "answer": "42"}
    encoded = _encode_example(example, Tokenizer(), 96)
    boundary = len(encoded["prompt_ids"])
    assert len(encoded["input_ids"]) <= 96
    assert encoded["labels"][:boundary] == [-100] * boundary
    assert encoded["labels"][boundary:] == encoded["input_ids"][boundary:]
    assert "".join(map(chr, encoded["input_ids"][boundary:])) == "Final answer: 42~"
    assert "question?" in "".join(map(chr, encoded["prompt_ids"]))


def test_pilot_deadline_rejects_expired_budget():
    with pytest.raises(TimeoutError, match="budget exhausted"):
        _check_deadline(time.perf_counter() - 1)
    _check_deadline(time.perf_counter() + 60)


def test_pilot_preset_and_notebook_are_valid():
    config = load_config("configs/local-pilot.yaml").raw
    assert config["model"]["max_length"] == 192
    assert config["training"]["max_steps"] == 24
    assert config["model"]["name_or_path"] != load_config("configs/experiment.yaml").model_name
    notebook = json.loads(Path("notebooks/local_pilot.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "local_pilot.ipynb", "exec")


def test_evaluation_weights_answer_tokens_and_decodes_only_generated_tokens():
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from llm_finetuning.pilot import _evaluate

    class Model:
        def eval(self):
            return self

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def __call__(self, input_ids, labels):
            # Losses differ to make an unweighted average detectably wrong.
            count = int((labels[:, 1:] != -100).sum())
            return SimpleNamespace(loss=torch.tensor(float(count)))

        def generate(self, input_ids, **kwargs):
            return torch.cat([input_ids, torch.tensor([[52, 50, 126]])], dim=1)

    tokenizer = Tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id = 126
    examples = [
        {"id": "a", "answer": "42", "input_ids": [1, 2], "labels": [-100, 2], "prompt_ids": [1]},
        {
            "id": "b",
            "answer": "42",
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, 2, 3, 4],
            "prompt_ids": [1],
        },
    ]
    metrics = _evaluate(Model(), tokenizer, examples, 3, time.perf_counter() + 60)
    assert metrics["completion_loss"] == pytest.approx((1 + 9) / 4)
    assert metrics["answer_tokens"] == 4
    assert metrics["numeric_accuracy"] == 1
    assert [row["prediction"] for row in metrics["predictions"]] == ["42~", "42~"]
    assert not any(row["hit_token_limit"] for row in metrics["predictions"])
