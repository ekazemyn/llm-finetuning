import json
import time
from pathlib import Path

import pytest

from llm_finetuning.config import load_config
from llm_finetuning.pilot import _check_deadline, _encode_example


class Tokenizer:
    eos_token = "~"
    eos_token_id = 126

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return messages[0]["content"] + " assistant:"

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(
            chr(token)
            for token in ids
            if not (kwargs.get("skip_special_tokens") and token in {126, 127})
        )


def test_pilot_masks_prompt_and_keeps_answer_under_truncation():
    from llm_finetuning.data import normalize_example

    example = normalize_example(
        {
            "id": "a",
            "pre_text": ["unrelated background " * 100],
            "table": [["Revenue in 2024", "42"]],
            "qa": {"question": "What was revenue in 2024?", "answer": "42"},
        }
    )
    encoded = _encode_example(example, Tokenizer(), 320, 32)
    boundary = len(encoded["prompt_ids"])
    assert len(encoded["input_ids"]) <= 320
    assert encoded["labels"][:boundary] == [-100] * boundary
    assert encoded["labels"][boundary:] == encoded["input_ids"][boundary:]
    assert "".join(map(chr, encoded["input_ids"][boundary:])) == "Final answer: 42~"
    prompt = "".join(map(chr, encoded["prompt_ids"]))
    assert "What was revenue in 2024?" in prompt
    assert "Reply only with Final answer:" in prompt
    assert "Revenue in 2024 | 42" in prompt
    assert prompt.endswith(" assistant:")
    assert encoded["context"]["evidence_truncated"]
    assert encoded["context"]["evidence_omitted_tokens"] > 0
    # Reference length/value must never change evidence selection or generation input.
    other = _encode_example({**example, "answer": "987654"}, Tokenizer(), 320, 32)
    assert other["prompt_ids"] == encoded["prompt_ids"]
    assert other["context"] == encoded["context"]


def test_pilot_rejects_budget_that_would_truncate_question():
    from llm_finetuning.data import normalize_example

    example = normalize_example({"id": "a", "question": "Q?", "answer": "42"})
    with pytest.raises(ValueError, match="complete question"):
        _encode_example(example, Tokenizer(), 32, 16)


def test_pilot_retains_all_evidence_when_it_fits():
    from llm_finetuning.data import normalize_example

    example = normalize_example(
        {"id": "a", "question": "Revenue?", "answer": "42", "pre_text": "Revenue was 42."}
    )
    encoded = _encode_example(example, Tokenizer(), 320, 32)
    assert not encoded["context"]["evidence_truncated"]
    assert encoded["context"]["evidence_omitted_tokens"] == 0
    assert encoded["context"]["selected_line_indices"] == [0]


def test_pilot_deadline_rejects_expired_budget():
    with pytest.raises(TimeoutError, match="budget exhausted"):
        _check_deadline(time.perf_counter() - 1)
    _check_deadline(time.perf_counter() + 60)


def test_pilot_preset_and_notebook_are_valid():
    config = load_config("configs/local-pilot.yaml").raw
    assert config["model"]["max_length"] == 384
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
    assert [row["prediction"] for row in metrics["predictions"]] == ["42", "42"]
    assert not any(row["hit_token_limit"] for row in metrics["predictions"])


@pytest.mark.parametrize(
    "generated,reason",
    [
        ([52, 50, 126], "eos"),
        ([52, 50, 127], "eos"),
        ([52, 50, 51], "token_limit"),
    ],
)
def test_generation_stop_reason_includes_model_eos_tokens(generated, reason):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from llm_finetuning.pilot import _evaluate

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[126, 127])

        def eval(self):
            pass

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def __call__(self, **kwargs):
            return SimpleNamespace(loss=torch.tensor(1.0))

        def generate(self, input_ids, **kwargs):
            assert kwargs["eos_token_id"] == [126, 127]
            assert not kwargs["do_sample"]
            return torch.cat([input_ids, torch.tensor([generated])], dim=1)

    tokenizer = Tokenizer()
    tokenizer.pad_token_id = 126
    example = {
        "id": "a",
        "answer": "42",
        "input_ids": [1, 2],
        "labels": [-100, 2],
        "prompt_ids": [1],
    }
    metrics = _evaluate(Model(), tokenizer, [example], 3, time.perf_counter() + 60)
    row = metrics["predictions"][0]
    assert row["stop_reason"] == reason
    assert row["hit_token_limit"] == (reason == "token_limit")


def test_real_cached_tokenizer_preserves_chat_and_loss_boundary():
    transformers = pytest.importorskip("transformers")
    from llm_finetuning.data import normalize_example

    snapshot = Path("data/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/") / (
        "snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac"
    )
    if not (snapshot / "tokenizer.json").exists():
        pytest.skip("Optional real tokenizer check requires the local pilot cache; no downloads")
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    example = normalize_example(
        {
            "id": "a",
            "question": "Revenue in 2024?",
            "answer": "42",
            "pre_text": "Revenue was 42 in 2024. " * 100,
        }
    )
    encoded = _encode_example(example, tokenizer, 384, 64)
    prompt = tokenizer.decode(encoded["prompt_ids"])
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "Question: Revenue in 2024?" in prompt
    assert encoded["input_ids"][-1] == tokenizer.eos_token_id
    boundary = len(encoded["prompt_ids"])
    assert all(label == -100 for label in encoded["labels"][:boundary])
    assert tokenizer.decode(encoded["labels"][boundary:]).startswith("Final answer: 42")
    assert len(encoded["prompt_ids"]) + 64 <= 384
