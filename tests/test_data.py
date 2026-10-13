import pytest

from llm_finetuning.data import _cache_metadata, normalize_example
from llm_finetuning.training import _chat_text, _warmup_steps


def test_normalize_finqa_nested_record():
    normalized = normalize_example(
        {
            "id": "example-1",
            "pre_text": ["Revenue increased."],
            "table": [["Year", "Revenue"], ["2024", "120"]],
            "post_text": "",
            "qa": {
                "question": "What was revenue in 2024?",
                "answer": "120",
                "rationale": "Read the 2024 row.",
            },
        }
    )

    assert normalized["id"] == "example-1"
    assert "Revenue increased." in normalized["prompt"]
    assert "2024 | 120" in normalized["prompt"]
    assert normalized["completion"].endswith("Final answer: 120")


def test_normalize_flat_record_and_missing_fields():
    assert normalize_example({"question": "Q?", "answer": "5"})["answer"] == "5"
    with pytest.raises(ValueError, match="missing a question or answer"):
        normalize_example({"question": "Q?"})


def test_normalize_uses_final_result_when_answer_is_empty():
    normalized = normalize_example(
        {
            "qa": {
                "question": "Q?",
                "answer": "",
                "program": "divide(10, 2)",
                "steps": [{"res": "5"}],
            }
        }
    )
    assert normalized["answer"] == "5"
    assert "Reference program: divide(10, 2)" in normalized["completion"]


class CharacterTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        suffix = "assistant:" if add_generation_prompt else ""
        return f"user:{messages[0]['content']}\n{suffix}"

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens, clean_up_tokenization_spaces):
        return "".join(chr(token_id) for token_id in token_ids)


def test_training_formatter_preserves_question_and_final_answer():
    example = {
        "prompt": f"Evidence: {'long evidence ' * 30} Question: preserve this question?",
        "completion": f"{'reasoning ' * 30}Final answer: 42",
    }
    formatted = _chat_text(example, CharacterTokenizer(), max_length=90)

    assert formatted["prompt"].endswith("Question: preserve this question?\nassistant:")
    assert formatted["completion"].endswith("Final answer: 42<eos>")


def test_prepared_cache_metadata_changes_with_sample_limit():
    config = {
        "project": {"seed": 42},
        "data": {
            "dataset_name": "czyssrs/FinQA",
            "dataset_revision": "a" * 40,
            "train_split": "train",
            "validation_split": "dev",
            "test_split": "test",
            "max_train_samples": None,
            "max_eval_samples": None,
            "max_test_samples": None,
        },
    }
    initial = _cache_metadata(config)
    config["data"]["max_eval_samples"] = 100

    assert _cache_metadata(config)["sample_limits"] != initial["sample_limits"]


def test_warmup_ratio_maps_to_transformers_ratio_form():
    assert _warmup_steps({"warmup_ratio": 0.03}) == 0.03
    with pytest.raises(ValueError, match=r"range \[0, 1\)"):
        _warmup_steps({"warmup_ratio": 1.0})
