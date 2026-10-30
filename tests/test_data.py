import hashlib
import json
from pathlib import Path

import pytest

from llm_finetuning.config import load_config
from llm_finetuning.data import (
    _cache_metadata,
    _source_cache_directory,
    _source_files_and_checksums,
    normalize_example,
)
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


def test_source_cache_verifies_raw_split_checksums(tmp_path):
    config = load_config(Path("configs/experiment.yaml")).raw
    config["data"]["prepared_dir"] = str(tmp_path / "finqa-prepared")
    cache_directory = _source_cache_directory(config)
    cache_directory.mkdir(parents=True)
    source_splits = {
        "train": config["data"]["train_split"],
        "validation": config["data"]["validation_split"],
        "test": config["data"]["test_split"],
    }
    checksums = {}
    for source_name in source_splits.values():
        content = f"{source_name} split fixture".encode()
        (cache_directory / f"{source_name}.json").write_bytes(content)
        checksums[source_name] = hashlib.sha256(content).hexdigest()
    (cache_directory / "source_manifest.json").write_text(
        json.dumps(
            {
                "source_repository": config["data"]["source_repository"],
                "dataset_revision": config["data"]["dataset_revision"],
                "source_splits": source_splits,
                "sha256": checksums,
            }
        ),
        encoding="utf-8",
    )

    source_files, verified_checksums = _source_files_and_checksums(config)
    assert verified_checksums == checksums
    assert set(source_files) == {"train", "dev", "test"}

    (cache_directory / "dev.json").write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch for dev"):
        _source_files_and_checksums(config)


def test_warmup_ratio_maps_to_transformers_ratio_form():
    assert _warmup_steps({"warmup_ratio": 0.03}) == 0.03
    with pytest.raises(ValueError, match=r"range \[0, 1\)"):
        _warmup_steps({"warmup_ratio": 1.0})


@pytest.fixture
def source_config(tmp_path):
    config = load_config(Path("configs/experiment.yaml")).raw
    config["data"]["prepared_dir"] = str(tmp_path / "prepared")
    return config


@pytest.fixture
def mock_source_downloads(monkeypatch):
    import io

    calls = []

    def open_source(request, timeout):
        assert timeout == 60
        calls.append(request.full_url)
        return io.BytesIO(b'[{"id": "fixture", "qa": {"question": "Q?", "answer": "5"}}]')

    monkeypatch.setattr("llm_finetuning.data.urlopen", open_source)
    return calls


def test_downloads_pinned_sources_and_reuses_verified_cache(source_config, mock_source_downloads):
    files, checksums = _source_files_and_checksums(source_config)
    revision = source_config["data"]["dataset_revision"]
    repository = source_config["data"]["source_repository"]
    assert mock_source_downloads == [
        f"https://raw.githubusercontent.com/{repository}/{revision}/dataset/{name}.json"
        for name in ("train", "dev", "test")
    ]
    assert checksums == {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
    }
    assert _source_files_and_checksums(source_config) == (files, checksums)
    assert len(mock_source_downloads) == 3


@pytest.mark.parametrize("damage", ["missing_file", "missing_digest", "wrong_repository"])
def test_source_cache_rejects_invalid_cache(source_config, mock_source_downloads, damage):
    files, _ = _source_files_and_checksums(source_config)
    manifest_path = _source_cache_directory(source_config) / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if damage == "missing_file":
        files["dev"].unlink()
    elif damage == "missing_digest":
        del manifest["sha256"]["dev"]
    else:
        manifest["source_repository"] = "other/repository"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        _source_files_and_checksums(source_config)
    assert len(mock_source_downloads) == 3


def test_failed_download_does_not_leave_partial_file(tmp_path, monkeypatch):
    from llm_finetuning.data import _download_source_file

    def fail_download(*args, **kwargs):
        raise OSError("interrupted download")

    monkeypatch.setattr("llm_finetuning.data.urlopen", fail_download)
    destination = tmp_path / "train.json"
    destination.write_bytes(b"existing source")
    with pytest.raises(OSError, match="interrupted download"):
        _download_source_file("https://example.com/train.json", destination)
    assert destination.read_bytes() == b"existing source"
    assert list(tmp_path.iterdir()) == [destination]


def test_prepare_records_checksums_and_verifies_cached_sources(
    source_config, mock_source_downloads, monkeypatch
):
    pytest.importorskip("datasets")
    from llm_finetuning.data import load_finqa, prepare_finqa

    monkeypatch.setattr("llm_finetuning.data.build_provenance", lambda config: {})
    destination = Path(source_config["data"]["prepared_dir"])
    destination.mkdir()
    prepare_finqa(source_config)
    manifest = json.loads((destination / "data_manifest.json").read_text(encoding="utf-8"))
    _, checksums = _source_files_and_checksums(source_config)
    assert manifest["source_sha256"] == checksums
    assert set(load_finqa(source_config)) == {"train", "validation", "test"}
    (_source_cache_directory(source_config) / "dev.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch for dev"):
        load_finqa(source_config)
