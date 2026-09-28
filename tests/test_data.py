from __future__ import annotations

import hashlib
import json

import pytest


def test_sources_are_disjoint():
    from kvrefine.data import DataError, validate_disjoint_sources

    good = [
        {"split": "dev", "domain": "prose", "source_ids": ["train:article-1"]},
        {"split": "heldout", "domain": "prose", "source_ids": ["validation:article-1"]},
        {"split": "dev", "domain": "code", "source_ids": ["asyncio/base_events.py"]},
        {"split": "heldout", "domain": "code", "source_ids": ["email/message.py"]},
    ]
    validate_disjoint_sources(good)
    with pytest.raises(DataError, match="overlap"):
        validate_disjoint_sources(good + [{"split": "heldout", "domain": "code", "source_ids": ["asyncio/tasks.py"]}])
    with pytest.raises(DataError, match="overlap"):
        validate_disjoint_sources(good + [{"split": "heldout", "domain": "prose", "source_ids": ["train:article-1"]}])


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and enable_thinking is False
        return [101, 102] + [ord(char) for char in messages[0]["content"]] + [103, 104]


def test_template_tokens_count_includes_suffix_and_bos():
    from kvrefine.data import fit_source_to_tokens

    ids, used = fit_source_to_tokens(_FakeTokenizer(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4, 80)
    assert len(ids) == 80
    assert ids[:2] == [101, 102]
    assert ids[-2:] == [103, 104]
    assert used


def test_interrupted_manifest_keeps_previous_file(tmp_path, monkeypatch):
    from kvrefine import data

    out = tmp_path / "manifest.jsonl"
    out.write_bytes(b"old manifest\n")
    old_hash = hashlib.sha256(out.read_bytes()).hexdigest()

    def fail_replace(_source, _target):
        raise OSError("simulated interruption")

    monkeypatch.setattr(data, "_atomic_replace", fail_replace)
    with pytest.raises(OSError, match="interruption"):
        data.write_manifest(out, [{"split": "dev", "tokens": 512}])
    assert hashlib.sha256(out.read_bytes()).hexdigest() == old_hash


def test_changed_tokenizer_is_rejected(tmp_path):
    from kvrefine.data import DataError, load_prompts

    tokens = [1, 2, 3]
    artifact = tmp_path / "tokens.json"
    artifact.write_text(json.dumps(tokens))
    payload = {
        "schema_version": 1,
        "kind": "prompt",
        "split": "dev",
        "domain": "code",
        "tokens": 3,
        "token_artifact": str(artifact),
        "token_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "tokenizer_revision": "a" * 40,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(payload) + "\n")
    with pytest.raises(DataError, match="tokenizer"):
        load_prompts(manifest, "dev", 3, model_lock={"tokenizer_revision": "b" * 40})


def test_changed_prompt_text_is_rejected(tmp_path):
    from kvrefine.data import DataError, load_prompts

    tokens = tmp_path / "tokens.json"
    tokens.write_text("[1,2,3]\n")
    text = tmp_path / "text.txt"
    text.write_text("original")
    record = {
        "schema_version": 1,
        "kind": "prompt",
        "split": "dev",
        "domain": "code",
        "tokens": 3,
        "token_artifact": str(tokens),
        "token_sha256": hashlib.sha256(tokens.read_bytes()).hexdigest(),
        "text_artifact": str(text),
        "text_sha256": hashlib.sha256(text.read_bytes()).hexdigest(),
        "tokenizer_revision": "a" * 40,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    text.write_text("changed")
    with pytest.raises(DataError, match="text artifact"):
        load_prompts(manifest, "dev", 3, model_lock={"tokenizer_revision": "a" * 40})
