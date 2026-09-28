from __future__ import annotations

import hashlib
import json

import pytest


def _lock(tmp_path):
    config = {
        "model_type": "qwen3",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"fixture")
    return {
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "local_dir": str(tmp_path),
        "original_dtype": "bfloat16",
        "context_limit": 32768,
        "trust_remote_code": False,
        "file_hashes": {
            "config.json": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model.safetensors": hashlib.sha256(weight.read_bytes()).hexdigest(),
        },
    }


def test_runtime_rejects_mutable_revision_before_loading(tmp_path):
    from kvrefine.model import ModelIdentityError, load_model

    lock = _lock(tmp_path)
    lock["revision"] = "main"
    with pytest.raises(ModelIdentityError, match="immutable"):
        load_model(lock)


def test_context_cap_is_conservative():
    from kvrefine.model import context_limit

    assert context_limit({"max_position_embeddings": 40960}) == 32768


@pytest.mark.parametrize(
    "changed_config",
    [
        {"num_hidden_layers": 27},
        {"quantization_config": {"bits": 4}},
        {"model_file": "custom.py"},
    ],
)
def test_wrong_geometry_quantization_or_custom_model_is_rejected(tmp_path, changed_config):
    from kvrefine.model import ModelIdentityError, load_model

    lock = _lock(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config.update(changed_config)
    path.write_text(json.dumps(config))
    lock["file_hashes"]["config.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ModelIdentityError):
        load_model(lock)


def test_remote_tokenizer_code_is_rejected(tmp_path):
    from kvrefine.model import ModelIdentityError, load_model

    lock = _lock(tmp_path)
    lock["trust_remote_code"] = True
    with pytest.raises(ModelIdentityError, match="remote code"):
        load_model(lock)


def test_file_hash_mismatch_is_rejected_before_loading(tmp_path):
    from kvrefine.model import ModelIdentityError, load_model

    lock = _lock(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ModelIdentityError, match="hash"):
        load_model(lock)


def test_resolution_rejects_any_other_repo_without_network():
    from kvrefine.model import ModelIdentityError, resolve_snapshot

    with pytest.raises(ModelIdentityError, match="official"):
        resolve_snapshot("someone/other-model")


@pytest.mark.model
def test_pinned_qwen_smoke_has_original_post_rope_bf16_kv():
    from pathlib import Path

    from kvrefine.model import load_model, smoke

    lock = json.loads(Path("data/model-lock.json").read_text())
    result = smoke(load_model(lock))
    assert result["logits_shape"] == [1, 2, 151936]
    assert result["layers"] == 28
    assert result["kv_shape"] == [1, 8, 2, 128]
    assert result["kv_dtype"] == "bfloat16"
    assert result["last_logit_finite"] is True
