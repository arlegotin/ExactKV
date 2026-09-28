"""Pinned, unmodified BF16 Qwen model identity and loading."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .records import JsonDict

MODEL_ID = "Qwen/Qwen3-0.6B"
EXPECTED = {
    "model_type": "qwen3",
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
}
ADVERTISED_CONTEXT_LIMIT = 32768
SNAPSHOT_PATTERNS = ("*.safetensors", "*.json", "*.model", "*.tiktoken", "*.jinja", "merges.txt", "vocab.txt")


class ModelIdentityError(ValueError):
    """The checkpoint/tokenizer is not the pinned original BF16 model."""


@dataclass(frozen=True, slots=True)
class ModelHandle:
    model: Any
    tokenizer: Any
    identity: JsonDict


def context_limit(config: JsonDict) -> int:
    positions = config.get("max_position_embeddings")
    if not isinstance(positions, int) or positions < 1:
        raise ModelIdentityError("missing positive max_position_embeddings")
    return min(ADVERTISED_CONTEXT_LIMIT, positions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_lock(lock: JsonDict) -> tuple[Path, JsonDict]:
    if lock.get("model_id") != MODEL_ID:
        raise ModelIdentityError("only the official Qwen/Qwen3-0.6B checkpoint is supported")
    for field in ("revision", "tokenizer_revision"):
        if not isinstance(lock.get(field), str) or re.fullmatch(r"[0-9a-f]{40}", lock[field]) is None:
            raise ModelIdentityError(f"{field} must be an immutable 40-hex commit")
    if lock["revision"] != lock["tokenizer_revision"]:
        raise ModelIdentityError("model and tokenizer revisions must match")
    if lock.get("trust_remote_code") is not False:
        raise ModelIdentityError("remote code is disabled for this model")
    if lock.get("original_dtype") != "bfloat16":
        raise ModelIdentityError("original BF16 weights are required")
    root = Path(lock.get("local_dir", ""))
    if not root.is_dir():
        raise ModelIdentityError(f"missing local model snapshot: {root}")
    hashes = lock.get("file_hashes")
    if not isinstance(hashes, dict) or not hashes or not any(name.endswith(".safetensors") for name in hashes):
        raise ModelIdentityError("snapshot requires recorded weight file hashes")
    for name, expected in hashes.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ModelIdentityError("invalid snapshot relative file path")
        path = root / name
        if not path.is_file() or _sha256(path) != expected:
            raise ModelIdentityError(f"snapshot file hash mismatch: {name}")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ModelIdentityError("missing config.json")
    config = json.loads(config_path.read_text())
    if any(config.get(key) != value for key, value in EXPECTED.items()):
        raise ModelIdentityError("Qwen model geometry or architecture does not match")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        raise ModelIdentityError("checkpoint config is not original BF16")
    if any(key in config for key in ("quantization", "quantization_config", "model_file")):
        raise ModelIdentityError("quantized weights or custom model code are unsupported")
    if lock.get("context_limit") != context_limit(config):
        raise ModelIdentityError("locked context limit does not match the conservative model limit")
    return root, config


def resolve_snapshot(model_id: str) -> JsonDict:
    if model_id != MODEL_ID:
        raise ModelIdentityError("only the official Qwen/Qwen3-0.6B checkpoint may be resolved")
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(model_id, revision="main", files_metadata=True)
    revision = info.sha
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ModelIdentityError("model hub did not return an immutable revision")
    expected_bytes = sum(
        (sibling.size or 0)
        for sibling in info.siblings or []
        if any(fnmatch.fnmatch(sibling.rfilename, pattern) for pattern in SNAPSHOT_PATTERNS)
    )
    free_bytes = shutil.disk_usage(Path.cwd()).free
    required_bytes = max(3 * 1024**3, 2 * expected_bytes)
    if free_bytes < required_bytes:
        raise ModelIdentityError(f"insufficient disk headroom: {free_bytes} free, {required_bytes} required")
    local_dir = Path("artifacts") / "models" / revision
    snapshot_download(
        model_id,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=list(SNAPSHOT_PATTERNS),
        max_workers=4,
    )
    config_path = local_dir / "config.json"
    if not config_path.is_file():
        raise ModelIdentityError("resolved snapshot has no config.json")
    config = json.loads(config_path.read_text())
    hashes = {
        str(path.relative_to(local_dir)): _sha256(path)
        for path in sorted(local_dir.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    }
    lock: JsonDict = {
        "schema_version": 1,
        "kind": "model-lock",
        "model_id": MODEL_ID,
        "revision": revision,
        "tokenizer_revision": revision,
        "local_dir": str(local_dir),
        "original_dtype": "bfloat16",
        "context_limit": context_limit(config),
        "trust_remote_code": False,
        "file_hashes": hashes,
    }
    _validate_lock(lock)
    return lock


def load_model(lock: JsonDict) -> ModelHandle:
    root, _ = _validate_lock(lock)
    from safetensors import safe_open

    for name in lock["file_hashes"]:
        if not name.endswith(".safetensors"):
            continue
        with safe_open(str(root / name), framework="np", device="cpu") as weight_file:
            for key in weight_file.keys():
                if weight_file.get_slice(key).get_dtype() != "BF16":
                    raise ModelIdentityError(f"weight {key} is not original BF16")
    from mlx_lm.utils import load

    model, tokenizer, loaded_config = load(
        str(root), tokenizer_config={"trust_remote_code": False}, lazy=False, return_config=True
    )
    if any(loaded_config.get(key) != value for key, value in EXPECTED.items()):
        raise ModelIdentityError("loaded Qwen geometry changed")
    return ModelHandle(model=model, tokenizer=tokenizer, identity=lock)


def smoke(handle: ModelHandle) -> JsonDict:
    import numpy as np
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    caches = [KVCache() for _ in range(EXPECTED["num_hidden_layers"])]
    tokens = mx.array([[1, 2]], dtype=mx.int32)
    logits = handle.model(tokens, cache=caches)
    mx.eval(logits, *(entry for cache in caches for entry in (cache.keys, cache.values)))
    valid_kv = [cache.state for cache in caches]
    expected_shape = (1, EXPECTED["num_key_value_heads"], 2, EXPECTED["head_dim"])
    if any(cache.offset != 2 or keys.shape != expected_shape or values.shape != expected_shape for cache, (keys, values) in zip(caches, valid_kv)):
        raise ModelIdentityError("post-RoPE cache geometry does not match the original model")
    if any(keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16 for keys, values in valid_kv):
        raise ModelIdentityError("post-RoPE K or actual V is not BF16")
    finite = bool(np.isfinite(np.array(logits[0, -1].astype(mx.float32))).all())
    return {
        "model_revision": handle.identity["revision"],
        "logits_shape": list(logits.shape),
        "layers": len(caches),
        "kv_shape": list(expected_shape),
        "kv_dtype": "bfloat16",
        "allocated_capacity_tokens": caches[0].keys.shape[2],
        "last_logit_finite": finite,
    }


def capture_prompt(handle: ModelHandle, prompt: JsonDict) -> Iterator[tuple[int, Any, Any]]:
    """Diagnostic whole-cache capture; it is not a final-system peak-memory path."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    ids = prompt["token_ids"]
    if not ids or len(ids) > handle.identity["context_limit"] or len(ids) != prompt["tokens"]:
        raise ModelIdentityError("prompt token extent is invalid for the pinned model")
    caches = [KVCache() for _ in range(EXPECTED["num_hidden_layers"])]
    hidden = handle.model.model(mx.array([ids], dtype=mx.int32), cache=caches)
    mx.eval(hidden, *(entry for cache in caches for entry in (cache.keys, cache.values)))
    del hidden
    try:
        for layer, cache in enumerate(caches):
            keys, values = cache.state
            expected_shape = (1, EXPECTED["num_key_value_heads"], len(ids), EXPECTED["head_dim"])
            if keys.shape != expected_shape or values.shape != expected_shape:
                raise ModelIdentityError(f"captured layer {layer} has unexpected K/V geometry")
            if keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16:
                raise ModelIdentityError(f"captured layer {layer} is not original BF16 K/V")
            yield layer, keys, values
            cache.keys = None
            cache.values = None
    finally:
        for cache in caches:
            cache.keys = None
            cache.values = None
