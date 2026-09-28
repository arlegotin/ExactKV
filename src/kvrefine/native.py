"""Bounded, bit-preserving access to the installed native MLX Q4 ABI."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .records import JsonDict

Words = NDArray[np.uint16]
QWords = NDArray[np.uint32]


@dataclass(frozen=True, slots=True)
class NativePage:
    q: QWords
    scales: Words
    biases: Words
    valid_count: int
    abi_id: str
    identity: str


@dataclass(frozen=True, slots=True)
class NativeKV:
    keys: tuple[Any, Any, Any]
    values: tuple[Any, Any, Any]
    valid_tokens: int
    capacity_tokens: int
    abi_id: str


def _hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_abi() -> JsonDict:
    import mlx_lm.models.base as base
    import mlx_lm.models.cache as cache
    import mlx_lm.models.qwen3 as qwen3

    return {
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "cache_source_sha256": _hash(cache.__file__),
        "attention_source_sha256": _hash(base.__file__),
        "qwen3_source_sha256": _hash(qwen3.__file__),
        "mode": "affine",
        "bits": 4,
        "group_size": 64,
        "code_dtype": "uint32",
        "metadata_dtype": "bfloat16",
        "nibble_order": "least-significant-first",
        "token_growth": cache.QuantizedKVCache.step,
    }


def _abi_id() -> str:
    return hashlib.sha256(json.dumps(native_abi(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def words_bf16(words: Words) -> Any:
    import mlx.core as mx

    array = np.asarray(words)
    if array.dtype != np.uint16:
        raise TypeError("BF16 words must have uint16 dtype")
    return mx.array(np.ascontiguousarray(array), dtype=mx.uint16).view(mx.bfloat16)


def bf16_words(x: Any) -> Words:
    import mlx.core as mx

    if x.dtype != mx.bfloat16:
        raise TypeError("expected an MLX BF16 array")
    bits = mx.contiguous(x).view(mx.uint16)
    mx.eval(bits)
    return np.array(bits, dtype=np.uint16, copy=True)


def quantize_kv(k: Any, v: Any, capacity_tokens: int) -> NativeKV:
    import mlx.core as mx
    from mlx_lm.models.cache import QuantizedKVCache

    if k.dtype != mx.bfloat16 or v.dtype != mx.bfloat16:
        raise TypeError("native Q4 requires original BF16 K and V")
    if k.ndim != 4 or v.ndim != 4 or k.shape[:3] != v.shape[:3] or k.shape[0] != 1:
        raise ValueError("expected batch-one K/V with matching heads and tokens")
    if k.shape[-1] % 64 or v.shape[-1] % 64:
        raise ValueError("K/V head dimensions must be divisible by group size 64")
    if capacity_tokens < k.shape[2] or capacity_tokens < 1:
        raise ValueError("capacity must contain the valid tokens")
    cache = QuantizedKVCache(group_size=64, bits=4)
    cache.step = ((capacity_tokens + 255) // 256) * 256
    cache.update_and_fetch(k, v)
    mx.eval(*cache.keys, *cache.values)
    capacity = cache.keys[0].shape[2]
    if cache.keys[0].dtype != mx.uint32:
        raise RuntimeError("installed MLX Q4 packing is not uint32")
    if any(x.dtype != mx.bfloat16 for x in (*cache.keys[1:], *cache.values[1:])):
        raise RuntimeError("installed MLX Q4 metadata is not BF16")
    return NativeKV(cache.keys, cache.values, cache.offset, capacity, _abi_id())


def export_page(
    native: NativeKV,
    role: str,
    head: int,
    start_token: int,
    tokens: int,
    identity: str,
) -> NativePage:
    import mlx.core as mx

    if role not in {"K", "V"}:
        raise ValueError("role must be K or V")
    tensors = native.keys if role == "K" else native.values
    if head < 0 or head >= tensors[0].shape[1]:
        raise ValueError("head is outside native tuple")
    if start_token < 0 or tokens < 1 or start_token + tokens > native.valid_tokens:
        raise ValueError("page is outside valid native tokens")
    rows = tuple(mx.contiguous(x[0, head, start_token : start_token + tokens]) for x in tensors)
    mx.eval(*rows)
    q = np.array(rows[0], dtype=np.uint32, copy=True).ravel()
    scales = np.array(rows[1].view(mx.uint16), dtype=np.uint16, copy=True).ravel()
    biases = np.array(rows[2].view(mx.uint16), dtype=np.uint16, copy=True).ravel()
    values_per_token = tensors[0].shape[-1] * 8
    return NativePage(q, scales, biases, tokens * values_per_token, native.abi_id, identity)
