"""Exact BF16 integer ordering and explicit FP32-to-BF16 RNE conversion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Words = NDArray[np.uint16]


def order(words: Words) -> Words:
    bits = np.asarray(words, dtype=np.uint16)
    return np.where(bits & 0x8000, ~bits, bits ^ 0x8000).astype(np.uint16)


def unorder(keys: Words) -> Words:
    bits = np.asarray(keys, dtype=np.uint16)
    return np.where(bits & 0x8000, bits ^ 0x8000, ~bits).astype(np.uint16)


def rne_bf16(values: NDArray[np.float32]) -> Words:
    """Round finite FP32 values to BF16 with ties to even using integer bits."""
    floats = np.asarray(values, dtype=np.float32)
    bits = floats.view(np.uint32).astype(np.uint64)
    rounded = (bits + np.uint64(0x7FFF) + ((bits >> 16) & 1)) & np.uint64(0xFFFFFFFF)
    return (rounded >> 16).astype(np.uint16)


def word_to_f32(word: int) -> np.float32:
    return np.array([np.uint32(int(word) << 16)], dtype=np.uint32).view(np.float32)[0]


def bf16_is_finite(word: int) -> bool:
    return (int(word) & 0x7F80) != 0x7F80


def bf16_is_nonzero_subnormal(word: int) -> bool:
    return (int(word) & 0x7F80) == 0 and (int(word) & 0x007F) != 0
