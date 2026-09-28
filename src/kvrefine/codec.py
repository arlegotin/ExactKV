"""CPU oracle for quantizer-conditioned exact BF16 interval-rank pages."""

from __future__ import annotations

import struct

import numpy as np
from numpy.typing import NDArray

from .bits import bf16_is_finite, bf16_is_nonzero_subnormal, order, rne_bf16, unorder, word_to_f32
from .format import FormatError, PAGE_HEADER, PAGE_MAGIC, VERSION, page_nbytes
from .native import NativePage, Words

_WIDTH_LIMITS = np.array([(1 << width) - 1 for width in range(17)], dtype=np.int32)
_FP32_TINY = np.finfo(np.float32).tiny


def _fp32_ok(values: NDArray[np.float32]) -> NDArray[np.bool_]:
    return np.isfinite(values) & ~((values != 0) & (np.abs(values) < _FP32_TINY))


def _bf16_ok(words: NDArray[np.uint16]) -> NDArray[np.bool_]:
    return ((words & 0x7F80) != 0x7F80) & ~(((words & 0x7F80) == 0) & ((words & 0x007F) != 0))


def _as_f32(words: NDArray[np.uint16]) -> NDArray[np.float32]:
    return (words.astype(np.uint32) << 16).view(np.float32)


def _interval_arrays(
    q: NDArray[np.uint8], scale_words: NDArray[np.uint16], bias_words: NDArray[np.uint16], mode: int
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.uint8], NDArray[np.bool_]]:
    if mode not in (0, 1):
        raise ValueError("interval mode must be tight or wide")
    s = _as_f32(scale_words)
    bias = _as_f32(bias_words)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        product = (q.astype(np.float32) * s).astype(np.float32)
        prediction = (product + bias).astype(np.float32)
        radius = (np.float32(0.5 if mode == 0 else 1.0) * np.abs(s)).astype(np.float32)
        low_float = (prediction - radius).astype(np.float32)
        high_float = (prediction + radius).astype(np.float32)
    safe = (
        _bf16_ok(scale_words)
        & _bf16_ok(bias_words)
        & (s != 0)
        & _fp32_ok(product)
        & _fp32_ok(prediction)
        & _fp32_ok(radius)
        & _fp32_ok(low_float)
        & _fp32_ok(high_float)
        & (low_float <= high_float)
    )
    low_word = rne_bf16(low_float)
    high_word = rne_bf16(high_float)
    low_key = order(low_word).astype(np.int32)
    high_key = order(high_word).astype(np.int32)
    lower = np.maximum(0, low_key - 1)
    upper = np.minimum(65535, high_key + 1)
    lower_word = unorder(lower.astype(np.uint16))
    upper_word = unorder(upper.astype(np.uint16))
    safe &= _bf16_ok(low_word) & _bf16_ok(high_word) & _bf16_ok(lower_word) & _bf16_ok(upper_word)
    safe &= lower <= upper
    span = np.maximum(1, upper - lower + 1)
    widths = np.searchsorted(_WIDTH_LIMITS, span - 1, side="left").astype(np.uint8)
    return lower, upper, widths, safe


def interval(q: int, scale_word: int, bias_word: int, mode: int) -> tuple[int, int, int] | None:
    if not 0 <= q <= 15:
        raise ValueError("Q4 code must be 0..15")
    if mode not in (0, 1):
        raise ValueError("interval mode must be tight or wide")
    if not all(bf16_is_finite(word) and not bf16_is_nonzero_subnormal(word) for word in (scale_word, bias_word)):
        return None
    scale = word_to_f32(scale_word)
    bias = word_to_f32(bias_word)
    if scale == 0:
        return None
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        product = np.float32(np.float32(q) * scale)
        prediction = np.float32(product + bias)
        radius = np.float32(np.float32(0.5 if mode == 0 else 1.0) * np.float32(abs(scale)))
        low_float = np.float32(prediction - radius)
        high_float = np.float32(prediction + radius)
    values = (product, prediction, radius, low_float, high_float)
    if not all(np.isfinite(value) and not (value != 0 and abs(value) < _FP32_TINY) for value in values):
        return None
    if low_float > high_float:
        return None
    low_word, high_word = (int(rne_bf16(np.array([value], dtype=np.float32))[0]) for value in (low_float, high_float))
    lower = max(0, int(order(np.array([low_word], dtype=np.uint16))[0]) - 1)
    upper = min(65535, int(order(np.array([high_word], dtype=np.uint16))[0]) + 1)
    endpoint_words = (low_word, high_word, int(unorder(np.array([lower], dtype=np.uint16))[0]), int(unorder(np.array([upper], dtype=np.uint16))[0]))
    if lower > upper or not all(bf16_is_finite(word) and not bf16_is_nonzero_subnormal(word) for word in endpoint_words):
        return None
    return lower, upper, (upper - lower).bit_length()


def _validate_side(side: NativePage, valid_count: int) -> None:
    groups = valid_count // 64
    if side.valid_count != valid_count or len(side.q) != valid_count // 8:
        raise FormatError("native Q4 page extent does not match the source page")
    if len(side.scales) != groups or len(side.biases) != groups:
        raise FormatError("native scale/bias extent does not match the source page")
    if side.q.dtype != np.uint32 or side.scales.dtype != np.uint16 or side.biases.dtype != np.uint16:
        raise FormatError("native side-information dtype does not match the format")


def _codes(side: NativePage) -> NDArray[np.uint8]:
    return np.stack([(side.q >> (4 * shift)) & 0xF for shift in range(8)], axis=1).reshape(-1).astype(np.uint8)


def _bounds(side: NativePage, mode: int):
    return _interval_arrays(
        _codes(side),
        np.repeat(side.scales, 64),
        np.repeat(side.biases, 64),
        mode,
    )


def encode_page(words: Words, side: NativePage, tensor_id: int, page_id: int) -> bytes:
    source = np.asarray(words)
    valid_count = len(source)
    if source.dtype != np.uint16:
        raise FormatError("source words must be uint16 BF16 bit patterns")
    if valid_count < 64 or valid_count > 4096 or valid_count % 64:
        raise FormatError("source page must contain 64..4096 group-aligned values")
    if not 0 <= tensor_id < 2**32 or not 0 <= page_id < 2**32:
        raise FormatError("tensor/page identifier exceeds u32 range")
    _validate_side(side, valid_count)
    groups = valid_count // 64
    keys = order(source).astype(np.int32)
    source_safe = _bf16_ok(source)
    candidates = (_bounds(side, 0), _bounds(side, 1))
    modes: list[int] = []
    for group in range(groups):
        sl = slice(group * 64, (group + 1) * 64)
        best_mode, best_bits = 2, 64 * 16
        for mode in (0, 1):
            lower, upper, widths, safe = candidates[mode]
            if not bool(np.all(source_safe[sl] & safe[sl] & (keys[sl] >= lower[sl]) & (keys[sl] <= upper[sl]))):
                continue
            bits = int(np.sum(widths[sl], dtype=np.int32))
            if bits < best_bits:
                best_mode, best_bits = mode, bits
        modes.append(best_mode)
    mode_map = bytearray(((groups + 3) // 4 + 3) // 4 * 4)
    for group, mode in enumerate(modes):
        mode_map[group // 4] |= mode << (2 * (group % 4))
    restarts = (valid_count + 127) // 128
    offsets = [0]
    payload = bytearray()
    bit_counts = []
    for restart in range(restarts):
        start = restart * 128
        end = min(start + 128, valid_count)
        packed = 0
        bit_position = 0
        for index in range(start, end):
            mode = modes[index // 64]
            if mode == 2:
                rank, width = int(source[index]), 16
            else:
                lower, _upper, widths, _safe = candidates[mode]
                rank, width = int(keys[index] - lower[index]), int(widths[index])
            packed |= rank << bit_position
            bit_position += width
        bit_counts.append(bit_position)
        byte_count = ((bit_position + 31) // 32) * 4
        payload.extend(packed.to_bytes(byte_count, "little"))
        offsets.append(len(payload) // 4)
    total = page_nbytes(valid_count, bit_counts)
    if total != PAGE_HEADER.size + len(mode_map) + 4 * len(offsets) + len(payload) + 4:
        raise AssertionError("serialized page length differs from estimator")
    header = PAGE_HEADER.pack(PAGE_MAGIC, VERSION, PAGE_HEADER.size, tensor_id, page_id, valid_count, groups, restarts, total, len(payload) // 4, 0)
    return header + mode_map + struct.pack(f"<{len(offsets)}I", *offsets) + payload + b"\0" * 4


def decode_page(blob: bytes, side: NativePage, tensor_id: int, page_id: int) -> Words:
    if len(blob) < PAGE_HEADER.size:
        raise FormatError("truncated page header")
    (magic, version, header_size, stored_tensor, stored_page, count, groups, restarts, total, extent, reserved) = PAGE_HEADER.unpack_from(blob)
    if magic != PAGE_MAGIC or version != VERSION or header_size != PAGE_HEADER.size or reserved:
        raise FormatError("unsupported page magic/version or reserved field")
    if (stored_tensor, stored_page) != (tensor_id, page_id):
        raise FormatError("page tensor identity mismatch")
    if count < 64 or count > 4096 or count % 64 or groups != count // 64 or restarts != (count + 127) // 128:
        raise FormatError("invalid page group/restart count")
    _validate_side(side, count)
    mode_length = ((groups + 3) // 4 + 3) // 4 * 4
    offsets_start = PAGE_HEADER.size + mode_length
    payload_start = offsets_start + 4 * (restarts + 1)
    if total != len(blob) or total != payload_start + extent * 4 + 4:
        raise FormatError("page byte extent is inconsistent")
    if blob[-4:] != b"\0" * 4:
        raise FormatError("page guard is not zero")
    raw_modes = blob[PAGE_HEADER.size:offsets_start]
    modes = [(raw_modes[group // 4] >> (2 * (group % 4))) & 3 for group in range(groups)]
    if 3 in modes or (int.from_bytes(raw_modes, "little") >> (2 * groups)):
        raise FormatError("invalid mode map or nonzero padding")
    offsets = struct.unpack_from(f"<{restarts + 1}I", blob, offsets_start)
    if offsets[0] != 0 or offsets[-1] != extent:
        raise FormatError("invalid restart extent")
    bounds = (_bounds(side, 0), _bounds(side, 1))
    decoded = np.empty(count, dtype=np.uint16)
    interval_keys = np.empty(count, dtype=np.uint16)
    interval_positions = np.zeros(count, dtype=np.bool_)
    expected_word_offset = 0
    for restart in range(restarts):
        start = restart * 128
        end = min(start + 128, count)
        if offsets[restart] != expected_word_offset or offsets[restart + 1] < offsets[restart]:
            raise FormatError("restart offsets are inconsistent")
        chunk = blob[payload_start + offsets[restart] * 4:payload_start + offsets[restart + 1] * 4]
        packed = int.from_bytes(chunk, "little")
        bit_position = 0
        for index in range(start, end):
            mode = modes[index // 64]
            if mode == 2:
                width = 16
            else:
                lower, upper, widths, safe = bounds[mode]
                if not bool(safe[index]):
                    raise FormatError("interval metadata is unsupported")
                width = int(widths[index])
            rank = (packed >> bit_position) & ((1 << width) - 1)
            bit_position += width
            if mode == 2:
                decoded[index] = rank
            else:
                if rank >= int(upper[index] - lower[index] + 1):
                    raise FormatError("rank is outside the predicted interval")
                interval_keys[index] = lower[index] + rank
                interval_positions[index] = True
        expected_word_offset += (bit_position + 31) // 32
        if offsets[restart + 1] != expected_word_offset or packed >> bit_position:
            raise FormatError("restart payload extent or padding is invalid")
    decoded[interval_positions] = unorder(interval_keys[interval_positions])
    return decoded
