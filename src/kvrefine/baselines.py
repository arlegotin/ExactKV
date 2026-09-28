"""Independent exact BF16 page controls with fully framed byte costs."""

from __future__ import annotations

import hashlib
import struct

import numpy as np
from numpy.typing import NDArray
import zstandard as zstd

from .native import Words

_PALETTE = struct.Struct("<4sHHHBBI")  # magic, version, count, palette count, width, mode, payload bytes
_ZSTD = struct.Struct("<4sHHII")  # magic, version, count, exponent frame, sf frame
_XOR = struct.Struct("<4sHHI32s")  # magic, version, count, frame bytes, exact predictor digest
_VERSION = 1


class BaselineError(ValueError):
    """An independent exact baseline page is malformed or mismatched."""


def _words(values: Words) -> Words:
    array = np.asarray(values)
    if array.dtype != np.uint16 or array.ndim != 1 or len(array) < 64 or len(array) > 4096 or len(array) % 64:
        raise BaselineError("baseline page must be a one-dimensional, group-aligned uint16 array")
    return np.ascontiguousarray(array)


def _fields(words: Words) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    sf = (((words >> 8) & 0x80) | (words & 0x7F)).astype(np.uint8)
    exponent = ((words >> 7) & 0xFF).astype(np.uint8)
    return sf, exponent


def _round4(size: int) -> int:
    return (size + 3) // 4 * 4


def palette_encode(words: Words) -> bytes:
    source = _words(words)
    sf, exponent = _fields(source)
    palette = np.unique(exponent)
    width = (len(palette) - 1).bit_length()
    indexes = np.searchsorted(palette, exponent)
    packed = 0
    if width:
        for index, value in enumerate(indexes):
            packed |= int(value) << (index * width)
    index_bytes = packed.to_bytes((len(source) * width + 7) // 8, "little")
    candidate = palette.tobytes() + sf.tobytes() + index_bytes
    candidate += b"\0" * (_round4(len(candidate)) - len(candidate))
    raw = source.astype("<u2", copy=False).tobytes()
    if len(candidate) >= len(raw):
        return _PALETTE.pack(b"EKVP", _VERSION, len(source), 0, 0, 1, len(raw)) + raw + b"\0" * 4
    return _PALETTE.pack(b"EKVP", _VERSION, len(source), len(palette), width, 0, len(candidate)) + candidate + b"\0" * 4


def palette_decode(blob: bytes) -> Words:
    if len(blob) < _PALETTE.size + 4:
        raise BaselineError("truncated palette page")
    magic, version, count, palette_size, width, mode, extent = _PALETTE.unpack_from(blob)
    if magic != b"EKVP" or version != _VERSION or count < 64 or count > 4096 or count % 64:
        raise BaselineError("invalid palette page header")
    if mode not in (0, 1) or len(blob) != _PALETTE.size + extent + 4 or blob[-4:] != b"\0" * 4:
        raise BaselineError("palette page extent or guard is invalid")
    payload = blob[_PALETTE.size:-4]
    if mode == 1:
        if palette_size or width or extent != count * 2:
            raise BaselineError("invalid raw palette fallback")
        return np.frombuffer(payload, dtype="<u2").astype(np.uint16, copy=True)
    if not 1 <= palette_size <= 256 or width != (palette_size - 1).bit_length():
        raise BaselineError("invalid palette index width")
    index_bytes = (count * width + 7) // 8
    used = palette_size + count + index_bytes
    if extent != _round4(used) or any(payload[used:]):
        raise BaselineError("palette payload or padding is invalid")
    palette = np.frombuffer(payload[:palette_size], dtype=np.uint8)
    if np.any(palette[1:] <= palette[:-1]):
        raise BaselineError("palette exponents are not distinct and sorted")
    sf = np.frombuffer(payload[palette_size:palette_size + count], dtype=np.uint8)
    packed = int.from_bytes(payload[palette_size + count:used], "little")
    if packed >> (count * width):
        raise BaselineError("nonzero palette index padding")
    indexes = np.zeros(count, dtype=np.uint16)
    if width:
        mask = (1 << width) - 1
        for index in range(count):
            value = (packed >> (index * width)) & mask
            if value >= palette_size:
                raise BaselineError("palette index outside table")
            indexes[index] = value
    exponents = palette[indexes].astype(np.uint16)
    return (((sf.astype(np.uint16) & 0x80) << 8) | (exponents << 7) | (sf.astype(np.uint16) & 0x7F)).astype(np.uint16)


def field_zstd_encode(words: Words, level: int = 3) -> bytes:
    source = _words(words)
    sf, exponent = _fields(source)
    compressor = zstd.ZstdCompressor(level=level)
    exponent_frame = compressor.compress(exponent.tobytes())
    sf_frame = compressor.compress(sf.tobytes())
    return _ZSTD.pack(b"EKVZ", _VERSION, len(source), len(exponent_frame), len(sf_frame)) + exponent_frame + sf_frame


def field_zstd_decode(blob: bytes) -> Words:
    if len(blob) < _ZSTD.size:
        raise BaselineError("truncated field-Zstd page")
    magic, version, count, exponent_length, sf_length = _ZSTD.unpack_from(blob)
    if magic != b"EKVZ" or version != _VERSION or count < 64 or count > 4096 or count % 64:
        raise BaselineError("invalid field-Zstd page header")
    if len(blob) != _ZSTD.size + exponent_length + sf_length:
        raise BaselineError("field-Zstd frame extent is invalid")
    decoder = zstd.ZstdDecompressor()
    try:
        exponent = decoder.decompress(blob[_ZSTD.size:_ZSTD.size + exponent_length], max_output_size=count)
        sf = decoder.decompress(blob[_ZSTD.size + exponent_length:], max_output_size=count)
    except zstd.ZstdError as exc:
        raise BaselineError("invalid field-Zstd frame") from exc
    if len(exponent) != count or len(sf) != count:
        raise BaselineError("field-Zstd decoded length is wrong")
    exponents = np.frombuffer(exponent, dtype=np.uint8).astype(np.uint16)
    sf_words = np.frombuffer(sf, dtype=np.uint8).astype(np.uint16)
    return (((sf_words & 0x80) << 8) | (exponents << 7) | (sf_words & 0x7F)).astype(np.uint16)


def xor_zstd_encode(words: Words, predictor: Words) -> bytes:
    source, predicted = _words(words), _words(predictor)
    if len(source) != len(predicted):
        raise BaselineError("XOR predictor length mismatch")
    predictor_bytes = predicted.astype("<u2", copy=False).tobytes()
    correction = np.bitwise_xor(source, predicted).astype("<u2", copy=False).tobytes()
    frame = zstd.ZstdCompressor(level=3).compress(correction)
    return _XOR.pack(b"EKVX", _VERSION, len(source), len(frame), hashlib.sha256(predictor_bytes).digest()) + frame


def xor_zstd_decode(blob: bytes, predictor: Words) -> Words:
    predicted = _words(predictor)
    if len(blob) < _XOR.size:
        raise BaselineError("truncated XOR page")
    magic, version, count, extent, digest = _XOR.unpack_from(blob)
    if magic != b"EKVX" or version != _VERSION or len(predicted) != count or len(blob) != _XOR.size + extent:
        raise BaselineError("invalid XOR page header or length")
    if hashlib.sha256(predicted.astype("<u2", copy=False).tobytes()).digest() != digest:
        raise BaselineError("XOR predictor identity mismatch")
    try:
        correction = zstd.ZstdDecompressor().decompress(blob[_XOR.size:], max_output_size=count * 2)
    except zstd.ZstdError as exc:
        raise BaselineError("invalid XOR-Zstd frame") from exc
    if len(correction) != count * 2:
        raise BaselineError("XOR correction length mismatch")
    return np.bitwise_xor(np.frombuffer(correction, dtype="<u2"), predicted).astype(np.uint16)
