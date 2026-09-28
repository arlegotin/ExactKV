"""Version-one ExactKV page/container framing and size arithmetic."""

from __future__ import annotations

import json
import struct
from typing import Any

PAGE_MAGIC = b"EKVR"
CONTAINER_MAGIC = b"EKVT"
VERSION = 1
PAGE_HEADER = struct.Struct("<4sHHIIHBBIII")
CONTAINER_HEADER = struct.Struct("<4sIQ")


class FormatError(ValueError):
    """An ExactKV stream is malformed or bound to incompatible side information."""


def _round_up(value: int, unit: int) -> int:
    return (value + unit - 1) // unit * unit


def page_nbytes(valid_count: int, restart_payload_bits: list[int]) -> int:
    if not isinstance(valid_count, int) or valid_count < 64 or valid_count > 4096 or valid_count % 64:
        raise ValueError("page valid count must be 64..4096 and group-aligned")
    restarts = (valid_count + 127) // 128
    if len(restart_payload_bits) != restarts or any(not isinstance(bits, int) or bits < 0 for bits in restart_payload_bits):
        raise ValueError("restart bit counts do not match the valid page")
    groups = valid_count // 64
    mode_bytes = _round_up((groups + 3) // 4, 4)
    offset_bytes = 4 * (restarts + 1)
    payload_bytes = sum(_round_up(bits, 32) // 8 for bits in restart_payload_bits)
    return PAGE_HEADER.size + mode_bytes + offset_bytes + payload_bytes + 4


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pack_container(manifest: dict[str, Any], pages: list[bytes]) -> bytes:
    if "page_offsets" in manifest:
        raise FormatError("caller must not supply page offsets")
    offsets = [0]
    arena = bytearray()
    for page in pages:
        if len(page) < PAGE_HEADER.size or len(page) % 4 or page[:4] != PAGE_MAGIC:
            raise FormatError("container page must be aligned EKVR bytes")
        arena.extend(page)
        offsets.append(len(arena))
    bound_manifest = {**manifest, "page_offsets": offsets}
    manifest_bytes = _canonical(bound_manifest)
    padding = b"\0" * (_round_up(CONTAINER_HEADER.size + len(manifest_bytes), 8) - CONTAINER_HEADER.size - len(manifest_bytes))
    return CONTAINER_HEADER.pack(CONTAINER_MAGIC, VERSION, len(manifest_bytes)) + manifest_bytes + padding + arena


def unpack_container(blob: bytes) -> tuple[dict[str, Any], list[memoryview]]:
    if len(blob) < CONTAINER_HEADER.size:
        raise FormatError("truncated container header")
    magic, version, manifest_length = CONTAINER_HEADER.unpack_from(blob)
    if magic != CONTAINER_MAGIC or version != VERSION:
        raise FormatError("unsupported container magic or version")
    manifest_end = CONTAINER_HEADER.size + manifest_length
    arena_start = _round_up(manifest_end, 8)
    if arena_start > len(blob) or any(blob[manifest_end:arena_start]):
        raise FormatError("truncated or nonzero container padding")
    try:
        manifest = json.loads(blob[CONTAINER_HEADER.size:manifest_end])
    except (ValueError, UnicodeDecodeError) as exc:
        raise FormatError("invalid container manifest JSON") from exc
    if not isinstance(manifest, dict) or _canonical(manifest) != blob[CONTAINER_HEADER.size:manifest_end]:
        raise FormatError("container manifest is not canonical JSON")
    offsets = manifest.get("page_offsets")
    arena = memoryview(blob)[arena_start:]
    if not isinstance(offsets, list) or not offsets or offsets[0] != 0 or offsets[-1] != len(arena):
        raise FormatError("container page offset extent is invalid")
    if any(not isinstance(offset, int) or offset < 0 or offset % 4 for offset in offsets):
        raise FormatError("container page offsets are not aligned integers")
    if any(left >= right for left, right in zip(offsets, offsets[1:])):
        raise FormatError("container page offsets do not increase")
    pages = [arena[left:right] for left, right in zip(offsets, offsets[1:])]
    if any(len(page) < PAGE_HEADER.size or bytes(page[:4]) != PAGE_MAGIC for page in pages):
        raise FormatError("container holds an invalid page")
    return manifest, pages
