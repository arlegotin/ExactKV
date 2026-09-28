"""Minimal versioned records and atomic artifact writes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

JsonDict = dict[str, Any]
SCHEMA_VERSION = 1


class RecordError(ValueError):
    """A result record is incomplete or from another schema."""


def validate_record(kind: str, record: JsonDict) -> None:
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise RecordError("unsupported or missing record schema_version")
    if record.get("kind") != kind:
        raise RecordError(f"expected {kind!r} record")
    if kind == "gate":
        required = {"name", "status", "evidence", "reason", "allowed_next", "producing_commit"}
    elif kind == "environment":
        required = {"ram_bytes", "python", "architecture", "unavailable"}
    else:
        required = set()
    missing = required - record.keys()
    if missing:
        raise RecordError(f"missing {kind} fields: {', '.join(sorted(missing))}")


def write_record(path: Path, record: JsonDict) -> str:
    kind = record.get("kind")
    if not isinstance(kind, str):
        raise RecordError("record kind must be a string")
    validate_record(kind, record)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()
