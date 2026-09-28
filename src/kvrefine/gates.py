"""Evidence-backed decisions between research stages."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .records import JsonDict, RecordError, validate_record, write_record

VALID_STATUSES = {"pass", "narrow", "stop", "blocked"}


class GateError(RuntimeError):
    """A prerequisite or its supporting evidence is unavailable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_decision(
    name: str,
    status: str,
    evidence: list[Path],
    reason: str,
    allowed_next: list[str],
    out: Path,
) -> str:
    if status not in VALID_STATUSES:
        raise GateError(f"invalid gate status {status!r}")
    if not evidence:
        raise GateError("gate evidence is required")
    if not name.startswith("G") or not name[1:].isdigit():
        raise GateError("gate name must be G0..G6")
    records: list[JsonDict] = []
    for path in evidence:
        if not path.is_file():
            raise GateError(f"missing evidence: {path}")
        records.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    record: JsonDict = {
        "schema_version": 1,
        "kind": "gate",
        "name": name,
        "status": status,
        "reason": reason,
        "allowed_next": allowed_next,
        "evidence": records,
        "producing_commit": _commit(),
    }
    return write_record(out, record)


def require_gate(name: str, decisions_dir: Path) -> JsonDict:
    path = decisions_dir / f"{name}.json"
    if not path.is_file():
        raise GateError(f"missing gate decision: {path}")
    try:
        record = json.loads(path.read_text())
        validate_record("gate", record)
    except (ValueError, RecordError) as exc:
        raise GateError(f"invalid gate decision {path}: {exc}") from exc
    if record["name"] != name:
        raise GateError(f"gate identity mismatch: expected {name}")
    for item in record["evidence"]:
        evidence_path = Path(item["path"])
        if not evidence_path.is_file() or _sha256(evidence_path) != item["sha256"]:
            raise GateError(f"gate evidence missing or changed: {evidence_path}")
    if record["status"] not in {"pass", "narrow"}:
        raise GateError(f"gate {name} is {record['status']}: {record['reason']}")
    return record
