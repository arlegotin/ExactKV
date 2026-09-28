"""Pinned, disjoint public prompts with exact final chat-template token IDs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .model import _validate_lock
from .records import JsonDict


class DataError(ValueError):
    """A source, prompt, or token artifact does not match its pinned identity."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_replace(source: Path, target: Path) -> None:
    os.replace(source, target)


def write_manifest(path: Path, records: list[JsonDict]) -> None:
    payload = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_disjoint_sources(records: list[JsonDict]) -> None:
    identities: dict[str, str] = {}
    code_dirs: dict[str, str] = {}
    text_hashes: dict[str, str] = {}
    for record in records:
        split = record["split"]
        for source in record.get("source_ids", []):
            previous = identities.setdefault(source, split)
            if previous != split:
                raise DataError(f"source overlap across splits: {source}")
            if record["domain"] == "code":
                top_level = source.split("/", 1)[0]
                previous_dir = code_dirs.setdefault(top_level, split)
                if previous_dir != split:
                    raise DataError(f"code-directory overlap across splits: {top_level}")
        text_hash = record.get("text_sha256")
        if text_hash:
            previous = text_hashes.setdefault(text_hash, split)
            if previous != split:
                raise DataError("text overlap across splits")


def fit_source_to_tokens(tokenizer: Any, source_text: str, target: int) -> tuple[list[int], str]:
    if target < 1 or not source_text:
        raise DataError("positive target and nonempty source text are required")
    source_text = source_text[: max(6_000, target * 20)]
    prefix = "Continue the following text exactly where it ends:\n\n"

    def tokens(chars: int) -> list[int]:
        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix + source_text[:chars]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(result, list) or not all(isinstance(value, int) for value in result):
            raise DataError("tokenizer did not return an integer token list")
        return result

    if len(tokens(len(source_text))) < target:
        raise DataError(f"source has too few tokens for target {target}")
    low, high = 0, len(source_text)
    while low < high:
        mid = (low + high) // 2
        if len(tokens(mid)) < target:
            low = mid + 1
        else:
            high = mid
    for chars in range(max(1, low - 96), min(len(source_text), low + 96) + 1):
        ids = tokens(chars)
        if len(ids) == target:
            return ids, source_text[:chars]
    raise DataError(f"cannot obtain exactly {target} templated tokens from this source")


def _checked_file(item: JsonDict, source: str, revision: str) -> Path:
    path = Path(item["path"])
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        if source == "wikitext":
            from huggingface_hub import hf_hub_download

            downloaded = Path(
                hf_hub_download(
                    "Salesforce/wikitext",
                    item["repo_path"],
                    repo_type="dataset",
                    revision=revision,
                    local_dir="artifacts/sources/wikitext",
                )
            )
            if downloaded.resolve() != path.resolve():
                raise DataError("WikiText download path differs from pinned source lock")
        else:
            url = f"https://raw.githubusercontent.com/python/cpython/{revision}/{item['repo_path']}"
            request = urllib.request.Request(url, headers={"User-Agent": "ExactKV-research"})
            path.write_bytes(urllib.request.urlopen(request, timeout=20).read())
    if _sha256(path.read_bytes()) != item["sha256"]:
        raise DataError(f"source hash mismatch: {path}")
    return path


_ARTICLE = re.compile(r"^ = [^=].* =\s*$")


def _wiki_material(lock: JsonDict, split: str) -> tuple[str, list[str], list[JsonDict]]:
    import pyarrow.parquet as pq

    partitions = ("train",) if split == "dev" else ("validation", "test")
    pieces: list[str] = []
    identities: list[str] = []
    bounds: list[JsonDict] = []
    for partition in partitions:
        item = lock["wikitext"]["splits"][partition]
        rows = pq.read_table(_checked_file(item, "wikitext", lock["wikitext"]["revision"]), columns=["text"]).column("text").to_pylist()
        starts = [index for index, row in enumerate(rows) if _ARTICLE.match(row)]
        for number, start in enumerate(starts):
            end = starts[number + 1] if number + 1 < len(starts) else len(rows)
            article = "\n".join(rows[start:end]).strip()
            if len(article) < 300:
                continue
            identity = f"{partition}:{start}:{rows[start].strip()}"
            offset = sum(len(piece) for piece in pieces)
            separator = "\n\n" if pieces else ""
            piece = separator + article
            pieces.append(piece)
            identities.append(identity)
            bounds.append({"source_id": identity, "start": offset + len(separator), "end": offset + len(piece)})
            if sum(len(piece) for piece in pieces) >= 150_000:
                return "".join(pieces), identities, bounds
    return "".join(pieces), identities, bounds


def _code_material(lock: JsonDict, split: str) -> tuple[str, list[str], list[JsonDict]]:
    pieces: list[str] = []
    identities: list[str] = []
    bounds: list[JsonDict] = []
    for item in lock["cpython"]["files"]:
        if item["split"] != split:
            continue
        source_id = item["repo_path"].removeprefix("Lib/")
        text = _checked_file(item, "cpython", lock["cpython"]["revision"]).read_text()
        separator = f"\n\n# Source: {source_id}\n"
        offset = sum(len(piece) for piece in pieces)
        piece = separator + text
        pieces.append(piece)
        identities.append(source_id)
        bounds.append({"source_id": source_id, "start": offset + len(separator), "end": offset + len(piece)})
    return "".join(pieces), identities, bounds


def _structured_material(lock: JsonDict, split: str) -> tuple[str, list[str], list[JsonDict]]:
    seeds = lock["structured"][f"{split}_seeds"]
    templates = lock["structured"][f"{split}_templates"]
    seed = seeds[0]
    rng = random.Random(seed)
    lines = []
    for index in range(5_000):
        template = templates[index % len(templates)]
        value = {
            "kind": template,
            "sequence": index,
            "event_id": f"{rng.getrandbits(64):016x}",
            "quantity": round(rng.uniform(-10_000, 10_000), 3),
            "status": rng.choice(["ok", "retry", "failed", "pending"]),
            "tags": [f"s{rng.randrange(1000)}", f"r{rng.randrange(1000)}"],
        }
        lines.append(json.dumps(value, sort_keys=True, separators=(",", ":")))
    text = "\n".join(lines)
    source_id = f"structured:{split}:{seed}:{','.join(templates)}"
    return text, [source_id], [{"source_id": source_id, "start": 0, "end": len(text)}]


def _store_content(directory: Path, suffix: str, payload: bytes) -> tuple[str, str]:
    digest = _sha256(payload)
    path = directory / f"{digest}{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DataError(f"content-addressed artifact conflict: {path}")
    else:
        path.write_bytes(payload)
    return str(path), digest


def build_manifest(source_lock: JsonDict, model_lock: JsonDict, split: str, out: Path, tokens: int = 512) -> list[JsonDict]:
    if split not in {"dev", "heldout"}:
        raise DataError("split must be dev or heldout")
    if source_lock.get("kind") != "source-lock" or source_lock.get("schema_version") != 1:
        raise DataError("invalid source lock")
    root, _ = _validate_lock(model_lock)
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(str(root), tokenizer_config_extra={"trust_remote_code": False})
    records: list[JsonDict] = []
    for domain, producer in (("prose", _wiki_material), ("code", _code_material), ("structured", _structured_material)):
        material, source_ids, bounds = producer(source_lock, split)
        ids, used = fit_source_to_tokens(tokenizer, material, tokens)
        relevant_bounds = [
            {**bound, "end": min(bound["end"], len(used))}
            for bound in bounds
            if bound["start"] < len(used)
        ]
        relevant_ids = [bound["source_id"] for bound in relevant_bounds]
        token_payload = (json.dumps(ids, separators=(",", ":")) + "\n").encode()
        token_path, token_hash = _store_content(Path("artifacts/tokens"), ".json", token_payload)
        text_path, text_hash = _store_content(Path("artifacts/text"), ".txt", used.encode())
        records.append({
            "schema_version": 1,
            "kind": "prompt",
            "prompt_id": f"{split}-{domain}-{tokens}",
            "split": split,
            "domain": domain,
            "tokens": tokens,
            "token_artifact": token_path,
            "token_sha256": token_hash,
            "text_artifact": text_path,
            "text_sha256": text_hash,
            "source_ids": relevant_ids,
            "source_bounds": relevant_bounds,
            "source_revision": source_lock["wikitext"]["revision"] if domain == "prose" else source_lock["cpython"]["revision"] if domain == "code" else f"seed-{source_lock['structured'][f'{split}_seeds'][0]}",
            "source_license": source_lock["wikitext"]["license"] if domain == "prose" else source_lock["cpython"]["license"] if domain == "code" else "generated",
            "model_revision": model_lock["revision"],
            "tokenizer_revision": model_lock["tokenizer_revision"],
            "generation": {"mode": "greedy", "enable_thinking": False, "add_generation_prompt": True},
        })
    previous = []
    if out.exists():
        previous = [json.loads(line) for line in out.read_text().splitlines() if line]
    existing = {(record["split"], record["domain"], record["tokens"]): record for record in previous}
    for record in records:
        key = (record["split"], record["domain"], record["tokens"])
        if key in existing and existing[key] != record:
            raise DataError(f"immutable prompt record changed: {key}")
        existing[key] = record
    combined = sorted(existing.values(), key=lambda record: (record["tokens"], record["split"], record["domain"]))
    validate_disjoint_sources(combined)
    write_manifest(out, combined)
    return records


def load_prompts(manifest: Path, split: str, tokens: int, *, model_lock: JsonDict | None = None) -> list[JsonDict]:
    if model_lock is None:
        model_lock = json.loads(Path("data/model-lock.json").read_text())
    if not manifest.is_file():
        raise DataError(f"missing manifest: {manifest}")
    selected: list[JsonDict] = []
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    validate_disjoint_sources(records)
    for record in records:
        if record["split"] != split or record["tokens"] != tokens:
            continue
        if record["tokenizer_revision"] != model_lock["tokenizer_revision"]:
            raise DataError("tokenizer revision mismatch")
        path = Path(record["token_artifact"])
        if not path.is_file() or _sha256(path.read_bytes()) != record["token_sha256"]:
            raise DataError(f"token artifact missing or changed: {path}")
        if "text_artifact" in record:
            text_path = Path(record["text_artifact"])
            if not text_path.is_file() or _sha256(text_path.read_bytes()) != record["text_sha256"]:
                raise DataError(f"text artifact missing or changed: {text_path}")
        ids = json.loads(path.read_text())
        if len(ids) != tokens or not all(isinstance(value, int) for value in ids):
            raise DataError("token artifact length or type mismatch")
        selected.append({**record, "token_ids": ids})
    if not selected:
        raise DataError(f"no {split} prompts with {tokens} tokens")
    return selected
