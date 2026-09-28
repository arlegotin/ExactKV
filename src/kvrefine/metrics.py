"""Non-overlapping owned-byte ledger and explicit cache-peak projections."""

from __future__ import annotations

from dataclasses import dataclass, field

from .records import JsonDict


def kv_payload_bytes(tokens: int, layers: int = 28, kv_heads: int = 8, head_dim: int = 128) -> int:
    if min(tokens, layers, kv_heads, head_dim) < 0:
        raise ValueError("KV shape values must be nonnegative")
    return 2 * layers * kv_heads * tokens * head_dim * 2


def tail_reservation(prompt_tokens: int, output_tokens: int = 256, verifier_workspace_tokens: int = 9) -> JsonDict:
    """Conservative batch-one capacity for the specified BF16 and native Q4 tails."""
    if min(prompt_tokens, output_tokens, verifier_workspace_tokens) < 0:
        raise ValueError("prompt, output and verifier workspace lengths must be nonnegative")
    step = 256  # installed QuantizedKVCache growth, asserted by G0's native ABI
    round_step = lambda count: (count + step - 1) // step * step
    extra_native_tokens = round_step(prompt_tokens + output_tokens + verifier_workspace_tokens) - round_step(prompt_tokens)
    exact_authoritative = kv_payload_bytes(output_tokens)
    exact_candidates = kv_payload_bytes(verifier_workspace_tokens)
    native_draft = kv_payload_bytes(extra_native_tokens) * 36 // 128  # Q4 codes + BF16 S/B per 64 values
    return {
        "exact_authoritative_bytes": exact_authoritative,
        "exact_candidate_bytes": exact_candidates,
        "native_extra_capacity_tokens": extra_native_tokens,
        "native_draft_bytes": native_draft,
        "shared_tail_bytes": exact_authoritative + exact_candidates + native_draft,
        "raw_exact_tail_bytes": exact_authoritative,
    }


@dataclass(slots=True)
class ByteLedger:
    _owners: dict[str, JsonDict] = field(default_factory=dict)

    def add(self, owner_id: str, category: str, allocated_bytes: int, logical_bytes: int) -> None:
        if not owner_id or not category or allocated_bytes < 0 or logical_bytes < 0 or logical_bytes > allocated_bytes:
            raise ValueError("invalid owner/category or byte size")
        if owner_id in self._owners:
            prior = self._owners[owner_id]
            if (prior["allocated_bytes"], prior["logical_bytes"]) != (allocated_bytes, logical_bytes):
                raise ValueError(f"owner {owner_id} has inconsistent allocation sizes")
            if category not in prior["aliases"] and category != prior["category"]:
                prior["aliases"].append(category)
            return
        self._owners[owner_id] = {
            "category": category,
            "allocated_bytes": allocated_bytes,
            "logical_bytes": logical_bytes,
            "aliases": [],
        }

    def snapshot(self) -> JsonDict:
        categories: JsonDict = {}
        for entry in self._owners.values():
            category = entry["category"]
            subtotal = categories.setdefault(category, {"allocated_bytes": 0, "logical_bytes": 0})
            subtotal["allocated_bytes"] += entry["allocated_bytes"]
            subtotal["logical_bytes"] += entry["logical_bytes"]
        return {
            "allocated_bytes": sum(entry["allocated_bytes"] for entry in self._owners.values()),
            "logical_bytes": sum(entry["logical_bytes"] for entry in self._owners.values()),
            "by_category": categories,
            "owners": dict(self._owners),
        }


def cache_projection(prompt: JsonDict, stage_bytes: int, tail_bytes: int, stage_count: int) -> JsonDict:
    if stage_count < 0 or stage_bytes < 0 or tail_bytes < 0 or any(not isinstance(value, int) or value < 0 for value in prompt.values()):
        raise ValueError("cache projection requires nonnegative integer bytes")
    transient = prompt.get("transient_copies", 0)
    resident = sum(value for key, value in prompt.items() if key != "transient_copies")
    stages = stage_bytes * stage_count
    return {
        "prompt_components": dict(prompt),
        "shared_prompt_bytes": resident,
        "stage_bytes": stages,
        "tail_bytes": tail_bytes,
        "transient_copies_bytes": transient,
        "prompt_plus_stages_bytes": resident + stages,
        "cache_peak_bytes": resident + stages + tail_bytes + transient,
    }
