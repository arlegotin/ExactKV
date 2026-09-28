"""G1 all-layer diagnostic probe and serialized-size comparison."""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from .baselines import (
    field_zstd_decode,
    field_zstd_encode,
    palette_decode,
    palette_encode,
    xor_zstd_decode,
    xor_zstd_encode,
)
from .codec import _bounds, decode_page, encode_page
from .data import load_prompts
from .format import PAGE_HEADER, container_nbytes
from .metrics import cache_projection, kv_payload_bytes, tail_reservation
from .model import capture_prompt, load_model
from .native import NativePage, bf16_words, quantize_kv
from .records import JsonDict


class ProbeError(RuntimeError):
    """A G1 page or aggregate is incomplete or not bit-exact."""


def measure_page(
    layer: int,
    role: str,
    head: int,
    page_id: int,
    words: np.ndarray,
    side: NativePage,
    predictor: np.ndarray,
) -> JsonDict:
    tensor_id = layer * 16 + (0 if role == "K" else 8) + head
    refinement = encode_page(words, side, tensor_id, page_id)
    palette = palette_encode(words)
    zstd = field_zstd_encode(words)
    xor = xor_zstd_encode(words, predictor)
    for name, reconstructed in (
        ("refinement", decode_page(refinement, side, tensor_id, page_id)),
        ("palette", palette_decode(palette)),
        ("field-Zstd", field_zstd_decode(zstd)),
        ("XOR-Zstd", xor_zstd_decode(xor, predictor)),
    ):
        if not np.array_equal(reconstructed, words):
            raise ProbeError(f"{name} reconstructed wrong BF16 words at layer={layer} role={role} head={head} page={page_id}")
    q_bytes = side.q.nbytes
    metadata_bytes = side.scales.nbytes + side.biases.nbytes
    payload_words = PAGE_HEADER.unpack_from(refinement)[9]
    payload_bytes = payload_words * 4
    groups = len(words) // 64
    modes = [(refinement[32 + group // 4] >> (2 * (group % 4))) & 3 for group in range(groups)]
    mode_counts = Counter(modes)
    widths = Counter()
    if 0 in modes or 1 in modes:
        bounds = {mode: _bounds(side, mode)[2] for mode in set(modes) if mode != 2}
        for group, mode in enumerate(modes):
            if mode != 2:
                for width in bounds[mode][group * 64 : (group + 1) * 64]:
                    widths[int(width)] += 1
    return {
        "layer": layer,
        "role": role,
        "head": head,
        "page_id": page_id,
        "values": len(words),
        "raw_bytes": words.nbytes,
        "q_bytes": q_bytes,
        "scales_biases_bytes": metadata_bytes,
        "refinement_bytes": len(refinement),
        "refinement_payload_bytes": payload_bytes,
        "metadata_bytes": len(refinement) - payload_bytes,
        "literal_bytes": mode_counts[2] * 128,
        "palette_bytes": len(palette),
        "field_zstd_bytes": len(zstd),
        "xor_zstd_bytes": len(xor),
        "shared_bytes": q_bytes + metadata_bytes + len(refinement),
        "independent_dual_palette_bytes": q_bytes + metadata_bytes + len(palette),
        "independent_dual_zstd_bytes": q_bytes + metadata_bytes + len(zstd),
        "mode_counts": {str(mode): mode_counts[mode] for mode in (0, 1, 2)},
        "rank_width_counts": {str(width): count for width, count in sorted(widths.items())},
    }


_BYTE_FIELDS = (
    "raw_bytes", "q_bytes", "scales_biases_bytes", "refinement_bytes", "refinement_payload_bytes",
    "metadata_bytes", "literal_bytes", "palette_bytes", "field_zstd_bytes", "xor_zstd_bytes",
    "shared_bytes", "independent_dual_palette_bytes", "independent_dual_zstd_bytes",
)


def aggregate_pages(pages: Iterable[JsonDict], *, source_identity: JsonDict | None = None) -> JsonDict:
    totals = {key: 0 for key in _BYTE_FIELDS}
    roles = {role: {"pages": 0, "raw_bytes": 0, "shared_bytes": 0} for role in ("K", "V")}
    layers: dict[str, JsonDict] = {}
    streams: dict[str, JsonDict] = {}
    modes: Counter[int] = Counter()
    widths: Counter[int] = Counter()
    page_count = value_count = 0
    page_lengths: list[int] = []
    descriptors: list[JsonDict] = []
    descriptor_by_stream: dict[str, JsonDict] = {}
    native_hash = hashlib.sha256()
    native_abi_id: str | None = None
    for item in pages:
        side: NativePage = item["side"]
        if native_abi_id is None:
            native_abi_id = side.abi_id
        elif native_abi_id != side.abi_id:
            raise ProbeError("native ABI changed within one prompt")
        measured = measure_page(**item)
        page_lengths.append(measured["refinement_bytes"])
        native_hash.update(side.q.tobytes())
        native_hash.update(side.scales.tobytes())
        native_hash.update(side.biases.tobytes())
        page_count += 1
        value_count += measured["values"]
        for key in _BYTE_FIELDS:
            totals[key] += measured[key]
        role = measured["role"]
        roles[role]["pages"] += 1
        roles[role]["raw_bytes"] += measured["raw_bytes"]
        roles[role]["shared_bytes"] += measured["shared_bytes"]
        layer = str(measured["layer"])
        layer_stats = layers.setdefault(layer, {"pages": 0, "raw_bytes": 0, "shared_bytes": 0})
        layer_stats["pages"] += 1
        layer_stats["raw_bytes"] += measured["raw_bytes"]
        layer_stats["shared_bytes"] += measured["shared_bytes"]
        stream_id = f"{layer}:{role}:{measured['head']}"
        descriptor = descriptor_by_stream.get(stream_id)
        if descriptor is None:
            if item["page_id"] != 0:
                raise ProbeError("first page of a tensor must have page ID zero")
            descriptor = {"layer": measured["layer"], "role": role, "head": measured["head"], "first_page": page_count - 1, "page_count": 0}
            descriptor_by_stream[stream_id] = descriptor
            descriptors.append(descriptor)
        if descriptor["first_page"] + descriptor["page_count"] != page_count - 1 or item["page_id"] != descriptor["page_count"]:
            raise ProbeError("tensor pages must be contiguous and ordered")
        descriptor["page_count"] += 1
        stream = streams.setdefault(stream_id, {"pages": 0, "raw_bytes": 0, "shared_bytes": 0})
        stream["pages"] += 1
        stream["raw_bytes"] += measured["raw_bytes"]
        stream["shared_bytes"] += measured["shared_bytes"]
        modes.update({int(mode): count for mode, count in measured["mode_counts"].items()})
        widths.update({int(width): count for width, count in measured["rank_width_counts"].items()})
    if page_count:
        identity = dict(source_identity or {})
        if any(key in identity for key in ("page_offsets", "streams", "native_abi_id", "native_side_sha256")):
            raise ProbeError("source identity conflicts with container fields")
        container_manifest = {
            **identity,
            "codec": "interval-rank-v1",
            "native_abi_id": native_abi_id,
            "native_side_sha256": native_hash.hexdigest(),
            "source_q_page_version": 1,
            "q_bits": 4,
            "group_size": 64,
            "scale_bias_dtype": "bf16",
            "streams": descriptors,
        }
        container_bytes = container_nbytes(container_manifest, page_lengths) - sum(page_lengths)
    else:
        container_bytes = 0
    totals["refinement_page_bytes"] = totals["refinement_bytes"]
    totals["container_metadata_bytes"] = container_bytes
    totals["refinement_bytes"] += container_bytes
    totals["metadata_bytes"] += container_bytes
    totals["shared_bytes"] += container_bytes
    return {
        "pages": page_count,
        "streams": len(streams),
        "values": value_count,
        "totals": totals,
        "roles": roles,
        "layers": layers,
        "heads": streams,
        "mode_counts": {str(mode): modes[mode] for mode in (0, 1, 2)},
        "rank_width_counts": {str(width): count for width, count in sorted(widths.items())},
    }


def _real_pages(handle, prompt: JsonDict, allocation: JsonDict):
    import mlx.core as mx

    tokens = prompt["tokens"]
    allocation["diagnostic_full_raw_cache"] = True
    allocation["raw_allocated_bytes"] = 0
    allocation["native_allocated_bytes"] = 0
    for layer, keys, values in capture_prompt(handle, prompt):
        if layer % 7 == 0:
            print(f"probe {prompt['prompt_id']}: layer {layer}/28", flush=True)
        native = quantize_kv(keys, values, tokens)
        allocation["raw_allocated_bytes"] += keys.nbytes + values.nbytes
        allocation["native_allocated_bytes"] += sum(array.nbytes for array in (*native.keys, *native.values))
        source = {"K": bf16_words(keys), "V": bf16_words(values)}
        quantized = {}
        predictors = {}
        for role, packed in (("K", native.keys), ("V", native.values)):
            valid = tuple(mx.contiguous(array[..., :tokens, :]) for array in packed)
            mx.eval(*valid)
            quantized[role] = (
                np.array(valid[0], dtype=np.uint32, copy=True),
                np.array(valid[1].view(mx.uint16), dtype=np.uint16, copy=True),
                np.array(valid[2].view(mx.uint16), dtype=np.uint16, copy=True),
            )
            dequantized = mx.dequantize(*packed, group_size=64, bits=4)[..., :tokens, :]
            predictors[role] = bf16_words(dequantized)
        for role in ("K", "V"):
            q, scales, biases = quantized[role]
            for head in range(8):
                for start in range(0, tokens, 32):
                    end = min(start + 32, tokens)
                    count = (end - start) * 128
                    side = NativePage(
                        q=q[0, head, start:end].reshape(-1),
                        scales=scales[0, head, start:end].reshape(-1),
                        biases=biases[0, head, start:end].reshape(-1),
                        valid_count=count,
                        abi_id=native.abi_id,
                        identity=f"{prompt['prompt_id']}:{layer}:{role}:{head}:{start // 32}",
                    )
                    yield {
                        "layer": layer,
                        "role": role,
                        "head": head,
                        "page_id": start // 32,
                        "words": source[role][0, head, start:end].reshape(-1),
                        "side": side,
                        "predictor": predictors[role][0, head, start:end].reshape(-1),
                    }
        del native, source, quantized, predictors


def run_probe(manifest: Path, tokens: int) -> JsonDict:
    from .gates import require_gate

    require_gate("G0", Path("results/decisions"))
    lock = json.loads(Path("data/model-lock.json").read_text())
    prompts = load_prompts(manifest, "dev", tokens, model_lock=lock)
    handle = load_model(lock)
    results: list[JsonDict] = []
    for prompt in prompts:
        print(f"probe {prompt['prompt_id']}: start", flush=True)
        allocation: JsonDict = {}
        summary = aggregate_pages(
            _real_pages(handle, prompt, allocation),
            source_identity={"model_revision": lock["revision"], "prompt_sha256": prompt["token_sha256"], "prompt_tokens": tokens},
        )
        expected_raw = kv_payload_bytes(tokens)
        if summary["streams"] != 28 * 8 * 2 or summary["totals"]["raw_bytes"] != expected_raw:
            raise ProbeError(f"all-layer capture incomplete for {prompt['prompt_id']}")
        totals = summary["totals"]
        layer_bytes = expected_raw // 28
        reservation = tail_reservation(tokens, output_tokens=256, verifier_workspace_tokens=9)
        shared_tail = reservation["shared_tail_bytes"]
        raw_tail = reservation["raw_exact_tail_bytes"]
        projections = {
            "shared_one_stage": cache_projection({"q_codes": totals["q_bytes"], "scales_biases": totals["scales_biases_bytes"], "refinement": totals["refinement_payload_bytes"], "metadata": totals["metadata_bytes"]}, layer_bytes, shared_tail, 1),
            "shared_two_stage": cache_projection({"q_codes": totals["q_bytes"], "scales_biases": totals["scales_biases_bytes"], "refinement": totals["refinement_payload_bytes"], "metadata": totals["metadata_bytes"]}, layer_bytes, shared_tail, 2),
            "raw_exact": cache_projection({"raw_bf16": expected_raw}, 0, raw_tail, 0),
        }
        for name, exact_bytes, native_view in (
            ("palette_dual", totals["palette_bytes"], True),
            ("zstd_dual", totals["field_zstd_bytes"], True),
            ("palette_exact_only", totals["palette_bytes"], False),
            ("zstd_exact_only", totals["field_zstd_bytes"], False),
        ):
            components = {"independent_exact": exact_bytes}
            if native_view:
                components.update({"q_codes": totals["q_bytes"], "scales_biases": totals["scales_biases_bytes"]})
            for label, count in (("one_stage", 1), ("two_stage", 2)):
                projections[f"{name}_{label}"] = cache_projection(components, layer_bytes, shared_tail if native_view else raw_tail, count)
        results.append({"prompt_id": prompt["prompt_id"], "prompt_sha256": prompt["token_sha256"], "tokens": tokens, "summary": summary, "allocation": allocation, "tail_reservation": reservation, "projections": projections})
    return {"schema_version": 1, "kind": "g1-probe", "model_revision": lock["revision"], "diagnostic_full_raw_cache": True, "prompts": results}
