"""Evidence-checked findings for an early research stop or later gate result."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .gates import GateError, read_decision
from .records import JsonDict, write_record


class ReportError(ValueError):
    """A findings claim cannot be traced to intact gate evidence."""


def _mib(value: int) -> str:
    return f"{value / 1024**2:.2f}"


def _range_mib(values: list[int]) -> str:
    low, high = min(values), max(values)
    return _mib(low) if _mib(low) == _mib(high) else f"{_mib(low)}–{_mib(high)}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_observation_lines(observations: JsonDict) -> list[str]:
    lines = []
    if observations.get("process_peak_bytes") is not None:
        lines.append(f"Process peak: {observations['process_peak_bytes']} bytes (physical/process observation).")
    if observations.get("mlx_active_peak_bytes") is not None:
        lines.append(f"MLX active peak: {observations['mlx_active_peak_bytes']} bytes (allocator observation; overlaps process memory).")
    if not lines:
        lines.append("Process peak and MLX allocator peak were not measured for G1; projected cache bytes are not physical high-water marks.")
    return lines


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _checked_inputs(results_root: Path) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict, JsonDict]:
    try:
        g0 = read_decision("G0", results_root / "decisions")
        g1 = read_decision("G1", results_root / "decisions")
    except GateError as exc:
        raise ReportError(f"gate evidence is unavailable or changed: {exc}") from exc
    if g0["status"] != "pass" or g1["status"] != "stop":
        raise ReportError("this report renderer requires a passed G0 and measured G1 stop")
    size_path = results_root / "evidence/G1/size-512.json"
    if not any((results_root / "decisions" / item["path"]).resolve() == size_path.resolve() for item in g1["evidence"]):
        raise ReportError("G1 size evidence is not bound by the gate decision")
    try:
        size = json.loads(size_path.read_text())
        environment = json.loads((results_root / "environment.json").read_text())
    except (OSError, ValueError) as exc:
        raise ReportError(f"measured evidence cannot be read: {exc}") from exc
    if size.get("kind") != "g1-probe" or size.get("schema_version") != 1 or not size.get("prompts"):
        raise ReportError("G1 probe record is incomplete")
    if environment.get("kind") != "environment" or environment.get("schema_version") != 1:
        raise ReportError("G0 environment record is incomplete")
    return g0, g1, size, environment, {"size_path": size_path}


def build_report(results_root: Path, out: Path) -> JsonDict:
    g0, g1, size, environment, paths = _checked_inputs(results_root)
    rows = size["prompts"]
    measured: list[JsonDict] = []
    total_pages = 0
    for row in rows:
        summary = row["summary"]
        totals = summary["totals"]
        raw = totals["raw_bytes"]
        shared = totals["shared_bytes"]
        if summary["streams"] != 448 or raw <= 0 or shared != totals["q_bytes"] + totals["scales_biases_bytes"] + totals["refinement_bytes"]:
            raise ReportError(f"incomplete or inconsistent byte accounting for {row.get('prompt_id')}")
        if totals["refinement_bytes"] != totals["refinement_payload_bytes"] + totals["metadata_bytes"]:
            raise ReportError("refinement payload and metadata do not add to serialized extent")
        if totals["refinement_bytes"] != totals["refinement_page_bytes"] + totals["container_metadata_bytes"] or totals["container_metadata_bytes"] <= 0:
            raise ReportError("tensor container is missing from conditional size accounting")
        if summary["pages"] * 4096 != summary["values"]:
            raise ReportError("page and captured-value totals disagree")
        total_pages += summary["pages"]
        control_projections = {}
        for name in ("palette_dual", "zstd_dual", "palette_exact_only", "zstd_exact_only"):
            control_projections[name] = {}
            for stage_label in ("one_stage", "two_stage"):
                key = f"{name}_{stage_label}"
                control_projections[name][stage_label] = row["projections"][key]["cache_peak_bytes"]
        measured.append({
            "prompt_id": row["prompt_id"],
            "tokens": row["tokens"],
            "raw_bytes": raw,
            "q_bytes": totals["q_bytes"],
            "scales_biases_bytes": totals["scales_biases_bytes"],
            "refinement_bytes": totals["refinement_bytes"],
            "metadata_bytes": totals["metadata_bytes"],
            "container_metadata_bytes": totals["container_metadata_bytes"],
            "shared_bytes": shared,
            "palette_dual_bytes": totals["q_bytes"] + totals["scales_biases_bytes"] + totals["palette_bytes"],
            "zstd_dual_bytes": totals["q_bytes"] + totals["scales_biases_bytes"] + totals["field_zstd_bytes"],
            "palette_exact_only_bytes": totals["palette_bytes"],
            "zstd_exact_only_bytes": totals["field_zstd_bytes"],
            "shared_one_stage_projected_bytes": row["projections"]["shared_one_stage"]["cache_peak_bytes"],
            "shared_two_stage_projected_bytes": row["projections"]["shared_two_stage"]["cache_peak_bytes"],
            "raw_exact_projected_bytes": row["projections"]["raw_exact"]["cache_peak_bytes"],
            "tail_reservation": row["tail_reservation"],
            "control_projections": control_projections,
            "pages": summary["pages"],
        })
    if any(item["shared_one_stage_projected_bytes"] <= item["raw_exact_projected_bytes"] for item in measured):
        raise ReportError("G1 stop reason contradicts one-stage projection")
    if any(item["shared_two_stage_projected_bytes"] <= item["raw_exact_projected_bytes"] for item in measured):
        raise ReportError("G1 stop reason contradicts two-stage projection")
    shared_tails = [item["tail_reservation"]["shared_tail_bytes"] for item in measured]
    raw_tails = [item["tail_reservation"]["raw_exact_tail_bytes"] for item in measured]
    stage_one = [item["shared_one_stage_projected_bytes"] - item["shared_bytes"] - tail for item, tail in zip(measured, shared_tails)]
    stage_two = [item["shared_two_stage_projected_bytes"] - item["shared_bytes"] - tail for item, tail in zip(measured, shared_tails)]
    if any(
        tail != item["tail_reservation"]["exact_authoritative_bytes"]
        or shared_tail != tail + item["tail_reservation"]["exact_candidate_bytes"] + item["tail_reservation"]["native_draft_bytes"]
        or item["raw_exact_projected_bytes"] != item["raw_bytes"] + tail
        or first <= 0 or second != 2 * first
        for item, tail, shared_tail, first, second in zip(measured, raw_tails, shared_tails, stage_one, stage_two)
    ):
        raise ReportError("stage/tail projections do not match the declared one/two-stage schedule")
    for item, shared_tail, raw_tail, one_stage, two_stages in zip(measured, shared_tails, raw_tails, stage_one, stage_two):
        for name, base, tail in (
            ("palette_dual", item["palette_dual_bytes"], shared_tail),
            ("zstd_dual", item["zstd_dual_bytes"], shared_tail),
            ("palette_exact_only", item["palette_exact_only_bytes"], raw_tail),
            ("zstd_exact_only", item["zstd_exact_only_bytes"], raw_tail),
        ):
            projection = item["control_projections"][name]
            if projection["one_stage"] != base + tail + one_stage or projection["two_stage"] != base + tail + two_stages:
                raise ReportError(f"control projection does not match measured page bytes: {name}")
    prompt_count = len(measured)
    prompt_noun = "prompt" if prompt_count == 1 else "prompts"
    raw_range = _range_mib([item["raw_bytes"] for item in measured])
    shared_range = _range_mib([item["shared_bytes"] for item in measured])
    saving_range = _range_mib([item["raw_bytes"] - item["shared_bytes"] for item in measured])
    metadata_range = _range_mib([item["metadata_bytes"] for item in measured])
    optimistic_one = [item["shared_bytes"] + stage + raw_tail for item, stage, raw_tail in zip(measured, stage_one, raw_tails)]
    hypothetical_headroom = max(
        item["raw_exact_projected_bytes"] - (optimistic - item["metadata_bytes"])
        for item, optimistic in zip(measured, optimistic_one)
    )
    hypothetical_two_still_loses = all(
        item["shared_bytes"] + stage + raw_tail - item["metadata_bytes"] > item["raw_exact_projected_bytes"]
        for item, stage, raw_tail in zip(measured, stage_two, raw_tails)
    )
    lines = [
        "# ExactKV findings — G1 size no-go",
        "",
        "The CPU conditional interval-rank codec reconstructed every original BF16 word on the measured real-cache pages, but its fully counted shared prompt view saved too little memory to pay for exact layer staging. G1 stopped the runtime branch. This is an informative early no-go for the measured format and inputs, not a proof that all conditional KV codecs fail.",
        "",
        f"Evidence: [G0 decision](decisions/G0.json), [G1 stop decision](decisions/G1.json), [all-layer raw size record](evidence/G1/size-512.json), and [source audit](../docs/research/2026-09-28-source-audit.md). The G1 decision hashes the size record and its producing code; this report refuses changed evidence.",
        "",
        f"Machine and model: Apple M3 Max, {environment['ram_bytes'] / 1024**3:.0f} GiB physical RAM; Python {environment['python']}; MLX {environment.get('packages', {}).get('mlx', 'unrecorded')}; MLX-LM {environment.get('packages', {}).get('mlx-lm', 'unrecorded')}; original Qwen/Qwen3-0.6B BF16 snapshot `{size['model_revision']}`. The prompt length is the final chat-template token count, including non-thinking instructions.",
        "",
        "## Exactness and measured storage",
        "",
        f"E1 passed in the CPU reference: {total_pages:,} real 4,096-value pages across all 28 layers, 8 KV heads, both K and V, and the listed development prompts. The conditional codec, independent palette, field-split Zstd, and XOR control each decoded to the original uint16 BF16 words on every page. Synthetic tests also exercise every 16-bit BF16 pattern. E2 paired-verifier equality was not tested. E3 stock greedy equivalence was not tested.",
        "",
        "All numbers in the next table are measured prompt MiB. Q4 includes both packed codes and native BF16 scales/biases. Conditional bytes are the complete version-one EKVT serialization: literal groups, EKVR page headers, modes, restart offsets, alignment, guards, and the tensor-container header, manifest, source-side hash, tensor descriptors, and page-offset index. Independent controls are measured page-local code streams; their tensor indexes are excluded, so their figures are lower bounds. Independent exact-only columns intentionally omit Q4.",
        "",
        "| Development prompt | Raw BF16 | Q4 codes + S/B | Q4 + conditional | Q4 + palette | Q4 + field-Zstd | Palette only | Field-Zstd only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in measured:
        q = item["q_bytes"] + item["scales_biases_bytes"]
        lines.append(
            f"| {item['prompt_id']} | {_mib(item['raw_bytes'])} | {_mib(q)} | {_mib(item['shared_bytes'])} | {_mib(item['palette_dual_bytes'])} | {_mib(item['zstd_dual_bytes'])} | {_mib(item['palette_exact_only_bytes'])} | {_mib(item['zstd_exact_only_bytes'])} |"
        )
    lines += [
        "",
        f"The EKVT container contributes {_range_mib([item['container_metadata_bytes'] for item in measured])} MiB per prompt. The page-local field-Zstd control is a CPU size comparator; it has no GPU decoder or fair inference runtime result. The simple palette is likewise a local baseline, not a reproduction of SplitZip.",
        "",
        "## Projected working bytes and decision",
        "",
        f"These figures add the measured serialized prompt to a {_range_mib(raw_tails)} MiB authoritative BF16 tail, {_range_mib([item['tail_reservation']['exact_candidate_bytes'] for item in measured])} MiB verifier-candidate workspace, {_range_mib([item['tail_reservation']['native_draft_bytes'] for item in measured])} MiB additional native Q4 tail capacity, and one or two {_range_mib(stage_one)} MiB layer-sized BF16 stages at {measured[0]['tokens']} tokens. The Q4 reservation covers 256 outputs plus 9 verifier positions and rounds to the installed 256-token allocation step; raw exact execution needs only its authoritative tail. These are **projected cache-related bytes**, not process high-water marks; other transient copies and allocator effects could increase them. No maximum fitting context was measured. The G1 capture deliberately retained a full raw cache while diagnosing each layer, so it cannot establish final-system physical peak savings.",
        "",
        "| Development prompt | Conditional + one stage + tail | Conditional + two stages + tail | Raw BF16 + tail |",
        "|---|---:|---:|---:|",
    ]
    for item in measured:
        lines.append(
            f"| {item['prompt_id']} | {_mib(item['shared_one_stage_projected_bytes'])} | {_mib(item['shared_two_stage_projected_bytes'])} | {_mib(item['raw_exact_projected_bytes'])} |"
        )
    lines += [
        "",
        "The table below applies the same one/two-stage and tail schedules to independent controls. Its controls omit a tensor-level index, making their bytes optimistic lower bounds. The exact-only controls have no native Q4 tail, but would need to decode for ordinary exact attention; that runtime was not measured. Ratios use the raw exact-plus-tail bytes for the same prompt.",
        "",
        "| Development prompt | Representation | One stage MiB | Two stages MiB | One-stage ratio to raw exact | Two-stage ratio to raw exact |",
        "|---|---|---:|---:|---:|---:|",
    ]
    control_names = (
        ("palette_dual", "Q4 + palette"),
        ("zstd_dual", "Q4 + field-Zstd"),
        ("palette_exact_only", "Palette exact only"),
        ("zstd_exact_only", "Field-Zstd exact only"),
    )
    for item in measured:
        raw = item["raw_exact_projected_bytes"]
        for key, label in control_names:
            one = item["control_projections"][key]["one_stage"]
            two = item["control_projections"][key]["two_stage"]
            lines.append(f"| {item['prompt_id']} | {label} | {_mib(one)} | {_mib(two)} | {one / raw:.2f}× | {two / raw:.2f}× |")
    lines += [
        "",
        f"Across {prompt_count} development {prompt_noun}, Q4 plus conditional refinement occupied {shared_range} MiB of the {raw_range} MiB raw prompt. The mandatory exact stage exceeded the {saving_range} MiB prompt saving even in an optimistic projection that omitted native draft and candidate tails: {_range_mib(optimistic_one)} MiB against {_range_mib([item['raw_exact_projected_bytes'] for item in measured])} MiB raw exact. Removing every refinement metadata byte ({metadata_range} MiB per prompt) from that optimistic case would leave at most {_mib(hypothetical_headroom)} MiB one-stage headroom" + (" and would still lose with two stages." if hypothetical_two_still_loses else "; the two-stage counterfactual is unresolved.") + " The specified-buffer projection is worse still. The proposed conditional format therefore has no credible measured memory advantage worth building a Metal decoder for. A longer context or different model may have different statistics; none was measured after this stop.",
        "",
        *physical_observation_lines({}),
        "",
        "G2–G6: not run. Committed throughput was not measured. Whole-request time, acceptance, output-burst latency, verifier correctness, CPU-to-Metal arithmetic agreement, and physical process peak were also not measured. There is no end-to-end speedup or usable verified-decoding claim. The low-bit Q4-only path is a lossy lower-memory choice and has a different correctness contract.",
        "",
        "## Prior art and scope",
        "",
        "[PackKV](https://arxiv.org/html/2512.24449v2) already combines lossy quantization, packing, and compute-integrated decompression. [QuantSpec](https://arxiv.org/html/2502.10424v1) and [Lynx](https://arxiv.org/html/2607.01831v1) establish hierarchical low-bit views and speculative refinement to their integer verification targets. [VeriCache](https://arxiv.org/html/2605.17613v1) and [FAFO](https://github.com/Escanord/FAFO/blob/main/README.md) establish approximate-cache proposals with exact verification. [SplitZip](https://arxiv.org/html/2605.01708v3) establishes independent exact BF16 field/palette coding. The narrowed conditional original-BF16 representation remains only a provisional distinction: full [Vayne](https://ieeexplore.ieee.org/document/11575358/) and [ProofKV](https://openreview.net/forum?id=3Zmk1MTztm) mechanisms could not be retrieved, so their overlap is unresolved. The source audit records the specific reviewed sections and access limits. This report makes no global novelty or patent-clearance claim.",
        "",
        "## Reproduction and limits",
        "",
        "The [README](../README.md) pins the environment, checkpoint, source partitions, small exactness tests, and explicit three-prompt G1 command. Re-run `python -m kvrefine.cli report --results results --out results/report.md` to regenerate this report after validating the committed evidence. The committed G1 artifact contains per-layer/head, K/V, mode, rank-width, and full serialized-byte details. The diagnostic capture and CPU packing are not a bounded construction implementation; no physical memory saving should be inferred from their temporary allocations. A fresh independent person's reproduction has not yet been recorded.",
        "",
    ]
    _atomic_text(out, "\n".join(lines))
    summary: JsonDict = {
        "schema_version": 1,
        "kind": "evidence-index",
        "conclusion": "size-no-go",
        "gate_statuses": {"G0": g0["status"], "G1": g1["status"], "G2-G6": "not-run"},
        "model_revision": size["model_revision"],
        "exact_cpu_pages": total_pages,
        "measured": measured,
        "report_sha256": _sha256(out),
        "evidence": {
            "G0": {"path": "decisions/G0.json", "sha256": _sha256(results_root / "decisions/G0.json")},
            "G1": {"path": "decisions/G1.json", "sha256": _sha256(results_root / "decisions/G1.json")},
            "size_512": {"path": "evidence/G1/size-512.json", "sha256": _sha256(paths["size_path"])},
        },
    }
    write_record(results_root / "evidence/index.json", summary)
    return summary
