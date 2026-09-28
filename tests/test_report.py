from __future__ import annotations

from pathlib import Path

import pytest


def _stopped_results(tmp_path: Path) -> Path:
    from kvrefine.gates import write_decision
    from kvrefine.records import write_record

    root = tmp_path / "results"
    environment = root / "environment.json"
    write_record(environment, {"schema_version": 1, "kind": "environment", "ram_bytes": 36 * 1024**3, "python": "3.12", "architecture": "arm64", "unavailable": {}, "packages": {"mlx": "0.32.2", "mlx-lm": "0.31.3"}})
    write_decision("G0", "pass", [environment], "capability works; novelty provisional", ["G1"], root / "decisions/G0.json")
    size = root / "evidence/G1/size-512.json"
    mib = 1024**2
    shared_tail = 28 * mib + 9 * 114688 + int(15.75 * mib)
    write_record(size, {
        "schema_version": 1,
        "kind": "g1-probe",
        "model_revision": "a" * 40,
        "diagnostic_full_raw_cache": True,
        "prompts": [{
            "prompt_id": "dev-prose-512", "tokens": 512,
            "summary": {"pages": 7168, "streams": 448, "values": 28 * 2 * 8 * 512 * 128, "mode_counts": {"0": 100, "1": 10, "2": 1}, "totals": {
                "raw_bytes": 56 * 1024**2, "q_bytes": 14 * 1024**2,
                "scales_biases_bytes": int(1.75 * 1024**2), "refinement_bytes": 39 * 1024**2,
                "refinement_payload_bytes": 38 * 1024**2, "metadata_bytes": 1 * 1024**2,
                "palette_bytes": 44 * 1024**2, "field_zstd_bytes": 40 * 1024**2,
                "xor_zstd_bytes": 41 * 1024**2, "shared_bytes": int(54.75 * 1024**2),
            }},
            "projections": {
                "shared_two_stage": {"cache_peak_bytes": int(54.75 * mib) + 4 * mib + shared_tail},
                "shared_one_stage": {"cache_peak_bytes": int(54.75 * mib) + 2 * mib + shared_tail},
                "raw_exact": {"cache_peak_bytes": 84 * 1024**2},
            },
            "tail_reservation": {
                "exact_authoritative_bytes": 28 * mib,
                "exact_candidate_bytes": 9 * 114688,
                "native_extra_capacity_tokens": 512,
                "native_draft_bytes": int(15.75 * mib),
                "shared_tail_bytes": shared_tail,
                "raw_exact_tail_bytes": 28 * mib,
            },
        }],
    })
    write_decision("G1", "stop", [size, root / "decisions/G0.json"], "size no-go", ["report"], root / "decisions/G1.json")
    return root


def test_early_stop_needs_no_gpu_or_runtime_records(tmp_path):
    from kvrefine.report import build_report

    root = _stopped_results(tmp_path)
    out = root / "report.md"
    summary = build_report(root, out)
    report = out.read_text()
    assert summary["conclusion"] == "size-no-go"
    assert "G2–G6: not run" in report
    assert "E1" in report and "E2" in report and "E3" in report
    assert "throughput was not measured" in report


def test_projection_is_not_reported_as_measured_capacity(tmp_path):
    from kvrefine.report import build_report

    root = _stopped_results(tmp_path)
    out = root / "report.md"
    build_report(root, out)
    report = out.read_text()
    assert "projected" in report
    assert "measured maximum context" not in report
    assert "additional native Q4 tail capacity" in report
    assert "verifier-candidate workspace" in report


def test_narrative_byte_ranges_are_derived_from_evidence(tmp_path):
    from kvrefine.report import build_report

    root = _stopped_results(tmp_path)
    out = root / "report.md"
    build_report(root, out)
    report = out.read_text()
    assert "54.75 MiB" in report
    assert "55.10–55.22 MiB" not in report
    assert "1 development prompt" in report


def test_e3_is_qualified_without_stock_generation_evidence(tmp_path):
    from kvrefine.report import build_report

    root = _stopped_results(tmp_path)
    out = root / "report.md"
    build_report(root, out)
    assert "E3 stock greedy equivalence was not tested" in out.read_text()


def test_process_and_allocator_counters_are_not_summed():
    from kvrefine.report import physical_observation_lines

    lines = physical_observation_lines({"process_peak_bytes": 1000, "mlx_active_peak_bytes": 600})
    assert "1000" in " ".join(lines)
    assert "600" in " ".join(lines)
    assert "1600" not in " ".join(lines)


def test_report_refuses_changed_hashed_evidence(tmp_path):
    from kvrefine.report import ReportError, build_report

    root = _stopped_results(tmp_path)
    size = root / "evidence/G1/size-512.json"
    size.write_bytes(size.read_bytes() + b" ")
    with pytest.raises(ReportError, match="evidence"):
        build_report(root, root / "report.md")


def test_evidence_index_paths_relocate_with_results(tmp_path):
    from kvrefine.report import build_report

    root = _stopped_results(tmp_path)
    index = build_report(root, root / "report.md")
    for item in index["evidence"].values():
        assert not Path(item["path"]).is_absolute()
        assert (root / item["path"]).is_file()
