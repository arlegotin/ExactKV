from __future__ import annotations

import shutil

import pytest

def test_gate_requires_matching_evidence(tmp_path):
    from kvrefine.gates import GateError, require_gate, write_decision

    evidence = tmp_path / "measurement.json"
    evidence.write_text('{"bytes": 42}\n')
    decision_path = tmp_path / "G0.json"
    write_decision("G0", "pass", [evidence], "smoke passed", ["G1"], decision_path)
    assert require_gate("G0", tmp_path)["status"] == "pass"
    evidence.write_text('{"bytes": 43}\n')
    with pytest.raises(GateError, match="evidence"):
        require_gate("G0", tmp_path)


def test_missing_gate_is_not_a_pass(tmp_path):
    from kvrefine.gates import GateError, require_gate

    with pytest.raises(GateError, match="missing"):
        require_gate("G0", tmp_path)


def test_gate_evidence_survives_repository_relocation(tmp_path):
    from kvrefine.gates import require_gate, write_decision

    original = tmp_path / "original"
    evidence = original / "results" / "evidence" / "smoke.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"passed": true}\n')
    decision = original / "results" / "decisions" / "G0.json"
    write_decision("G0", "pass", [evidence], "smoke passed", ["G1"], decision)

    moved = tmp_path / "moved"
    shutil.move(str(original), str(moved))
    assert require_gate("G0", moved / "results" / "decisions")["status"] == "pass"
