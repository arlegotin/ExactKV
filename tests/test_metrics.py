from __future__ import annotations

import pytest


def test_ledger_counts_alias_once_and_real_mirror_twice():
    from kvrefine.metrics import ByteLedger

    ledger = ByteLedger()
    ledger.add("q-host", "codes", 1024, 512)
    ledger.add("q-host", "alias", 1024, 512)
    ledger.add("q-gpu", "mirror", 1024, 512)
    assert ledger.snapshot()["allocated_bytes"] == 2048
    assert ledger.snapshot()["logical_bytes"] == 1024
    with pytest.raises(ValueError, match="owner"):
        ledger.add("q-host", "codes", 2048, 512)


def test_cache_projection_includes_stage_tail_and_metadata():
    from kvrefine.metrics import cache_projection, kv_payload_bytes

    mib = 1024 * 1024
    assert kv_payload_bytes(2048) == 224 * mib
    assert kv_payload_bytes(256) == 28 * mib
    raw = kv_payload_bytes(16384)
    q = raw * 36 // 128
    assert q / raw == 0.28125
    projection = cache_projection(
        {"q_codes": q * 32 // 36, "scales_biases": q * 4 // 36, "refinement": raw // 2, "metadata": 0},
        stage_bytes=raw // 28,
        tail_bytes=kv_payload_bytes(256),
        stage_count=2,
    )
    assert projection["prompt_plus_stages_bytes"] == 1528 * mib
    assert projection["cache_peak_bytes"] == 1556 * mib
