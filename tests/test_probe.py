from __future__ import annotations

import gc
import weakref

import numpy as np

from kvrefine.native import NativePage


def _page(layer: int, role: str, head: int):
    words = np.full(64, 0x4100, dtype=np.uint16)
    side = NativePage(
        q=np.full(8, 0x88888888, dtype=np.uint32),
        scales=np.array([0x3F80], dtype=np.uint16),
        biases=np.array([0], dtype=np.uint16),
        valid_count=64,
        abi_id="fixture",
        identity=f"{layer}:{role}:{head}",
    )
    return {"layer": layer, "role": role, "head": head, "page_id": 0, "words": words, "side": side, "predictor": words}


def test_probe_visits_all_layers_heads_roles():
    from kvrefine.probe import aggregate_pages

    pages = (_page(layer, role, head) for layer in range(28) for role in ("K", "V") for head in range(8))
    result = aggregate_pages(pages)
    assert result["streams"] == 28 * 2 * 8
    assert result["pages"] == 28 * 2 * 8
    assert result["values"] == 28 * 2 * 8 * 64
    assert result["roles"]["K"]["pages"] == result["roles"]["V"]["pages"] == 28 * 8
    assert len(result["layers"]) == 28


def test_probe_counts_native_side_information_and_page_overhead():
    from kvrefine.probe import measure_page

    item = _page(0, "K", 0)
    measured = measure_page(**item)
    assert measured["q_bytes"] == 32
    assert measured["scales_biases_bytes"] == 4
    assert measured["raw_bytes"] == 128
    assert measured["refinement_bytes"] == measured["refinement_payload_bytes"] + measured["metadata_bytes"]
    assert measured["shared_bytes"] == 32 + 4 + measured["refinement_bytes"]
    assert measured["independent_dual_palette_bytes"] == 32 + 4 + measured["palette_bytes"]


def test_probe_charges_tensor_container_and_offsets():
    from kvrefine.probe import aggregate_pages

    result = aggregate_pages([_page(0, "K", 0)], source_identity={"model_revision": "a" * 40, "prompt_sha256": "b" * 64})
    totals = result["totals"]
    assert totals["container_metadata_bytes"] > 16
    assert totals["refinement_bytes"] == totals["refinement_page_bytes"] + totals["container_metadata_bytes"]
    assert totals["shared_bytes"] == totals["q_bytes"] + totals["scales_biases_bytes"] + totals["refinement_bytes"]
    assert result["roles"]["K"]["shared_bytes"] < totals["shared_bytes"]


def test_probe_aggregation_does_not_keep_source_arrays():
    from kvrefine.probe import aggregate_pages

    refs = []

    def records():
        for head in range(3):
            item = _page(0, "K", head)
            refs.extend((weakref.ref(item["words"]), weakref.ref(item["side"].q)))
            yield item

    result = aggregate_pages(records())
    assert result["pages"] == 3
    gc.collect()
    assert all(ref() is None for ref in refs)
