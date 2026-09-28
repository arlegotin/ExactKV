from __future__ import annotations

import pytest


def test_page_size_estimator_includes_header_mode_offsets_and_guard():
    from kvrefine.format import page_nbytes

    assert page_nbytes(192, [2048, 1024]) == 436
    assert page_nbytes(64, [0]) == 48
    with pytest.raises(ValueError):
        page_nbytes(65, [0])


def test_container_round_trip_and_identity():
    from kvrefine.format import FormatError, pack_container, unpack_container

    manifest = {"model_revision": "a" * 40, "abi_id": "test", "tensors": [{"role": "K", "head": 0}]}
    pages = [b"EKVR" + b"\x00" * 44, b"EKVR" + b"\x01" * 44]
    blob = pack_container(manifest, pages)
    recovered, views = unpack_container(blob)
    assert recovered["model_revision"] == manifest["model_revision"]
    assert recovered["page_offsets"] == [0, 48, 96]
    assert [bytes(view) for view in views] == pages
    with pytest.raises(FormatError):
        unpack_container(blob[:-1])
