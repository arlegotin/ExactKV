from __future__ import annotations

import numpy as np
import struct


def test_palette_preserves_all_bf16_patterns():
    from kvrefine.baselines import palette_decode, palette_encode

    all_words = np.arange(65536, dtype=np.uint16)
    for start in range(0, 65536, 4096):
        words = all_words[start : start + 4096]
        np.testing.assert_array_equal(palette_decode(palette_encode(words)), words)


def test_singleton_palette_has_zero_index_width():
    from kvrefine.baselines import palette_decode, palette_encode

    words = np.full(4096, 0x3F80, dtype=np.uint16)
    encoded = palette_encode(words)
    assert encoded[10] == 0  # width byte
    assert encoded[11] == 0  # palette mode
    np.testing.assert_array_equal(palette_decode(encoded), words)


def test_palette_golden_page_layout_for_metal_decoder():
    from kvrefine.baselines import palette_encode

    words = np.full(64, 0x3F80, dtype=np.uint16)
    expected = struct.pack("<4sHHHBBI", b"EKVP", 1, 64, 1, 0, 0, 68) + b"\x7f" + b"\0" * 71
    assert palette_encode(words) == expected


def test_palette_uses_raw_fallback_when_it_would_expand():
    from kvrefine.baselines import palette_decode, palette_encode

    words = np.arange(4096, dtype=np.uint16) * np.uint16(17)
    encoded = palette_encode(words)
    assert encoded[11] == 1
    assert len(encoded) <= 16 + words.nbytes + 4
    np.testing.assert_array_equal(palette_decode(encoded), words)


def test_zstd_frames_are_page_local_and_exact():
    from kvrefine.baselines import field_zstd_decode, field_zstd_encode

    first = np.full(4096, 0x3F80, dtype=np.uint16)
    second = np.arange(4096, dtype=np.uint16)
    np.testing.assert_array_equal(field_zstd_decode(field_zstd_encode(first)), first)
    np.testing.assert_array_equal(field_zstd_decode(field_zstd_encode(second)), second)


def test_xor_rejects_wrong_predictor():
    from kvrefine.baselines import BaselineError, xor_zstd_decode, xor_zstd_encode

    words = np.arange(128, dtype=np.uint16)
    predictor = words ^ np.uint16(0x100)
    encoded = xor_zstd_encode(words, predictor)
    np.testing.assert_array_equal(xor_zstd_decode(encoded, predictor), words)
    with np.testing.assert_raises(BaselineError):
        xor_zstd_decode(encoded, np.zeros_like(predictor))
