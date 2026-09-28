from __future__ import annotations

import numpy as np

from kvrefine.native import NativePage


def _side(groups: int) -> NativePage:
    return NativePage(
        q=np.full(groups * 8, 0x88888888, dtype=np.uint32),
        scales=np.full(groups, 0x3F80, dtype=np.uint16),
        biases=np.zeros(groups, dtype=np.uint16),
        valid_count=groups * 64,
        abi_id="fixture-abi",
        identity="fixture-page",
    )


def test_modes_round_trip_with_final_single_group_restart():
    from kvrefine.codec import decode_page, encode_page

    words = np.concatenate([
        np.full(64, 0x4100, dtype=np.uint16),  # 8: tight
        np.full(64, 0x4110, dtype=np.uint16),  # 9: wide
        np.full(64, 0x42C8, dtype=np.uint16),  # 100: literal
    ])
    blob = encode_page(words, _side(3), tensor_id=5, page_id=7)
    assert blob[32] & 0x3F == 0x24
    np.testing.assert_array_equal(decode_page(blob, _side(3), tensor_id=5, page_id=7), words)


def test_literal_preserves_every_special_bf16_word():
    from kvrefine.codec import decode_page, encode_page

    all_words = np.arange(65536, dtype=np.uint16)
    for page_id in range(16):
        words = all_words[page_id * 4096 : (page_id + 1) * 4096]
        side = _side(64)
        blob = encode_page(words, side, tensor_id=0, page_id=page_id)
        np.testing.assert_array_equal(decode_page(blob, side, tensor_id=0, page_id=page_id), words)


def test_unaligned_nonempty_page_rejected():
    from kvrefine.codec import encode_page
    from kvrefine.format import FormatError

    with np.testing.assert_raises(FormatError):
        encode_page(np.zeros(65, dtype=np.uint16), _side(1), 0, 0)


def test_checked_interval_supports_negative_scale_and_rejects_zero_scale():
    from kvrefine.bits import order
    from kvrefine.codec import interval

    key = int(order(np.array([0x4100], dtype=np.uint16))[0])
    lower, upper, width = interval(8, 0x3F80, 0, 0)
    assert lower <= key <= upper and width < 16
    assert interval(8, 0xBF80, 0x4180, 1) is not None  # p=8 from -8+16
    assert interval(8, 0, 0, 0) is None


def test_scalar_and_vector_intervals_agree_on_boundaries():
    from kvrefine.codec import _interval_arrays, interval

    metadata = [0, 0x0001, 0x0080, 0x3C00, 0x3F80, 0x7F7F, 0x7F80, 0x8000, 0xBF80, 0xFF80]
    for scale in metadata:
        for bias in (0, 0x3F80, 0xBF80, 0x7F7F):
            codes = np.arange(16, dtype=np.uint8)
            scales = np.full(16, scale, dtype=np.uint16)
            biases = np.full(16, bias, dtype=np.uint16)
            for mode in (0, 1):
                low, high, widths, safe = _interval_arrays(codes, scales, biases, mode)
                for q in range(16):
                    expected = (int(low[q]), int(high[q]), int(widths[q])) if safe[q] else None
                    assert interval(q, scale, bias, mode) == expected


def test_literal_page_actual_bytes_match_independent_estimator():
    from kvrefine.codec import decode_page, encode_page
    from kvrefine.format import page_nbytes

    words = np.full(64, 0x42C8, dtype=np.uint16)
    blob = encode_page(words, _side(1), 2, 3)
    assert blob[32] & 3 == 2
    assert len(blob) == page_nbytes(64, [64 * 16]) == 176
    np.testing.assert_array_equal(decode_page(blob, _side(1), 2, 3), words)
