from __future__ import annotations

import numpy as np


def test_bf16_order_is_a_permutation_of_all_words():
    from kvrefine.bits import order, unorder

    words = np.arange(65536, dtype=np.uint16)
    np.testing.assert_array_equal(unorder(order(words)), words)
    assert len(np.unique(order(words))) == 65536


def test_rne_bf16_ties_and_signed_zero():
    from kvrefine.bits import rne_bf16

    values = np.array([0.0, -0.0, 1.00390625, 1.01171875, -1.00390625], dtype=np.float32)
    np.testing.assert_array_equal(rne_bf16(values), np.array([0x0000, 0x8000, 0x3F80, 0x3F82, 0xBF80], dtype=np.uint16))
