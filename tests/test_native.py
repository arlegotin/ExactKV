from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.metal
def test_bf16_bitview_preserves_words():
    from kvrefine.native import bf16_words, words_bf16

    source = np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC1], dtype=np.uint16)
    np.testing.assert_array_equal(bf16_words(words_bf16(source)), source)


@pytest.mark.metal
def test_native_nibble_order_and_original_tuple():
    import mlx.core as mx

    from kvrefine.native import export_page, quantize_kv

    increasing = np.tile(np.arange(16, dtype=np.float32), 4).reshape(1, 1, 1, 64)
    decreasing = np.tile(np.arange(15, -1, -1, dtype=np.float32), 4).reshape(1, 1, 1, 64)
    native = quantize_kv(mx.array(increasing, dtype=mx.bfloat16), mx.array(decreasing, dtype=mx.bfloat16), 256)
    key_page = export_page(native, "K", 0, 0, 1, "fixture")
    value_page = export_page(native, "V", 0, 0, 1, "fixture")
    assert int(key_page.q[0]) == 0x89ABCDEF
    assert int(value_page.q[0]) == 0x76543210
    assert key_page.valid_count == value_page.valid_count == 64
    assert key_page.scales[0] == 0xBF80  # BF16 -1, as stored by native affine MLX
    assert key_page.biases[0] == 0x4170  # BF16 15
    direct = mx.quantize(mx.array(increasing, dtype=mx.bfloat16), group_size=64, bits=4, mode="affine")
    np.testing.assert_array_equal(np.array(direct[0]), np.array(native.keys[0][..., :1, :]))
    reconstructed = mx.dequantize(*native.keys, group_size=64, bits=4)
    mx.eval(reconstructed)
    np.testing.assert_array_equal(np.array(reconstructed[..., :1, :].astype(mx.float32)), increasing)


@pytest.mark.metal
def test_native_capacity_and_metadata():
    import mlx.core as mx

    from kvrefine.native import native_abi, quantize_kv

    source = mx.array(np.arange(128, dtype=np.float32).reshape(1, 1, 1, 128), dtype=mx.bfloat16)
    native = quantize_kv(source, source * 0.5, 256)
    assert native.valid_tokens == 1
    assert native.capacity_tokens == 256
    assert native.keys[0].dtype == mx.uint32
    assert native.keys[0].shape == (1, 1, 256, 16)
    assert native.keys[1].dtype == native.keys[2].dtype == mx.bfloat16
    assert native.keys[1].shape == native.keys[2].shape == (1, 1, 256, 2)
    assert native.keys[0].nbytes + native.keys[1].nbytes + native.keys[2].nbytes == 256 * 72
    assert native_abi()["group_size"] == 64


@pytest.mark.metal
def test_doctor_smoke_runs_stock_bf16_and_native_q4_attention():
    from kvrefine.doctor import collect_environment

    report = collect_environment(smoke=True)
    assert report["smoke"]["bf16_shape"] == [1, 16, 1, 128]
    assert report["smoke"]["q4_shape"] == [1, 16, 1, 128]
    assert len(report["native_abi"]["cache_source_sha256"]) == 64
    assert report["critical_issues"] == []
