# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import pytest
import torch

from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions.blockwise_gemm_sm100 import (
    blockwise_gemm_sm100,
    blockwise_grouped_gemm_sm100,
)
from transformer_engine.pytorch.cpp_extensions.gemm import (
    general_grouped_gemm,
    general_gemm,
    _flatten_fp8_storage_to_2d,
)
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.storage.float8_blockwise_tensor_storage import (
    Float8BlockwiseQTensorStorage,
)
import transformer_engine_torch as tex


@pytest.mark.parametrize("num_gemms", [4])
@pytest.mark.parametrize("m", [2048])
@pytest.mark.parametrize("k", [7168])
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("accumulate", [True, False])
def test_blockwise_grouped_gemm_sm100(layout, num_gemms, m, k, n, accumulate):
    # we use force_pow_2_scales=True, so we can use cublas as a reference.
    quantizer_2d = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=2,
    )
    quantizer_1d = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=1,
    )

    m_splits = [m] * num_gemms
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    out_dtype = torch.float32 if accumulate else torch.bfloat16

    if layout == "TN":
        # weights
        a_list = [
            quantizer_2d.quantize(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
            for _ in range(num_gemms)
        ]
        # inputs
        b_list = [
            quantizer_1d.quantize(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
            for m in m_splits
        ]

        out_ref = [torch.empty(sum(m_splits), n, dtype=out_dtype, device="cuda")]
        out = [t.detach().clone() for t in out_ref]
    elif layout == "NN":
        # weights
        a_list = [
            quantizer_2d.quantize(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
            for _ in range(num_gemms)
        ]
        # grads
        b_list = [
            quantizer_1d.quantize(torch.randn(m, n, dtype=torch.bfloat16, device="cuda"))
            for m in m_splits
        ]

        out_ref = [torch.empty(sum(m_splits), k, dtype=out_dtype, device="cuda")]
        out = [t.detach().clone() for t in out_ref]
    elif layout == "NT":
        # inputs
        a_list = [
            quantizer_1d.quantize(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"))
            for m in m_splits
        ]
        # grads
        b_list = [
            quantizer_1d.quantize(torch.randn(m, n, dtype=torch.bfloat16, device="cuda"))
            for m in m_splits
        ]

        out_ref = [torch.empty(n, k, dtype=out_dtype, device="cuda") for _ in range(num_gemms)]
        out = [t.detach().clone() for t in out_ref]

    blockwise_grouped_gemm_sm100(
        b_list, transb, a_list, transa, out, TE_DType[out_dtype], m_splits, accumulate=accumulate
    )

    # since we don't initialize fp8 recipe here, so this function will dispatch to
    # the cublas implementation.
    general_grouped_gemm(
        a_list,
        b_list,
        out_ref,
        [None] * num_gemms,
        out_dtype,
        layout=layout,
        m_splits=m_splits,
        single_output=(layout != "NT"),
        accumulate=accumulate,
    )

    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Helper: reshape a 2D-quantized Float8BlockwiseQTensorStorage to 3D
# ---------------------------------------------------------------------------
def _reshape_storage_to_3d(storage, s, b):
    """Reshape 2D Float8BlockwiseQTensorStorage data to 3D for testing.

    rowwise_data  (s*b, k)  -> (s, b, k)
    columnwise_data (k, s*b) -> (k, s, b)
    Scales remain 2D (unchanged).
    """
    new = object.__new__(Float8BlockwiseQTensorStorage)
    new._fp8_dtype = storage._fp8_dtype
    new._quantizer = storage._quantizer
    new._is_2D_scaled = storage._is_2D_scaled
    new._rowwise_scale_inv = storage._rowwise_scale_inv
    new._columnwise_scale_inv = storage._columnwise_scale_inv

    new._rowwise_data = storage._rowwise_data
    if new._rowwise_data is not None:
        k = new._rowwise_data.shape[-1]
        new._rowwise_data = new._rowwise_data.view(s, b, k)

    new._columnwise_data = storage._columnwise_data
    if new._columnwise_data is not None:
        d0 = new._columnwise_data.shape[0]
        new._columnwise_data = new._columnwise_data.view(d0, s, b)

    return new


# ---------------------------------------------------------------------------
# Tests for _flatten_fp8_storage_to_2d
# ---------------------------------------------------------------------------
class TestFlattenFP8StorageTo2D:
    """Unit tests for _flatten_fp8_storage_to_2d helper function."""

    @staticmethod
    def _make_storage(rowwise_data, rowwise_scale, colwise_data, colwise_scale):
        return Float8BlockwiseQTensorStorage(
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale,
            columnwise_data=colwise_data,
            columnwise_scale_inv=colwise_scale,
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=None,
            is_2D_scaled=False,
        )

    def test_2d_noop(self):
        """2D storage should be returned with data pointers preserved."""
        row_data = torch.randn(32, 64, device="cuda").to(torch.float8_e4m3fn)
        row_scale = torch.ones(1, 32, device="cuda")
        storage = self._make_storage(row_data, row_scale, None, None)

        result = _flatten_fp8_storage_to_2d(storage)
        assert result._rowwise_data.shape == (32, 64)
        assert result._rowwise_data.data_ptr() == row_data.data_ptr()

    def test_3d_rowwise_flattened(self):
        """3D rowwise data (s, b, k) -> (s*b, k)."""
        s, b, k = 4, 8, 64
        row_data = torch.randn(s, b, k, device="cuda").to(torch.float8_e4m3fn)
        row_scale = torch.ones(1, s * b, device="cuda")
        storage = self._make_storage(row_data, row_scale, None, None)

        result = _flatten_fp8_storage_to_2d(storage)
        assert result._rowwise_data.shape == (s * b, k)
        torch.testing.assert_close(
            result._rowwise_data, row_data.reshape(s * b, k)
        )

    def test_3d_columnwise_flattened(self):
        """3D columnwise data (k, s, b) -> (k, s*b)."""
        s, b, k = 4, 8, 64
        col_data = torch.randn(k, s, b, device="cuda").to(torch.float8_e4m3fn)
        col_scale = torch.ones(1, s * b, device="cuda")
        storage = self._make_storage(None, None, col_data, col_scale)

        result = _flatten_fp8_storage_to_2d(storage)
        assert result._columnwise_data.shape == (k, s * b)
        torch.testing.assert_close(
            result._columnwise_data, col_data.reshape(k, s * b)
        )

    def test_scales_unchanged(self):
        """Scales (already 2D) should remain unchanged after flattening."""
        s, b, k = 4, 8, 128
        row_data = torch.randn(s, b, k, device="cuda").to(torch.float8_e4m3fn)
        row_scale = torch.ones(1, s * b, device="cuda")
        col_data = torch.randn(k, s, b, device="cuda").to(torch.float8_e4m3fn)
        col_scale = torch.ones(1, s * b, device="cuda")
        storage = self._make_storage(row_data, row_scale, col_data, col_scale)

        result = _flatten_fp8_storage_to_2d(storage)
        # Scales are shared references, not copies
        assert result._rowwise_scale_inv.data_ptr() == row_scale.data_ptr()
        assert result._columnwise_scale_inv.data_ptr() == col_scale.data_ptr()

    def test_original_unchanged(self):
        """Original storage should not be modified."""
        s, b, k = 4, 8, 64
        row_data = torch.randn(s, b, k, device="cuda").to(torch.float8_e4m3fn)
        row_scale = torch.ones(1, s * b, device="cuda")
        storage = self._make_storage(row_data, row_scale, None, None)

        _flatten_fp8_storage_to_2d(storage)
        # Original data should still be 3D
        assert storage._rowwise_data.shape == (s, b, k)


# ---------------------------------------------------------------------------
# End-to-end tests: general_gemm with 3D FP8 blockwise input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("s,b", [(1, 32), (4, 8), (16, 1)])
@pytest.mark.parametrize("k,n", [(256, 512)])
@pytest.mark.parametrize("accumulate", [False, True])
def test_general_gemm_3d_blockwise(layout, s, b, k, n, accumulate):
    """general_gemm with 3D FP8 blockwise input should match 2D reference.

    Covers all GEMM operations used in attention and MLP linear layers:
      - fprop (TN): linear_qkv, linear_proj, linear_fc1, linear_fc2
      - dgrad (NN): backward input gradient for all linear layers
      - wgrad (NT): backward weight gradient for all linear layers

    Strategy:
      1. Quantize 2D tensors and run general_gemm as reference.
      2. Reshape quantized storage data to 3D.
      3. Run general_gemm with 3D input.
      4. Verify output shape and numerical equivalence.
    """
    m = s * b
    out_dtype = torch.float32 if accumulate else torch.bfloat16

    quantizer_2d = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=2,
    )
    quantizer_1d = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=1,
    )

    if layout == "TN":
        # fprop: A=weight(2D, n x k), B=input(3D, s x b x k)
        weight = quantizer_2d.quantize(
            torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        )
        input_2d = quantizer_1d.quantize(
            torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        )

        out_2d = torch.randn(m, n, dtype=out_dtype, device="cuda") if accumulate else None
        out_3d_pre = out_2d.view(s, b, n).clone() if accumulate else None

        # 2D reference
        out_ref, _, _, _ = general_gemm(
            weight, input_2d, out_dtype=out_dtype, layout="TN",
            accumulate=accumulate, out=out_2d,
        )
        # 3D test
        input_3d = _reshape_storage_to_3d(input_2d, s, b)
        out_3d, _, _, _ = general_gemm(
            weight, input_3d, out_dtype=out_dtype, layout="TN",
            accumulate=accumulate, out=out_3d_pre,
        )

        assert out_3d.shape == (s, b, n), (
            f"fprop: expected shape ({s}, {b}, {n}), got {out_3d.shape}"
        )
        torch.testing.assert_close(out_3d.view(m, n), out_ref, atol=0, rtol=0)

    elif layout == "NN":
        # dgrad: A=weight(2D, n x k), B=grad_output(3D, s x b x n)
        weight = quantizer_2d.quantize(
            torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        )
        grad_2d = quantizer_1d.quantize(
            torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        )

        out_2d = torch.randn(m, k, dtype=out_dtype, device="cuda") if accumulate else None
        out_3d_pre = out_2d.view(s, b, k).clone() if accumulate else None

        # 2D reference
        out_ref, _, _, _ = general_gemm(
            weight, grad_2d, out_dtype=out_dtype, layout="NN",
            accumulate=accumulate, out=out_2d,
        )
        # 3D test
        grad_3d = _reshape_storage_to_3d(grad_2d, s, b)
        out_3d, _, _, _ = general_gemm(
            weight, grad_3d, out_dtype=out_dtype, layout="NN",
            accumulate=accumulate, out=out_3d_pre,
        )

        assert out_3d.shape == (s, b, k), (
            f"dgrad: expected shape ({s}, {b}, {k}), got {out_3d.shape}"
        )
        torch.testing.assert_close(out_3d.view(m, k), out_ref, atol=0, rtol=0)

    elif layout == "NT":
        # wgrad: A=input(3D), B=grad_output(3D) -> output is 2D weight gradient
        input_2d = quantizer_1d.quantize(
            torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        )
        grad_2d = quantizer_1d.quantize(
            torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        )

        out_2d = torch.randn(n, k, dtype=out_dtype, device="cuda") if accumulate else None
        out_wgrad = out_2d.clone() if accumulate else None

        # 2D reference
        out_ref, _, _, _ = general_gemm(
            input_2d, grad_2d, out_dtype=out_dtype, layout="NT", grad=True,
            accumulate=accumulate, out=out_2d,
        )
        # 3D test
        input_3d = _reshape_storage_to_3d(input_2d, s, b)
        grad_3d = _reshape_storage_to_3d(grad_2d, s, b)
        out_3d, _, _, _ = general_gemm(
            input_3d, grad_3d, out_dtype=out_dtype, layout="NT", grad=True,
            accumulate=accumulate, out=out_wgrad,
        )

        assert out_3d.shape == (n, k), (
            f"wgrad: expected 2D shape ({n}, {k}), got {out_3d.shape}"
        )
        torch.testing.assert_close(out_3d, out_ref, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Numerical correctness: 3D FP8 blockwise GEMM vs BF16 reference
# ---------------------------------------------------------------------------
def _assert_cosine_similarity(actual, expected, min_cos_sim=0.99):
    """Assert cosine similarity between two tensors exceeds threshold.

    More robust than atol/rtol for FP8 vs BF16 comparison because FP8 E4M3
    (3-bit mantissa) introduces significant per-element quantization noise,
    but the overall direction of the output vector should be preserved.
    """
    a = actual.flatten().float()
    b = expected.flatten().float()
    cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    assert cos_sim >= min_cos_sim, (
        f"Cosine similarity {cos_sim:.6f} < {min_cos_sim} threshold"
    )


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("s,b", [(4, 8)])
@pytest.mark.parametrize("k,n", [(256, 512)])
def test_general_gemm_3d_blockwise_vs_bf16(layout, s, b, k, n):
    """3D FP8 blockwise GEMM result should be numerically close to BF16 matmul.

    This test verifies that the 3D handling does not introduce numerical
    errors beyond the expected FP8 quantization noise. Uses cosine similarity
    since FP8 E4M3 (3-bit mantissa) has large per-element error but preserves
    the overall output direction.
    """
    m = s * b

    quantizer_2d = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=2,
    )
    quantizer_1d = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=1,
    )

    if layout == "TN":
        # fprop: C = A^T @ B  where A=(n,k), B=(m,k) -> C=(m,n)
        a_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        ref = b_bf16 @ a_bf16.T  # (m, n)

        a_q = quantizer_2d.quantize(a_bf16)
        b_q = quantizer_1d.quantize(b_bf16)
        b_3d = _reshape_storage_to_3d(b_q, s, b)
        out, _, _, _ = general_gemm(a_q, b_3d, out_dtype=torch.bfloat16, layout="TN")

        assert out.shape == (s, b, n)
        _assert_cosine_similarity(out.view(m, n), ref)

    elif layout == "NN":
        # dgrad: C = A @ B  where A=(n,k), B=(m,n) -> C=(m,k)
        a_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        ref = b_bf16 @ a_bf16  # (m, k)

        a_q = quantizer_2d.quantize(a_bf16)
        b_q = quantizer_1d.quantize(b_bf16)
        b_3d = _reshape_storage_to_3d(b_q, s, b)
        out, _, _, _ = general_gemm(a_q, b_3d, out_dtype=torch.bfloat16, layout="NN")

        assert out.shape == (s, b, k)
        _assert_cosine_similarity(out.view(m, k), ref)

    elif layout == "NT":
        # wgrad: C = A @ B^T  where A=(m,k), B=(m,n) -> C=(n,k)
        a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        ref = b_bf16.T @ a_bf16  # (n, k)

        a_q = quantizer_1d.quantize(a_bf16)
        b_q = quantizer_1d.quantize(b_bf16)
        a_3d = _reshape_storage_to_3d(a_q, s, b)
        b_3d = _reshape_storage_to_3d(b_q, s, b)
        out, _, _, _ = general_gemm(
            a_3d, b_3d, out_dtype=torch.bfloat16, layout="NT", grad=True,
        )

        assert out.shape == (n, k)
        _assert_cosine_similarity(out, ref)
