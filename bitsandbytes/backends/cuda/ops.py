'''
from collections.abc import Sequence
import ctypes as ct
from math import prod
from typing import Optional

import torch

from bitsandbytes.functional import CUBLAS_Context, _cuda_device_of, _get_tensor_stream, get_ptr

from ..._ops import register_kernel
from ...cextension import lib


@register_kernel("bitsandbytes::int8_linear_matmul", "cuda")
def _(A: torch.Tensor, B: torch.Tensor):
    out = torch.empty((*A.shape[:-1], B.shape[0]), device=A.device, dtype=torch.int32)
    return _int8_linear_matmul_impl(A, B, out)


@register_kernel("bitsandbytes::int8_linear_matmul.out", "cuda")
def _(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    _int8_linear_matmul_impl(A, B, out)


def _int8_linear_matmul_impl(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    A, B = B, A

    shapeA = A.shape
    shapeB = B.shape

    torch._check(A.dtype == torch.int8, lambda: "B must be int8")
    torch._check(B.dtype == torch.int8, lambda: "A must be int8")
    torch._check(A.ndim == 2, lambda: "Only two dimensional matrices are supported for argument B")
    torch._check(B.ndim in [2, 3], lambda: "Only two or three dimensional matrices are supported for argument A")
    torch._check(prod(shapeB) > 0, lambda: f"Input tensor dimensions need to be > 0: {shapeB}")
    torch._check(out.dtype == torch.int32)

    shapeC = (*shapeB[:-1], shapeA[0])
    torch._check(out.shape == shapeC, lambda: f"Output shape {out.shape} does not match expected shape {shapeC}")

    k, m = shapeA
    n = prod(shapeB[:-1])
    lda = shapeA[-1]  # Weights (outputs, inputs)
    ldb = shapeB[-1]  # Activations (batch, tokens, inputs)
    ldc = shapeC[-1]  # Output (batch, tokens, outputs)

    torch._check(
        lda == ldb,
        lambda: f"int8_linear_matmul only supports B^T @ A. Inner dimensions do not match: B @ A = {shapeB} @ {shapeA}",
    )

    # cuBLASLt does not support int8 matmul with inner dimensions that are not divisible by 4.
    # We'll fall back to a slower fp32 calculation in this circumstance.
    # Fortunately, this should not be very common.
    if lda % 4 != 0:
        result = torch.matmul(B.float(), A.float().t()).to(torch.int32)
        return out.copy_(result)

    with _cuda_device_of(A):
        ctx = CUBLAS_Context.get_instance().get_context(A.device)
        ptrA = get_ptr(A)
        ptrB = get_ptr(B)
        ptrC = get_ptr(out)
        ptrRowScale = None
        m = ct.c_int32(m)
        n = ct.c_int32(n)
        k = ct.c_int32(k)
        lda = ct.c_int32(lda)
        ldb = ct.c_int32(ldb)
        ldc = ct.c_int32(ldc)
        stream = _get_tensor_stream(A)

        has_error = lib.cigemmlt_32(ctx, m, n, k, ptrA, ptrB, ptrC, ptrRowScale, lda, ldb, ldc, stream)

    if has_error:
        if has_error == 100:
            # `ERR_NOT_IMPLEMENTED` is defined as 100 in `ops.cu`
            # TODO: Warn and implement a fallback to fp32 compute?
            raise NotImplementedError("int8_linear_matmul not implemented!")
        else:
            raise RuntimeError(
                f"cublasLt ran into an error!\n\t{shapeA=}, {shapeB=}, {shapeC=}\n\t{(lda, ldb, ldc)=}\n\t{(m, n, k)=}"
            )

    return out


@register_kernel("bitsandbytes::int8_mm_dequant", "cuda")
def _(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch._check(A.dtype == torch.int32, lambda: f"A must be int32, got {A.dtype}")
    torch._check(row_stats.dtype == torch.float32, lambda: f"row_stats must be float32, got {row_stats.dtype}")
    torch._check(col_stats.dtype == torch.float32, lambda: f"col_stats must be float32, got {col_stats.dtype}")

    # Note: cuda kernel only currently supports fp16 output.
    # We'll later cast to desired dtype if needed.
    out = torch.empty_like(A, dtype=torch.float16)

    ptrA = get_ptr(A)
    ptrOut = get_ptr(out)
    ptrRowStats = get_ptr(row_stats)
    ptrColStats = get_ptr(col_stats)
    numRows = ct.c_int32(prod(A.shape[:-1]))
    numCols = ct.c_int32(A.shape[-1])

    # Note: fused bias in the kernel is only supported for fp16
    # TODO(matthewdouglas): Consider supporting bf16 fused bias
    ptrBias = get_ptr(bias) if bias is not None and bias.dtype == torch.float16 else None

    with _cuda_device_of(A):
        lib.cdequant_mm_int32_fp16(
            ptrA, ptrRowStats, ptrColStats, ptrOut, ptrBias, numRows, numCols, _get_tensor_stream(A)
        )

    # Add bias separately if not fused in kernel
    if bias is not None and bias.dtype != torch.float16:
        out.add_(bias)

    return out.to(dtype or torch.float16)


@register_kernel("bitsandbytes::int8_vectorwise_quant", "cuda")
def _(A: torch.Tensor, threshold=0.0):
    torch._check(A.dtype == torch.float16, lambda: f"A must be float16, got {A.dtype}")
    torch._check(threshold >= 0.0, lambda: "threshold must be non-negative")

    rows = prod(A.shape[:-1])
    cols = A.shape[-1]

    row_stats = torch.empty(rows, device=A.device, dtype=torch.float32)
    out_row = torch.empty(A.shape, device=A.device, dtype=torch.int8)

    outlier_cols = None

    if threshold > 0.0:
        # TODO we could improve perf of this
        outliers = A.abs() >= threshold

        if outliers.any():
            outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)
        else:
            # Needed for torch.compile support.
            outlier_cols = torch.empty(0, device=A.device, dtype=torch.int64)

    with _cuda_device_of(A):
        lib.cint8_vector_quant(
            get_ptr(A),
            get_ptr(out_row),
            get_ptr(row_stats),
            ct.c_float(threshold),
            ct.c_int32(rows),
            ct.c_int32(cols),
            _get_tensor_stream(A),
        )

    # Zero out values from outlier columns across all rows.
    # The kernel will handle this for outliers themselves, so we can optimize for rows=1.
    if rows > 1 and outlier_cols is not None:
        out_row[:, outlier_cols] = 0

    return out_row, row_stats, outlier_cols


@register_kernel("bitsandbytes::int8_double_quant", "cuda")
def _(
    A: torch.Tensor,
    threshold=0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # Use CUDA kernel for rowwise and COO tensor
    quant_row, row_stats, outlier_cols = torch.ops.bitsandbytes.int8_vectorwise_quant.default(
        A,
        threshold=threshold,
    )

    # PyTorch impl for colwise
    col_stats, outlier_mask = _get_col_absmax(A, threshold=threshold)
    if threshold > 0.0 and outlier_mask is not None:
        A = A.masked_fill(outlier_mask, 0.0)
    quant_col = torch.round(A.mul(127.0) / col_stats.unsqueeze(0)).to(torch.int8)

    return quant_row, quant_col, row_stats, col_stats.flatten().float(), outlier_cols


def _get_col_absmax(
    A: torch.Tensor,
    threshold=0.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    torch._check(A.is_floating_point())

    outlier_mask = None

    absA = A.abs().view(-1, A.shape[-1])

    if threshold > 0.0:
        # Filter outliers from stats when enabled
        outlier_mask = absA >= threshold
        absA.masked_fill_(outlier_mask, 0.0)

    # shape [cols]; unsqueeze(0) gives [1,cols]
    col_stats = absA.amax(dim=0, keepdim=False).float()

    return col_stats, outlier_mask


@register_kernel("bitsandbytes::quantize_blockwise", "cuda")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    torch._check(code.dtype == torch.float32, lambda: f"code must be float32, got {code.dtype}")

    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty_like(A, dtype=torch.uint8)

    with _cuda_device_of(A):
        args = (
            get_ptr(code),
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int32(blocksize),
            ct.c_int(A.numel()),
        )

        if A.dtype == torch.float16:
            lib.cquantize_blockwise_fp16(*args)
        elif A.dtype == torch.bfloat16:
            lib.cquantize_blockwise_bf16(*args)
        elif A.dtype == torch.float32:
            lib.cquantize_blockwise_fp32(*args)
        else:
            raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")

    return out, absmax


@register_kernel("bitsandbytes::dequantize_blockwise", "cuda")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    out = torch.empty_like(A, dtype=dtype)
    _dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out=out)
    return out


@register_kernel("bitsandbytes::dequantize_blockwise.out", "cuda")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
    torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
    _dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out=out)


def _dequantize_blockwise_impl(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype, out: torch.Tensor
) -> None:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(
        dtype in [torch.float16, torch.bfloat16, torch.float32],
        lambda: f"Blockwise dequantization only supports 16bit/32bit floating types, got {dtype}",
    )

    with _cuda_device_of(A):
        args = (
            get_ptr(code),
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int(blocksize),
            ct.c_int(A.numel()),
            _get_tensor_stream(A),
        )

        if dtype == torch.float16:
            lib.cdequantize_blockwise_fp16(*args)
        elif dtype == torch.bfloat16:
            lib.cdequantize_blockwise_bf16(*args)
        elif dtype == torch.float32:
            lib.cdequantize_blockwise_fp32(*args)


@register_kernel("bitsandbytes::quantize_4bit", "cuda")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    torch._check(quant_type in ["fp4", "nf4"])
    torch._check(
        A.dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}",
    )

    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty(((n + 1) // (quant_storage.itemsize * 2), 1), device=A.device, dtype=quant_storage)

    with _cuda_device_of(A):
        args = (
            None,
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int32(blocksize),
            ct.c_int(n),
        )

        if A.dtype == torch.bfloat16:
            if quant_type == "fp4":
                lib.cquantize_blockwise_bf16_fp4(*args)
            else:
                lib.cquantize_blockwise_bf16_nf4(*args)
        elif A.dtype == torch.float16:
            if quant_type == "fp4":
                lib.cquantize_blockwise_fp16_fp4(*args)
            else:
                lib.cquantize_blockwise_fp16_nf4(*args)
        elif A.dtype == torch.float32:
            if quant_type == "fp4":
                lib.cquantize_blockwise_fp32_fp4(*args)
            else:
                lib.cquantize_blockwise_fp32_nf4(*args)

    return out, absmax


@register_kernel("bitsandbytes::dequantize_4bit", "cuda")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty(shape, dtype=dtype, device=A.device)
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)
    return out


@register_kernel("bitsandbytes::dequantize_4bit.out", "cuda")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(out.shape == shape, lambda: f"Expected out.shape == {shape}, got {out.shape}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)


def _dequantize_4bit_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(blocksize in [4096, 2048, 1024, 512, 256, 128, 64])
    torch._check(quant_type in ["fp4", "nf4"])
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )

    with _cuda_device_of(A):
        args = (
            None,
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int(blocksize),
            ct.c_int(out.numel()),
            _get_tensor_stream(A),
        )

        if out.dtype == torch.bfloat16:
            if quant_type == "fp4":
                lib.cdequantize_blockwise_bf16_fp4(*args)
            else:
                lib.cdequantize_blockwise_bf16_nf4(*args)
        elif out.dtype == torch.float16:
            if quant_type == "fp4":
                lib.cdequantize_blockwise_fp16_fp4(*args)
            else:
                lib.cdequantize_blockwise_fp16_nf4(*args)
        elif out.dtype == torch.float32:
            if quant_type == "fp4":
                lib.cdequantize_blockwise_fp32_fp4(*args)
            else:
                lib.cdequantize_blockwise_fp32_nf4(*args)


@register_kernel("bitsandbytes::gemv_4bit", "cuda")
def _(
    A: torch.Tensor, B: torch.Tensor, shapeB: Sequence[int], absmax: torch.Tensor, code: torch.Tensor, blocksize: int
) -> torch.Tensor:
    shape = (*A.shape[:-1], shapeB[0])
    out = torch.empty(shape, device=A.device, dtype=A.dtype)
    _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out=out)
    return out


@register_kernel("bitsandbytes::gemv_4bit.out", "cuda")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
    torch._check(
        out.shape == (*A.shape[:-1], shapeB[0]),
        lambda: f"Expected out.shape == {(*A.shape[:-1], shapeB[0])}, got {out.shape}",
    )
    torch._check(out.dtype == A.dtype, lambda: f"Expected out.dtype == {A.dtype}, got {out.dtype}")
    _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out=out)


def _gemv_4bit_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
    torch._check_is_size(blocksize)
    torch._check(
        A.numel() == A.size(-1),
        lambda: f"A must be a vector with leading dimensions of 1, got {A.shape}",
    )
    torch._check(
        A.dtype in [torch.float16, torch.bfloat16, torch.float32],
        lambda: f"A must be float16, bfloat16, or float32, got {A.dtype}",
    )
    torch._check(
        B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32],
        lambda: f"B must be backed by storage of type uint8, bfloat16, float16, or float32, got {B.dtype}",
    )
    torch._check(absmax.dtype == torch.float32, lambda: f"absmax must be float32, got {absmax.dtype}")
    torch._check(code.dtype == torch.float32, lambda: f"code must be float32, got {code.dtype}")

    m = ct.c_int32(shapeB[0])
    n = ct.c_int32(1)
    k = ct.c_int32(shapeB[1])

    lda = m
    ldb = ct.c_int32((A.shape[-1] + 1) // 2)
    ldc = m

    stream = _get_tensor_stream(A)

    with _cuda_device_of(A):
        if A.dtype == torch.float16:
            lib.cgemm_4bit_inference_naive_fp16(
                m,
                n,
                k,
                get_ptr(A),
                get_ptr(B),
                get_ptr(absmax),
                get_ptr(code),
                get_ptr(out),
                lda,
                ldb,
                ldc,
                ct.c_int32(blocksize),
                stream,
            )
        elif A.dtype == torch.bfloat16:
            lib.cgemm_4bit_inference_naive_bf16(
                m,
                n,
                k,
                get_ptr(A),
                get_ptr(B),
                get_ptr(absmax),
                get_ptr(code),
                get_ptr(out),
                lda,
                ldb,
                ldc,
                ct.c_int32(blocksize),
                stream,
            )
        elif A.dtype == torch.float32:
            lib.cgemm_4bit_inference_naive_fp32(
                m,
                n,
                k,
                get_ptr(A),
                get_ptr(B),
                get_ptr(absmax),
                get_ptr(code),
                get_ptr(out),
                lda,
                ldb,
                ldc,
                ct.c_int32(blocksize),
                stream,
            )
'''

from collections.abc import Sequence
import ctypes as ct
from math import prod
from typing import Optional

import torch
import logging 

from bitsandbytes.functional import (
    CUBLAS_Context,
    _cuda_device_of,
    get_tensor_stream as _get_tensor_stream, 
    get_ptr
)

from bitsandbytes.cextension import HIP_ENVIRONMENT, lib

from bitsandbytes._ops import register_kernel

logger = logging.getLogger(__name__)

@register_kernel("bitsandbytes::int8_linear_matmul", "cuda")
def _int8_linear_matmul_entry(A: torch.Tensor, B: torch.Tensor): 

    out = torch.empty((*A.shape[:-1], B.shape[0]), device=A.device, dtype=torch.int32)
    _int8_linear_matmul_impl(A, B, out) 
    return out

@register_kernel("bitsandbytes::int8_linear_matmul.out", "cuda")
def _int8_linear_matmul_out_entry(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor): 
    _int8_linear_matmul_impl(A, B, out)


def _int8_linear_matmul_impl(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    """
    Implementation for int8_linear_matmul (A @ B.T).
    A: Activations (e.g., [batch, tokens, features_in])
    B: Weights (e.g., [features_out, features_in])
    out: Output tensor (e.g., [batch, tokens, features_out])
    """
    shapeA_orig = A.shape
    shapeB_orig = B.shape

    if A.ndim == 3:
        A_reshaped = A.reshape(-1, A.shape[-1])
    elif A.ndim == 2:
        A_reshaped = A
    else:
        raise ValueError(f"Input A must be 2D or 3D, got {A.ndim}D")
    
    shapeA_reshaped = A_reshaped.shape # [M, K]
    shapeB_weights = B.shape # [N, K]

    torch._check(A.dtype == torch.int8, lambda: f"A must be int8, got {A.dtype}")
    torch._check(B.dtype == torch.int8, lambda: f"B must be int8, got {B.dtype}")
    torch._check(B.ndim == 2, lambda: f"B (weights) must be 2D, got {B.ndim}")
    torch._check(prod(shapeA_reshaped) > 0, lambda: f"Input A dimensions need to be > 0: {shapeA_orig}")
    torch._check(prod(shapeB_weights) > 0, lambda: f"Input B dimensions need to be > 0: {shapeB_orig}")

    expected_out_shape_2d = (shapeA_reshaped[0], shapeB_weights[0])
    
    if A.ndim == 3:
        expected_out_shape_orig = (*A.shape[:-1], shapeB_weights[0])
    else: # A.ndim == 2
        expected_out_shape_orig = (A.shape[0], shapeB_weights[0])

    torch._check(out.dtype == torch.int32, lambda: f"Output tensor dtype must be int32, got {out.dtype}")
    torch._check(out.shape == expected_out_shape_orig, 
                 lambda: f"Output shape {out.shape} does not match expected {expected_out_shape_orig}")

    m_act = shapeA_reshaped[0] 
    k_act = shapeA_reshaped[1] 
    n_weights = shapeB_weights[0] 
    k_weights = shapeB_weights[1]

    torch._check(k_act == k_weights, 
                 lambda: f"Inner dimensions for A @ B.T do not match: A_cols={k_act}, B_cols={k_weights}")

    if k_act % 4 != 0:
        logger.warning(f"int8_linear_matmul: Inner dimension ({k_act}) not divisible by 4. "
                       "Falling back to slower fp32 calculation. This will impact performance.")
        result_fp32 = torch.matmul(A.float(), B.float().t()).to(torch.int32)
        out.copy_(result_fp32)
        return

    out_reshaped = out.reshape(-1, n_weights) if out.shape != expected_out_shape_2d else out

    with _cuda_device_of(A):
        ctx = CUBLAS_Context.get_instance().get_context(A.device)

        m_ffi = ct.c_int32(n_weights)  
        n_ffi = ct.c_int32(m_act)      
        k_ffi = ct.c_int32(k_act)     

        lda_ffi = ct.c_int32(B.stride(0) if B.is_contiguous() else k_act) 
        ldb_ffi = ct.c_int32(A_reshaped.stride(0) if A_reshaped.is_contiguous() else k_act) 
        ldc_ffi = ct.c_int32(out_reshaped.stride(0) if out_reshaped.is_contiguous() else n_weights) 

        ptrA_arg_kernel = get_ptr(B) 
        ptrB_arg_kernel = get_ptr(A_reshaped) 
        ptrC_out_kernel = get_ptr(out_reshaped)
        ptrRowScale = None 

        stream = _get_tensor_stream(A)

        has_error = lib.cigemmlt_32(
            ctx, m_ffi, n_ffi, k_ffi, 
            ptrA_arg_kernel, ptrB_arg_kernel, ptrC_out_kernel,
            ptrRowScale, 
            lda_ffi, ldb_ffi, ldc_ffi, 
            stream
        )

        if has_error:
            if has_error == 100: # ERR_NOT_IMPLEMENTED
                err_msg = f"int8_linear_matmul not implemented for these dimensions on {A.device.type}."
                backend_specific_msg = " This might be due to hipBLASLt limitations or missing kernel." if HIP_ENVIRONMENT \
                                  else " This might be due to cuBLASLt limitations or missing kernel."
                raise NotImplementedError(err_msg + backend_specific_msg)
            else:
                raise RuntimeError(
                    f"bitsandbytes int8_linear_matmul kernel failed with error code {has_error}!\n"
                    f"\tShapes: A_orig={shapeA_orig}, B_orig={shapeB_orig}, Out_orig={out.shape}\n"
                    f"\tKernel Dims (m,n,k for W@A.T): ({m_ffi.value},{n_ffi.value},{k_ffi.value})\n"
                    f"\tKernel LDs (ldW,ldA,ldOut): ({lda_ffi.value},{ldb_ffi.value},{ldc_ffi.value})"
                )

@register_kernel("bitsandbytes::int8_mm_dequant", "cuda")
def _int8_mm_dequant_entry( # Renamed
    A: torch.Tensor, 
    row_stats: torch.Tensor, 
    col_stats: torch.Tensor, 
    dtype: Optional[torch.dtype] = None, 
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch._check(A.dtype == torch.int32, lambda: f"A must be int32, got {A.dtype}")
    torch._check(row_stats.dtype == torch.float32, lambda: f"row_stats must be float32, got {row_stats.dtype}")
    torch._check(col_stats.dtype == torch.float32, lambda: f"col_stats must be float32, got {col_stats.dtype}")

    kernel_output_dtype = torch.float16 # C kernel cdequant_mm_int32_fp16 outputs fp16
    final_output_dtype = dtype if dtype is not None else kernel_output_dtype

    out_kernel = torch.empty_like(A, dtype=kernel_output_dtype)

    ptrA_kernel = get_ptr(A)
    ptrOut_kernel = get_ptr(out_kernel)
    ptrRowStats_kernel = get_ptr(row_stats)
    ptrColStats_kernel = get_ptr(col_stats)
    
    numRows_kernel = ct.c_int32(prod(A.shape[:-1]))
    numCols_kernel = ct.c_int32(A.shape[-1])

    ptrBias_kernel = None
    can_fuse_bias = (bias is not None and bias.dtype == kernel_output_dtype)
    if can_fuse_bias:
        ptrBias_kernel = get_ptr(bias)

    with _cuda_device_of(A):
        stream = _get_tensor_stream(A)
        lib.cdequant_mm_int32_fp16(
            ptrA_kernel, ptrRowStats_kernel, ptrColStats_kernel, ptrOut_kernel, ptrBias_kernel, 
            numRows_kernel, numCols_kernel, stream
        )

    result = out_kernel
    if bias is not None and not can_fuse_bias:
        result = result.add_(bias.to(result.dtype))

    return result.to(final_output_dtype)

@register_kernel("bitsandbytes::int8_vectorwise_quant", "cuda")
def _int8_vectorwise_quant_entry(A: torch.Tensor, threshold: float = 0.0): # Renamed
    torch._check(A.dtype == torch.float16, lambda: f"A must be float16, got {A.dtype}")
    torch._check(threshold >= 0.0, lambda: f"threshold must be non-negative, got {threshold}")

    rows = prod(A.shape[:-1])
    cols = A.shape[-1]

    row_stats_out = torch.empty(rows, device=A.device, dtype=torch.float32)
    quantized_data_out = torch.empty(A.shape, device=A.device, dtype=torch.int8)
    outlier_col_indices_out: Optional[torch.Tensor] = None

    if threshold > 0.0:
        outliers_mask = A.abs() >= threshold
        if outliers_mask.any():
            outlier_col_indices_out = torch.argwhere(outliers_mask.any(dim=0)).view(-1)
        else:
            outlier_col_indices_out = torch.empty(0, device=A.device, dtype=torch.int64)
    
    with _cuda_device_of(A):
        stream = _get_tensor_stream(A)
        lib.cint8_vector_quant(
            get_ptr(A), get_ptr(quantized_data_out), get_ptr(row_stats_out),
            ct.c_float(threshold),
            ct.c_int32(rows), ct.c_int32(cols),
            stream
        )

    if rows > 1 and outlier_col_indices_out is not None and outlier_col_indices_out.numel() > 0 :
        quantized_data_out[:, outlier_col_indices_out] = 0
        
    return quantized_data_out, row_stats_out, outlier_col_indices_out

@register_kernel("bitsandbytes::quantize_blockwise", "cuda")
def _quantize_blockwise_entry(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]: 
    torch._check_is_size(blocksize)
    torch._check(code.dtype == torch.float32, lambda: f"code map must be float32, got {code.dtype}")

    supported_blocksizes_cuda = [4096, 2048, 1024, 512, 256, 128, 64]
    supported_blocksizes_hip = [4096, 2048, 1024, 512, 256, 128] 
    current_supported_blocksizes = supported_blocksizes_hip if HIP_ENVIRONMENT else supported_blocksizes_cuda
    torch._check(blocksize in current_supported_blocksizes, 
                 lambda: f"For {A.device.type}, blocksize must be one of {current_supported_blocksizes}, got {blocksize}")

    n = A.numel()
    num_blocks = (n + blocksize - 1) // blocksize

    absmax_out = torch.empty((num_blocks,), device=A.device, dtype=torch.float32)
    quantized_out = torch.empty_like(A, dtype=torch.uint8)

    with _cuda_device_of(A):
        stream = _get_tensor_stream(A)
    
        code_on_device = code.to(A.device) if code.device != A.device else code
        
        c_args = (
            get_ptr(code_on_device), get_ptr(A), get_ptr(absmax_out), get_ptr(quantized_out),
            ct.c_int32(blocksize), ct.c_int(n), stream 
        )
        if A.dtype == torch.float16: lib.cquantize_blockwise_fp16(*c_args)
        elif A.dtype == torch.bfloat16: lib.cquantize_blockwise_bf16(*c_args)
        elif A.dtype == torch.float32: lib.cquantize_blockwise_fp32(*c_args)
        else: raise ValueError(f"Blockwise quantization input A supports float16/bf16/float32, got {A.dtype}")
            
    return quantized_out, absmax_out

def _dequantize_blockwise_impl(
    A_quantized: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, 
    blocksize: int, dtype_out: torch.dtype, out: torch.Tensor
):
    torch._check_is_size(blocksize)
    torch._check(A_quantized.dtype == torch.uint8, lambda: f"Quantized input A must be uint8, got {A_quantized.dtype}")
    torch._check(dtype_out in [torch.float16, torch.bfloat16, torch.float32],
                 lambda: f"Output dtype must be float16, bfloat16, or float32, got {dtype_out}")

    supported_blocksizes_cuda = [4096, 2048, 1024, 512, 256, 128, 64]
    supported_blocksizes_hip = [4096, 2048, 1024, 512, 256, 128]
    current_supported_blocksizes = supported_blocksizes_hip if HIP_ENVIRONMENT else supported_blocksizes_cuda
    torch._check(blocksize in current_supported_blocksizes, lambda: f"Blocksize check failed for {blocksize} on {A_quantized.device.type}")

    n_out = out.numel()

    with _cuda_device_of(A_quantized):
        stream = _get_tensor_stream(A_quantized)
        code_on_device = code.to(A_quantized.device) if code.device != A_quantized.device else code
        absmax_on_device_fp32 = absmax.to(device=A_quantized.device, dtype=torch.float32) \
                                if absmax.device != A_quantized.device or absmax.dtype != torch.float32 \
                                else absmax

        c_args = (
            get_ptr(code_on_device), get_ptr(A_quantized), get_ptr(absmax_on_device_fp32), get_ptr(out),
            ct.c_int(blocksize), ct.c_int(n_out), stream
        )
        if dtype_out == torch.float16: lib.cdequantize_blockwise_fp16(*c_args)
        elif dtype_out == torch.bfloat16: lib.cdequantize_blockwise_bf16(*c_args)
        elif dtype_out == torch.float32: lib.cdequantize_blockwise_fp32(*c_args)

@register_kernel("bitsandbytes::dequantize_blockwise", "cuda")
def _dequantize_blockwise_entry(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    # Output shape is same as A_quantized for this op, but with target dtype
    out = torch.empty_like(A, dtype=dtype) 
    _dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out)
    return out

@register_kernel("bitsandbytes::dequantize_blockwise.out", "cuda")
def _dequantize_blockwise_out_entry(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype, out: torch.Tensor): 
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
    torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
    _dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out)

@register_kernel("bitsandbytes::quantize_4bit", "cuda")
def _quantize_4bit_entry( 
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(quant_type in ["fp4", "nf4"], lambda: f"quant_type must be 'fp4' or 'nf4', got {quant_type}")
    torch._check(A.dtype in [torch.bfloat16, torch.float16, torch.float32],
                 lambda: f"4-bit quantization input A supports bfloat16/float16/float32, got {A.dtype}")

    supported_blocksizes_cuda = [4096, 2048, 1024, 512, 256, 128, 64]
    supported_blocksizes_hip = [4096, 2048, 1024, 512, 256, 128]
    current_supported_blocksizes = supported_blocksizes_hip if HIP_ENVIRONMENT else supported_blocksizes_cuda
    torch._check(blocksize in current_supported_blocksizes, lambda: f"Blocksize check failed for {blocksize} on {A.device.type}")

    n = A.numel()
    num_blocks = (n + blocksize - 1) // blocksize
    absmax_out = torch.empty((num_blocks,), device=A.device, dtype=torch.float32)
    
    elements_per_storage_unit = quant_storage.itemsize * 2 
    num_packed_elements = (n + elements_per_storage_unit - 1) // elements_per_storage_unit
    quantized_out = torch.empty((num_packed_elements, 1), device=A.device, dtype=quant_storage)

    with _cuda_device_of(A):
        stream = _get_tensor_stream(A)
        c_args = (
            None, get_ptr(A), get_ptr(absmax_out), get_ptr(quantized_out),
            ct.c_int32(blocksize), ct.c_int(n), stream
        )
        if A.dtype == torch.bfloat16:
            if quant_type == "fp4": lib.cquantize_blockwise_bf16_fp4(*c_args)
            else: lib.cquantize_blockwise_bf16_nf4(*c_args)
        elif A.dtype == torch.float16:
            if quant_type == "fp4": lib.cquantize_blockwise_fp16_fp4(*c_args)
            else: lib.cquantize_blockwise_fp16_nf4(*c_args)
        elif A.dtype == torch.float32:
            if quant_type == "fp4": lib.cquantize_blockwise_fp32_fp4(*c_args)
            else: lib.cquantize_blockwise_fp32_nf4(*c_args)
            
    return quantized_out, absmax_out

def _dequantize_4bit_impl(
    A_quantized: torch.Tensor, absmax: torch.Tensor, blocksize: int,
    quant_type: str, dtype_out: torch.dtype, out: torch.Tensor
):
    torch._check_is_size(blocksize)
    torch._check(quant_type in ["fp4", "nf4"], lambda: f"quant_type must be 'fp4' or 'nf4', got {quant_type}")
    torch._check(dtype_out in [torch.bfloat16, torch.float16, torch.float32],
                 lambda: f"4-bit dequantization output supports bfloat16/float16/float32, got {dtype_out}")

    supported_blocksizes_cuda = [4096, 2048, 1024, 512, 256, 128, 64]
    supported_blocksizes_hip = [4096, 2048, 1024, 512, 256, 128]
    current_supported_blocksizes = supported_blocksizes_hip if HIP_ENVIRONMENT else supported_blocksizes_cuda
    torch._check(blocksize in current_supported_blocksizes, lambda: f"Blocksize check failed for {blocksize} on {A_quantized.device.type}")

    n_out = out.numel()

    with _cuda_device_of(A_quantized):
        stream = _get_tensor_stream(A_quantized)
        absmax_on_device_fp32 = absmax.to(device=A_quantized.device, dtype=torch.float32) \
                                if absmax.device != A_quantized.device or absmax.dtype != torch.float32 \
                                else absmax
        c_args = (
            None, get_ptr(A_quantized), get_ptr(absmax_on_device_fp32), get_ptr(out),
            ct.c_int(blocksize), ct.c_int(n_out), stream
        )
        if out.dtype == torch.bfloat16:
            if quant_type == "fp4": lib.cdequantize_blockwise_bf16_fp4(*c_args)
            else: lib.cdequantize_blockwise_bf16_nf4(*c_args)
        elif out.dtype == torch.float16:
            if quant_type == "fp4": lib.cdequantize_blockwise_fp16_fp4(*c_args)
            else: lib.cdequantize_blockwise_fp16_nf4(*c_args)
        elif out.dtype == torch.float32:
            if quant_type == "fp4": lib.cdequantize_blockwise_fp32_fp4(*c_args)
            else: lib.cdequantize_blockwise_fp32_nf4(*c_args)

@register_kernel("bitsandbytes::dequantize_4bit", "cuda")
def _dequantize_4bit_entry( #Renamed
    A: torch.Tensor, absmax: torch.Tensor, blocksize: int,
    quant_type: str, shape: Sequence[int], dtype: torch.dtype
) -> torch.Tensor:
    out = torch.empty(shape, dtype=dtype, device=A.device)
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out)
    return out

@register_kernel("bitsandbytes::dequantize_4bit.out", "cuda")
def _dequantize_4bit_out_entry( #Renamed
    A: torch.Tensor, absmax: torch.Tensor, blocksize: int,
    quant_type: str, shape: Sequence[int], dtype: torch.dtype, out: torch.Tensor
):
    torch._check(list(out.shape) == list(shape), lambda: f"Expected out.shape == {list(shape)}, got {list(out.shape)}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
    _dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out)

def _gemv_4bit_impl(
    A_vec: torch.Tensor, B_q4: torch.Tensor, shapeB_orig: Sequence[int],
    absmax_B: torch.Tensor, code_B_map: torch.Tensor, blocksize_B: int, out: torch.Tensor
):
    torch._check_is_size(blocksize_B)
    torch._check(A_vec.numel() == A_vec.size(-1), lambda: f"A_vec must be a vector, got {A_vec.shape}")
    torch._check(A_vec.dtype in [torch.float16, torch.bfloat16, torch.float32],
                 lambda: f"A_vec dtype must be float16/bf16/fp32, got {A_vec.dtype}")
    torch._check(absmax_B.dtype == torch.float32, lambda: f"absmax_B must be float32, got {absmax_B.dtype}")
    torch._check(code_B_map.dtype == torch.float32, lambda: f"code_B_map (quant map) must be float32, got {code_B_map.dtype}")
    
    m_ffi = ct.c_int32(shapeB_orig[0]) 
    n_ffi = ct.c_int32(1)
    k_ffi = ct.c_int32(shapeB_orig[1]) 

    lda_ffi = m_ffi 
    ldb_ffi = ct.c_int32((k_ffi.value + 1) // 2) 
    ldc_ffi = m_ffi 

    stream = _get_tensor_stream(A_vec)
    with _cuda_device_of(A_vec):
        code_B_map_on_device = code_B_map.to(A_vec.device) if code_B_map.device != A_vec.device else code_B_map
        absmax_B_on_device = absmax_B.to(A_vec.device) if absmax_B.device != A_vec.device else absmax_B

        c_kernel_args = [
            m_ffi, n_ffi, k_ffi,
            get_ptr(A_vec.contiguous()), get_ptr(B_q4), 
            get_ptr(absmax_B_on_device.contiguous()), get_ptr(code_B_map_on_device.contiguous()),
            get_ptr(out),
            lda_ffi, ldb_ffi, ldc_ffi,
            ct.c_int32(blocksize_B),
            stream
        ]
        if A_vec.dtype == torch.float16: lib.cgemm_4bit_inference_naive_fp16(*c_kernel_args)
        elif A_vec.dtype == torch.bfloat16: lib.cgemm_4bit_inference_naive_bf16(*c_kernel_args)
        elif A_vec.dtype == torch.float32: lib.cgemm_4bit_inference_naive_fp32(*c_kernel_args)
        else: raise TypeError(f"Unsupported dtype for A_vec in _gemv_4bit_impl: {A_vec.dtype}")

@register_kernel("bitsandbytes::gemv_4bit", "cuda")
def _gemv_4bit_entry( 
    A: torch.Tensor, B: torch.Tensor, shapeB: Sequence[int], 
    absmax: torch.Tensor, code: torch.Tensor, blocksize: int
) -> torch.Tensor:
    out_shape = (*A.shape[:-1], shapeB[0]) 
    out = torch.empty(out_shape, device=A.device, dtype=A.dtype)
    _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out)
    return out

@register_kernel("bitsandbytes::gemv_4bit.out", "cuda")
def _gemv_4bit_out_entry( 
    A: torch.Tensor, B: torch.Tensor, shapeB: Sequence[int],
    absmax: torch.Tensor, code: torch.Tensor, blocksize: int, out: torch.Tensor
):
    expected_out_shape = (*A.shape[:-1], shapeB[0])
    torch._check(list(out.shape) == list(expected_out_shape), 
                 lambda: f"Expected out.shape == {list(expected_out_shape)}, got {list(out.shape)}")
    torch._check(out.dtype == A.dtype, lambda: f"Expected out.dtype == {A.dtype}, got {out.dtype}")
    _gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out)
