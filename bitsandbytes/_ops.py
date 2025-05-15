'''
from collections.abc import Sequence
from math import prod
from typing import Optional

import torch

_IS_TORCH_GTE_24 = False

if hasattr(torch.library, "register_fake"):
    _IS_TORCH_GTE_24 = True
    register_fake = torch.library.register_fake
    register_kernel = torch.library.register_kernel
else:
    # PyTorch <= 2.3
    register_fake = torch.library.impl_abstract
    register_kernel = torch.library.impl

# Int8 mixed precision matmul + dequant + bias
torch.library.define(
    "bitsandbytes::int8_mixed_scaled_mm",
    "(Tensor A, Tensor CA, Tensor CB, Tensor SCA, Tensor SCB, Tensor? outlier_cols=None, Tensor? bias=None) -> (Tensor, Tensor?)",
)


@register_fake("bitsandbytes::int8_mixed_scaled_mm")
def _(
    A: torch.Tensor,
    CA: torch.Tensor,
    CB: torch.Tensor,
    SCA: torch.Tensor,
    SCB: torch.Tensor,
    outlier_cols: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    shapeC = (*CA.shape[:-1], CB.shape[0])

    out = torch.empty(shapeC, device=A.device, dtype=A.dtype)

    outlier_cols = torch.library.get_ctx().new_dynamic_size()
    subA = A.new_empty(outlier_cols, dtype=torch.int64)

    return out, subA


# Higher level op: int8 matmul + dequant + bias
torch.library.define(
    "bitsandbytes::int8_scaled_mm",
    "(Tensor A, Tensor B, Tensor row_stats, Tensor col_stats, Tensor? bias=None, ScalarType? dtype=None) -> Tensor",
)


@register_fake("bitsandbytes::int8_scaled_mm")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    shapeC = (*A.shape[:-1], B.shape[0])
    return torch.empty(shapeC, device=A.device, dtype=dtype or torch.float16)


torch.library.define(
    "bitsandbytes::int8_linear_matmul",
    "(Tensor A, Tensor B) -> Tensor",
)


@register_fake("bitsandbytes::int8_linear_matmul")
def _(A: torch.Tensor, B: torch.Tensor):
    torch._check(A.dtype == torch.int8, lambda: "A must be int8")
    torch._check(B.dtype == torch.int8, lambda: "B must be int8")
    shapeC = (*A.shape[:-1], B.shape[0])
    return torch.empty(shapeC, device=A.device, dtype=torch.int32)


# More info on `out` overloads:
# https://github.com/pytorch/pytorch/issues/125044
torch.library.define(
    "bitsandbytes::int8_linear_matmul.out",
    "(Tensor A, Tensor B, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::int8_linear_matmul.out")
def _(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
    shapeC = (*A.shape[:-1], B.shape[0])

    torch._check(A.dtype == torch.int8, lambda: "A must be int8")
    torch._check(B.dtype == torch.int8, lambda: "B must be int8")
    torch._check(out.shape == shapeC, lambda: f"Expected out.shape == {shapeC}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == torch.int32, lambda: f"Expected out.dtype == int32, got {out.dtype}")


torch.library.define(
    "bitsandbytes::int8_vectorwise_quant",
    "(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor?)",
)


@register_fake("bitsandbytes::int8_vectorwise_quant")
def _(A: torch.Tensor, threshold=0.0):
    out_row = torch.empty(A.shape, device=A.device, dtype=torch.int8)
    row_stats = torch.empty(prod(A.shape[:-1]), device=A.device, dtype=torch.float32)

    if threshold == 0.0:
        return out_row, row_stats, None

    outlier_cols = torch.library.get_ctx().new_dynamic_size()

    return out_row, row_stats, A.new_empty(outlier_cols, dtype=torch.int64)


torch.library.define("bitsandbytes::int8_vectorwise_dequant", "(Tensor A, Tensor stats) -> Tensor")


@register_fake("bitsandbytes::int8_vectorwise_dequant")
def _(A: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    torch._check(A.dtype == torch.int8, lambda: "A must be int8")
    return torch.empty_like(A, dtype=torch.float32)


# Default PyTorch-native implementation
@register_kernel("bitsandbytes::int8_vectorwise_dequant", "default")
def _(A: torch.Tensor, stats: torch.Tensor):
    # To dequantize we divide by 127, or multiply by the reciprocal.
    return A * stats.view(-1, 1) * 7.874015718698502e-3


torch.library.define(
    "bitsandbytes::int8_mm_dequant",
    "(Tensor A, Tensor row_stats, Tensor col_stats, ScalarType? dtype=None, Tensor? bias=None) -> Tensor",
)


@register_fake("bitsandbytes::int8_mm_dequant")
def _(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch._check(A.dtype == torch.int32, lambda: "A must be int32")
    return torch.empty_like(A, dtype=dtype or torch.float16)


torch.library.define(
    "bitsandbytes::int8_double_quant",
    "(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor?)",
)


@register_fake("bitsandbytes::int8_double_quant")
def _(
    A: torch.Tensor,
    threshold=0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    out_row = torch.empty_like(A, dtype=torch.int8)
    out_col = torch.empty_like(A, dtype=torch.int8)
    row_stats = torch.empty(prod(A.shape[:-1]), device=A.device, dtype=torch.float32)
    col_stats = torch.empty(A.shape[-1], device=A.device, dtype=torch.float32)
    outlier_n = torch.library.get_ctx().new_dynamic_size()
    outlier_cols = A.new_empty(outlier_n, dtype=torch.int64)
    return out_row, out_col, row_stats, col_stats, outlier_cols


torch.library.define(
    "bitsandbytes::dequantize_4bit",
    "(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_4bit")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    return torch.empty(shape, dtype=dtype, device=A.device)


torch.library.define(
    "bitsandbytes::dequantize_4bit.out",
    "(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::dequantize_4bit.out")
def _(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check_is_size(blocksize)
    torch._check(out.shape == shape, lambda: f"Expected out.shape == {shape}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")


torch.library.define(
    "bitsandbytes::quantize_4bit",
    "(Tensor A, int blocksize, str quant_type, ScalarType quant_storage) -> (Tensor, Tensor)",
)


@register_fake("bitsandbytes::quantize_4bit")
def _(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)

    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty(((n + 1) // (quant_storage.itemsize * 2), 1), device=A.device, dtype=quant_storage)
    return out, absmax


torch.library.define(
    "bitsandbytes::dequantize_blockwise",
    "(Tensor A, Tensor absmax, Tensor code, int blocksize, ScalarType dtype) -> Tensor",
)


@register_fake("bitsandbytes::dequantize_blockwise")
def _(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    return torch.empty_like(A, dtype=dtype)


torch.library.define(
    "bitsandbytes::dequantize_blockwise.out",
    "(Tensor A, Tensor absmax, Tensor code, int blocksize, ScalarType dtype, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::dequantize_blockwise.out")
def _(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype, out: torch.Tensor
):
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")


torch.library.define("bitsandbytes::quantize_blockwise", "(Tensor A, Tensor code, int blocksize) -> (Tensor, Tensor)")


@register_fake("bitsandbytes::quantize_blockwise")
def _(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    n = A.numel()
    blocks = -(n // -blocksize)
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty_like(A, dtype=torch.uint8)
    return out, absmax


torch.library.define(
    "bitsandbytes::gemv_4bit",
    "(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code, int blocksize) -> Tensor",
)


@register_fake("bitsandbytes::gemv_4bit")
def _(
    A: torch.Tensor, B: torch.Tensor, shapeB: Sequence[int], absmax: torch.Tensor, code: torch.Tensor, blocksize: int
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.numel() == A.size(-1), lambda: f"A must be a vector with leading dimensions of 1, got {A.shape}")
    torch._check(
        A.dtype in [torch.float16, torch.bfloat16, torch.float32],
        lambda: f"A must be float16, bfloat16, or float32, got {A.dtype}",
    )
    torch._check(
        B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32],
        lambda: f"B must be backed by storage of type uint8, bfloat16, float16, or float32, got {B.dtype}",
    )
    shape = (*A.shape[:-1], shapeB[0])
    return torch.empty(shape, device=A.device, dtype=A.dtype)


torch.library.define(
    "bitsandbytes::gemv_4bit.out",
    "(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code, int blocksize, Tensor! out) -> ()",
)


@register_fake("bitsandbytes::gemv_4bit.out")
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    out: torch.Tensor,
) -> None:
    torch._check_is_size(blocksize)
    torch._check(A.numel() == A.size(-1), lambda: f"A must be a vector with leading dimensions of 1, got {A.shape}")
    torch._check(
        A.dtype in [torch.float16, torch.bfloat16, torch.float32],
        lambda: f"A must be float16, bfloat16, or float32, got {A.dtype}",
    )
    torch._check(
        B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32],
        lambda: f"B must be backed by storage of type uint8, bfloat16, float16, or float32, got {B.dtype}",
    )
    torch._check(
        out.shape == (*A.shape[:-1], shapeB[0]),
        lambda: f"Expected out.shape == {(*A.shape[:-1], shapeB[0])}, got {out.shape}",
    )
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == A.dtype, lambda: f"Expected out.dtype == {A.dtype}, got {out.dtype}")
'''


from collections.abc import Sequence
from math import prod # Use math.prod if available, or a fallback
from typing import Optional

import torch


_IS_TORCH_GTE_24 = False
if hasattr(torch.library, "register_fake"): # PyTorch 2.4+
    _IS_TORCH_GTE_24 = True
    register_fake = torch.library.register_fake
    register_kernel = torch.library.register_kernel # register_kernel also available
else: 
    register_fake = torch.library.impl_abstract
    register_kernel = torch.library.impl


torch.library.define(
    "bitsandbytes::int8_mixed_scaled_mm", # Schema
    "(Tensor A, Tensor CA, Tensor CB, Tensor SCA, Tensor SCB, Tensor? outlier_cols=None, Tensor? bias=None) -> (Tensor, Tensor?)",
)

@register_fake("bitsandbytes::int8_mixed_scaled_mm")
def _fake_int8_mixed_scaled_mm(
    A: torch.Tensor, # Original high-precision activations/weights
    CA: torch.Tensor, # Quantized A (e.g., int8)
    CB: torch.Tensor, # Quantized B (e.g., int8)
    SCA: torch.Tensor, # Scales for A
    SCB: torch.Tensor, # Scales for B
    outlier_cols: Optional[torch.Tensor] = None, # Indices of outlier columns/features
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

    shapeC = (*CA.shape[:-1], CB.shape[0])
    out = torch.empty(shapeC, device=A.device, dtype=A.dtype) # Output dtype matches original A

    if _IS_TORCH_GTE_24 : # get_ctx is available in newer PyTorch for fake tensors
        dynamic_dim = torch.library.get_ctx().new_dynamic_size()

        subA = A.new_empty(dynamic_dim, dtype=torch.int64) # Placeholder shape
    else: # Older PyTorch, dynamic shapes might not be fully supported this way in fake impls.
          # Return a 0-sized tensor as a fallback if dynamic not possible.
        subA = A.new_empty(0, dtype=torch.int64) if outlier_cols is not None else None


    return out, subA


torch.library.define(
    "bitsandbytes::int8_scaled_mm",
    "(Tensor A, Tensor B, Tensor row_stats, Tensor col_stats, Tensor? bias=None, ScalarType? dtype=None) -> Tensor",
)

@register_fake("bitsandbytes::int8_scaled_mm")
def _fake_int8_scaled_mm(
    A: torch.Tensor, 
    B: torch.Tensor, 
          
    row_stats: torch.Tensor, # Scales for A
    col_stats: torch.Tensor, # Scales for B
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None, 
) -> torch.Tensor:

    torch._check(A.dtype == torch.int8, lambda: f"Input A for int8_scaled_mm must be int8, got {A.dtype}")
    torch._check(B.dtype == torch.int8, lambda: f"Input B for int8_scaled_mm must be int8, got {B.dtype}")
    
    shapeC = (*A.shape[:-1], B.shape[0]) # Output shape for A @ B.T
    output_dtype = dtype if dtype is not None else torch.float16 # Default output to float16
    return torch.empty(shapeC, device=A.device, dtype=output_dtype)



torch.library.define(
    "bitsandbytes::int8_linear_matmul",
    "(Tensor A, Tensor B) -> Tensor", # A @ B.T
)

@register_fake("bitsandbytes::int8_linear_matmul")
def _fake_int8_linear_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    torch._check(A.dtype == torch.int8, lambda: f"A must be int8, got {A.dtype}")
    torch._check(B.dtype == torch.int8, lambda: f"B must be int8, got {B.dtype}")
    # A is (..., K), B is (N_out, K). Output is (..., N_out)
    shapeC = (*A.shape[:-1], B.shape[0])
    return torch.empty(shapeC, device=A.device, dtype=torch.int32) # Output is int32

# .out variant for int8_linear_matmul
torch.library.define(
    "bitsandbytes::int8_linear_matmul.out",
    "(Tensor A, Tensor B, Tensor(a!) out) -> ()", # Tensor(a!) means out is modified in-place
)

@register_fake("bitsandbytes::int8_linear_matmul.out")
def _fake_int8_linear_matmul_out(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    # Validations for the .out variant
    torch._check(A.dtype == torch.int8, lambda: f"A must be int8, got {A.dtype}")
    torch._check(B.dtype == torch.int8, lambda: f"B must be int8, got {B.dtype}")
    
    expected_shapeC = (*A.shape[:-1], B.shape[0])
    torch._check(out.shape == expected_shapeC, lambda: f"Expected out.shape == {expected_shapeC}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == torch.int32, lambda: f"Expected out.dtype == int32, got {out.dtype}")
    # No return value for .out variants



torch.library.define(
    "bitsandbytes::int8_vectorwise_quant",
    "(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor?)", # QuantizedA, Scales, OutlierIndices?
)

@register_fake("bitsandbytes::int8_vectorwise_quant")
def _fake_int8_vectorwise_quant(A: torch.Tensor, threshold: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # Input A is typically float16
    # Output1: Quantized A (int8), same shape as A
    out_row_quantized = torch.empty_like(A, dtype=torch.int8)
    # Output2: Row-wise scales (absmax), shape [prod(A.shape[:-1])] (number of rows)
    num_rows = prod(A.shape[:-1]) if A.ndim > 1 else 1
    row_stats = torch.empty(num_rows, device=A.device, dtype=torch.float32)

    outlier_cols_indices: Optional[torch.Tensor] = None
    if threshold > 0.0:
        if _IS_TORCH_GTE_24:
            # Dynamic shape for outlier column indices
            dynamic_dim_outliers = torch.library.get_ctx().new_dynamic_size()
            outlier_cols_indices = A.new_empty(dynamic_dim_outliers, dtype=torch.int64) # Placeholder shape
        else: # Fallback for older PyTorch
            outlier_cols_indices = A.new_empty(0, dtype=torch.int64) # 0-sized tensor

    return out_row_quantized, row_stats, outlier_cols_indices


torch.library.define(
    "bitsandbytes::int8_vectorwise_dequant", 
    "(Tensor A, Tensor stats) -> Tensor"
)

@register_fake("bitsandbytes::int8_vectorwise_dequant")
def _fake_int8_vectorwise_dequant(A: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    # Input A is int8, stats are float32 scales
    torch._check(A.dtype == torch.int8, lambda: f"A must be int8, got {A.dtype}")
    torch._check(stats.dtype == torch.float32, lambda: f"stats must be float32, got {stats.dtype}")
    # Output is dequantized A, typically float32 or original float type (e.g., float16)
    # Main PDF's default kernel returns float32.
    return torch.empty_like(A, dtype=torch.float32)


@register_kernel("bitsandbytes::int8_vectorwise_dequant", "default")
def _default_int8_vectorwise_dequant(A: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:

    if A.ndim == 1: # Vector case
        # stats should be scalar or match A's device for broadcasting
        return A.to(torch.float32) * stats.to(A.device) * (1.0 / 127.0)
    
    # General case: stats correspond to all but the last dimension of A
    stats_shape_for_broadcast = (*A.shape[:-1], 1)
    stats_reshaped = stats.reshape(stats_shape_for_broadcast)
    
    return A.to(torch.float32) * stats_reshaped.to(A.device) * (1.0 / 127.0)


torch.library.define(
    "bitsandbytes::int8_mm_dequant",
    "(Tensor A, Tensor row_stats, Tensor col_stats, ScalarType? dtype=None, Tensor? bias=None) -> Tensor",
)

@register_fake("bitsandbytes::int8_mm_dequant")
def _fake_int8_mm_dequant(
    A: torch.Tensor, # int32 input (result of int8 matmul)
    row_stats: torch.Tensor, # float32 scales for LHS of original matmul
    col_stats: torch.Tensor, # float32 scales for RHS of original matmul
    dtype: Optional[torch.dtype] = None, # Desired output dtype
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch._check(A.dtype == torch.int32, lambda: f"A must be int32, got {A.dtype}")
    # Output dtype defaults to float16 if not specified
    output_dtype = dtype if dtype is not None else torch.float16
    return torch.empty_like(A, dtype=output_dtype) # Shape matches input A, dtype is output_dtype


torch.library.define(
    "bitsandbytes::int8_double_quant",
    "(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor?)",
    # Out: (QuantA_row, QuantA_col, Scales_row, Scales_col, OutlierIndices?)
)

@register_fake("bitsandbytes::int8_double_quant")
def _fake_int8_double_quant(
    A: torch.Tensor, threshold: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    out_row_quant = torch.empty_like(A, dtype=torch.int8)

    out_col_quant = torch.empty_like(A, dtype=torch.int8)
    
    num_rows = prod(A.shape[:-1]) if A.ndim > 1 else 1
    num_cols = A.shape[-1]


    row_stats_scales = torch.empty(num_rows, device=A.device, dtype=torch.float32)

    col_stats_scales = torch.empty(num_cols, device=A.device, dtype=torch.float32)

    outlier_col_indices: Optional[torch.Tensor] = None
    if threshold > 0.0:
        if _IS_TORCH_GTE_24:
            dynamic_dim_outliers = torch.library.get_ctx().new_dynamic_size()
            outlier_col_indices = A.new_empty(dynamic_dim_outliers, dtype=torch.int64)
        else:
            outlier_col_indices = A.new_empty(0, dtype=torch.int64)
            
    return out_row_quant, out_col_quant, row_stats_scales, col_stats_scales, outlier_col_indices


torch.library.define(
    "bitsandbytes::dequantize_4bit",
    "(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype) -> Tensor",
)

@register_fake("bitsandbytes::dequantize_4bit")
def _fake_dequantize_4bit(
    A: torch.Tensor, # Packed 4-bit data (e.g., uint8)
    absmax: torch.Tensor, # Scales (float32)
    blocksize: int,
    quant_type: str, # "nf4" or "fp4"
    shape: Sequence[int], # Original shape of the dequantized tensor
    dtype: torch.dtype, # Original dtype to dequantize to
) -> torch.Tensor:
    torch._check_is_size(blocksize) # Ensure blocksize is a single int
    # A.dtype (quant_storage) and absmax.dtype checks are usually done in kernel or Python wrapper.
    # quant_type should be "nf4" or "fp4".
    return torch.empty(shape, dtype=dtype, device=A.device)

# .out variant for 4-bit Dequantization
torch.library.define(
    "bitsandbytes::dequantize_4bit.out",
    "(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype, Tensor(a!) out) -> ()",
)

@register_fake("bitsandbytes::dequantize_4bit.out")
def _fake_dequantize_4bit_out(
    A: torch.Tensor, absmax: torch.Tensor, blocksize: int,
    quant_type: str, shape: Sequence[int], dtype: torch.dtype, out: torch.Tensor,
) -> None:
    torch._check_is_size(blocksize)
    torch._check(list(out.shape) == list(shape), lambda: f"Expected out.shape == {list(shape)}, got {list(out.shape)}") # list() for Sequence
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")


torch.library.define(
    "bitsandbytes::quantize_4bit",
    "(Tensor A, int blocksize, str quant_type, ScalarType quant_storage) -> (Tensor, Tensor)",
    # Out: (QuantizedData_packed, AbsmaxScales)
)

@register_fake("bitsandbytes::quantize_4bit")
def _fake_quantize_4bit(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)


    n = A.numel()
    num_blocks = (n + blocksize - 1) // blocksize
    

    absmax_scales = torch.empty((num_blocks,), device=A.device, dtype=torch.float32)

    num_4bit_values_per_storage_unit = quant_storage.itemsize * 2 # if itemsize is in bytes (e.g. 1 for uint8)
    
    num_packed_elements = (n + num_4bit_values_per_storage_unit - 1) // num_4bit_values_per_storage_unit

    packed_data_shape_dim0 = (n + (quant_storage.itemsize * 2 -1) ) // (quant_storage.itemsize * 2) # Corrected from (n+1) for general case

    dim0_packed = (n + 1) // (quant_storage.itemsize * 2) if quant_storage.itemsize > 0 else n # Avoid div by zero if itemsize is 0 (should not happen)
    quantized_data_packed = torch.empty((dim0_packed, 1), device=A.device, dtype=quant_storage)
    
    return quantized_data_packed, absmax_scales



torch.library.define(
    "bitsandbytes::dequantize_blockwise",
    "(Tensor A, Tensor absmax, Tensor code, int blocksize, ScalarType dtype) -> Tensor",
)

@register_fake("bitsandbytes::dequantize_blockwise")
def _fake_dequantize_blockwise(
    A: torch.Tensor, # Quantized data (uint8)
    absmax: torch.Tensor, # Scales per block (float32)
    code: torch.Tensor, # Quantization map (float32)
    blocksize: int,
    dtype: torch.dtype, # Target output dtype
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    # absmax.dtype and code.dtype checks in kernel/wrapper.
    # dtype (output) can be fp16, bf16, fp32.
    # Output shape is same as A (quantized data), but with new dtype.
    return torch.empty_like(A, dtype=dtype)

# .out variant for Blockwise Dequantization
torch.library.define(
    "bitsandbytes::dequantize_blockwise.out",
    "(Tensor A, Tensor absmax, Tensor code, int blocksize, ScalarType dtype, Tensor(a!) out) -> ()",
)

@register_fake("bitsandbytes::dequantize_blockwise.out")
def _fake_dequantize_blockwise_out(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, 
    blocksize: int, dtype: torch.dtype, out: torch.Tensor,
) -> None:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")



torch.library.define(
    "bitsandbytes::quantize_blockwise",
    "(Tensor A, Tensor code, int blocksize) -> (Tensor, Tensor)",
    # Out: (QuantizedData_uint8, AbsmaxScales_float32)
)

@register_fake("bitsandbytes::quantize_blockwise")
def _fake_quantize_blockwise(
    A: torch.Tensor, code: torch.Tensor, blocksize: int
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    # A.dtype (input precision) and code.dtype checks in kernel/wrapper.
    
    n = A.numel()
    num_blocks = (n + blocksize - 1) // blocksize # Correct way to get num_blocks
    

    absmax_scales = torch.empty((num_blocks,), device=A.device, dtype=torch.float32)

    quantized_data = torch.empty_like(A, dtype=torch.uint8) # Same shape as A
    
    return quantized_data, absmax_scales


torch.library.define(
    "bitsandbytes::gemv_4bit",
    "(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code, int blocksize) -> Tensor",
    # A: input vector (float)
    # B: packed 4-bit quantized matrix (uint8)
    # shapeB: original shape of dequantized B matrix [rows_B, cols_B]
    # absmax: scales for B
    # code: quantization map for B (e.g. NF4/FP4 table)
    # blocksize: blocksize used for B's quantization
    # Output: A @ B_dequantized
)

@register_fake("bitsandbytes::gemv_4bit")
def _fake_gemv_4bit(
    A: torch.Tensor, B: torch.Tensor, shapeB: Sequence[int], 
    absmax: torch.Tensor, code: torch.Tensor, blocksize: int
) -> torch.Tensor:
    torch._check_is_size(blocksize)

    if not shapeB or len(shapeB) == 0: # Ensure shapeB is not empty
        raise ValueError("shapeB cannot be empty for gemv_4bit fake implementation.")
        
    output_shape = (*A.shape[:-1], shapeB[0])
    return torch.empty(output_shape, device=A.device, dtype=A.dtype) # Output dtype matches input vector A


torch.library.define(
    "bitsandbytes::gemv_4bit.out",
    "(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code, int blocksize, Tensor(a!) out) -> ()",
)

@register_fake("bitsandbytes::gemv_4bit.out")
def _fake_gemv_4bit_out(
    A: torch.Tensor, B: torch.Tensor, shapeB: Sequence[int],
    absmax: torch.Tensor, code: torch.Tensor, blocksize: int, out: torch.Tensor,
) -> None:
    torch._check_is_size(blocksize)
    if not shapeB or len(shapeB) == 0:
        raise ValueError("shapeB cannot be empty for gemv_4bit.out fake implementation.")

    expected_out_shape = (*A.shape[:-1], shapeB[0])
    torch._check(list(out.shape) == list(expected_out_shape), lambda: f"Expected out.shape == {list(expected_out_shape)}, got {list(out.shape)}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == A.dtype, lambda: f"Expected out.dtype == {A.dtype}, got {out.dtype}")

