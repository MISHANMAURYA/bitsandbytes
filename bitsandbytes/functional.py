'''
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Iterable
import ctypes as ct
import itertools
from math import prod
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import Tensor
from typing_extensions import deprecated

from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict

from .cextension import lib

name2qmap = {}

if lib and lib.compiled_with_cuda:
    """C FUNCTIONS FOR OPTIMIZERS"""
    str2optimizer32bit = {
        "adam": (
            lib.cadam32bit_grad_fp32,
            lib.cadam32bit_grad_fp16,
            lib.cadam32bit_grad_bf16,
        ),
        "momentum": (
            lib.cmomentum32bit_grad_32,
            lib.cmomentum32bit_grad_16,
        ),
        "rmsprop": (
            lib.crmsprop32bit_grad_32,
            lib.crmsprop32bit_grad_16,
        ),
        "lion": (
            lib.clion32bit_grad_fp32,
            lib.clion32bit_grad_fp16,
            lib.clion32bit_grad_bf16,
        ),
        "adagrad": (
            lib.cadagrad32bit_grad_32,
            lib.cadagrad32bit_grad_16,
        ),
        "lamb": (
            lib.cadam32bit_grad_fp32,
            lib.cadam32bit_grad_fp16,
            lib.cadam32bit_grad_bf16,
        ),
        "ademamix": (
            lib.cademamix32bit_grad_fp32,
            lib.cademamix32bit_grad_fp16,
            lib.cademamix32bit_grad_bf16,
        ),
    }

    str2optimizer8bit = {
        "adam": (
            lib.cadam_static_8bit_grad_32,
            lib.cadam_static_8bit_grad_16,
        ),
        "momentum": (
            lib.cmomentum_static_8bit_grad_32,
            lib.cmomentum_static_8bit_grad_16,
        ),
        "rmsprop": (
            lib.crmsprop_static_8bit_grad_32,
            lib.crmsprop_static_8bit_grad_16,
        ),
        "lion": (
            lib.clion_static_8bit_grad_32,
            lib.clion_static_8bit_grad_16,
        ),
        "lamb": (
            lib.cadam_static_8bit_grad_32,
            lib.cadam_static_8bit_grad_16,
        ),
        "lars": (
            lib.cmomentum_static_8bit_grad_32,
            lib.cmomentum_static_8bit_grad_16,
        ),
    }

    str2optimizer8bit_blockwise = {
        "adam": (
            lib.cadam_8bit_blockwise_grad_fp32,
            lib.cadam_8bit_blockwise_grad_fp16,
            lib.cadam_8bit_blockwise_grad_bf16,
        ),
        "momentum": (
            lib.cmomentum_8bit_blockwise_grad_fp32,
            lib.cmomentum_8bit_blockwise_grad_fp16,
            lib.cmomentum_8bit_blockwise_grad_bf16,
        ),
        "rmsprop": (
            lib.crmsprop_8bit_blockwise_grad_fp32,
            lib.crmsprop_8bit_blockwise_grad_fp16,
            lib.crmsprop_8bit_blockwise_grad_bf16,
        ),
        "lion": (
            lib.clion_8bit_blockwise_grad_fp32,
            lib.clion_8bit_blockwise_grad_fp16,
            lib.clion_8bit_blockwise_grad_bf16,
        ),
        "adagrad": (
            lib.cadagrad_8bit_blockwise_grad_fp32,
            lib.cadagrad_8bit_blockwise_grad_fp16,
            lib.cadagrad_8bit_blockwise_grad_bf16,
        ),
        "ademamix": (
            lib.cademamix_8bit_blockwise_grad_fp32,
            lib.cademamix_8bit_blockwise_grad_fp16,
            lib.cademamix_8bit_blockwise_grad_bf16,
        ),
    }


class GlobalPageManager:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.paged_tensors = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def prefetch_all(self, to_cpu=False):
        # assume the first added, will be the
        # ones that are used first, so swap them in last
        # in the case they are evicted again
        for t in self.paged_tensors[::-1]:
            prefetch_tensor(t, to_cpu)


class CUBLAS_Context:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.context = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def get_context(self, device):
        if device.index not in self.context:
            prev_device = torch.cuda.current_device()
            torch.cuda.set_device(device)
            self.context[device.index] = ct.c_void_p(lib.get_context())
            torch.cuda.set_device(prev_device)
        return self.context[device.index]


class Cusparse_Context:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.context = ct.c_void_p(lib.get_cusparse())

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


FIRST_CUDA_DEVICE = torch.device("cuda", index=0)

# When multiple GPUs are present, we use a context manager to
# switch to the correct device of a tensor before invoking our CUDA
# kernels in the C++ library. However, when there's only one device
# there is no need to incur the overhead of cudaGetDevice/cudaSetDevice.
if torch.cuda.device_count() > 1:

    def _cuda_device_of(a: torch.Tensor):
        return torch.cuda.device_of(a)
else:
    import contextlib

    def _cuda_device_of(a: torch.Tensor):
        return contextlib.nullcontext()


def get_paged(*shape, dtype=torch.float32, device=FIRST_CUDA_DEVICE):
    num_bytes = dtype.itemsize * prod(shape)
    cuda_ptr = lib.cget_managed_ptr(ct.c_size_t(num_bytes))
    c_ptr = ct.cast(cuda_ptr, ct.POINTER(ct.c_int))
    new_array = np.ctypeslib.as_array(c_ptr, shape=shape)
    out = torch.frombuffer(new_array, dtype=dtype, count=prod(shape)).view(shape)
    out.is_paged = True
    out.page_deviceid = device.index
    return out


def prefetch_tensor(A: torch.Tensor, to_cpu=False):
    assert A.is_paged, "Only paged tensors can be prefetched!"
    if to_cpu:
        deviceid = -1
    else:
        deviceid = A.page_deviceid

    lib.cprefetch(get_ptr(A), ct.c_size_t(A.nbytes), ct.c_int32(deviceid))


def elementwise_func(func_name, A, B, value, prefetch=True):
    func = None
    if A.dtype == torch.float32:
        func = getattr(lib, f"c{func_name}_fp32", None)
        cvalue = ct.c_float(value)
    elif A.dtype == torch.uint8:
        func = getattr(lib, f"c{func_name}_uint8", None)
        cvalue = ct.c_uint8(value)

    if func is None:
        raise NotImplementedError(f"Function not implemented: {func_name}")

    is_managed = getattr(A, "is_managed", False)
    if is_managed and prefetch:
        prefetch_tensor(A)
        if B is not None:
            prefetch_tensor(B)

    func(get_ptr(A), get_ptr(B), cvalue, ct.c_int64(A.numel()))
    if A.is_paged or B.is_paged:
        # paged function are fully asynchronous
        # if we return from this function, we want to the tensor
        # to be in the correct state, that is the final state after the
        # operation occurred. So we synchronize.
        torch.cuda.synchronize()


def fill(A, value, device=None, prefetch=True):
    elementwise_func("fill", A, None, value)


def _mul(A, B, device=None):
    elementwise_func("_mul", A, B, 0)


def create_linear_map(signed=True, total_bits=8, add_zero=True):
    sign = -1.0 if signed else 0.0
    total_values = 2**total_bits
    if add_zero or total_bits < 8:
        # add a zero
        # since we simulate less bits by having zeros in the data type, we
        # we need to center the quantization around zero and as such lose
        # a single value
        total_values = 2**total_bits if not signed else 2**total_bits - 1

    values = torch.linspace(sign, 1.0, total_values)
    gap = 256 - values.numel()
    if gap == 0:
        return values
    else:
        l = values.numel() // 2  # noqa: E741
        return torch.Tensor(values[:l].tolist() + [0] * gap + values[l:].tolist())


def create_normal_map(offset=0.9677083, use_extra_value=True):
    try:
        from scipy.stats import norm
    except ImportError as ie:
        raise ImportError(
            "Scipy is required for `create_normal_map`. Install `bitsandbytes` with the `[test]` extra.",
        ) from ie

    if use_extra_value:
        # one more positive value, this is an asymmetric type
        v1 = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1]).tolist()
        v2 = [0] * (256 - 15)  ## we have 15 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, 8)[:-1]).tolist()
        v2 = [0] * (256 - 14)  ## we have 14 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()

    v = v1 + v2 + v3

    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()

    assert values.numel() == 256

    return values


def create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    assert e + p == total_bits - has_sign
    # the exponent is biased to 2^(e-1) -1 == 0
    evalues = []
    pvalues = []
    for i, val in enumerate(range(-(2 ** (exponent_bits - has_sign)), 2 ** (exponent_bits - has_sign), 1)):
        evalues.append(2**val)

    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))
    # for ev in evalues:
    bias = 2 ** (exponent_bits - 1)
    for evalue in range(2 ** (exponent_bits)):
        for bit_pattern in lst:
            value = 1 if evalue != 0 else 0
            for i, pval in enumerate(list(bit_pattern)):
                value += pval * (2 ** -(i + 1))
            if evalue == 0:
                # subnormals
                value = value * 2**-(bias)
            else:
                # normals
                value = value * 2 ** -(evalue - bias - 1)
            values.append(value)
            if signed:
                values.append(-value)

    assert len(values) == 2**total_bits
    values.sort()
    if total_bits < 8:
        gap = 256 - len(values)
        for i in range(gap):
            values.append(0)
    values.sort()
    code = torch.tensor(values)
    code /= code.max()

    return code


def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    non_sign_bits = total_bits - (1 if signed else 1)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1,
        )
        boundaries = torch.linspace(0.1, 1, fraction_items, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    assert len(data) == 2**total_bits

    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)

    data.sort()
    return torch.tensor(data, dtype=torch.float32)


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def create_quantile_map(A, total_bits=8):
    q = estimate_quantiles(A, num_quantiles=2**total_bits - 1)
    q = q.tolist()
    q.append(0)

    gap = 256 - len(q)
    for i in range(gap):
        q.append(0)

    q.sort()

    q = Tensor(q)
    q = q / q.abs().max()
    return q


def is_on_gpu(tensors: Iterable[Optional[torch.Tensor]]):
    """Verifies that the input tensors are all on the same device.

    An input tensor may also be marked as `paged`, in which case the device placement is ignored.

    Args:
        tensors (`Iterable[Optional[torch.Tensor]]`): A list of tensors to verify.

    Raises:
        `RuntimeError`: Raised when the verification fails.

    Returns:
        `Literal[True]`
    """

    on_gpu = True
    gpu_ids = set()

    for t in tensors:
        # NULL pointers and paged tensors are OK.
        if t is not None and not getattr(t, "is_paged", False):
            on_gpu &= t.is_cuda
            gpu_ids.add(t.device.index)

    if not on_gpu:
        raise RuntimeError(
            f"All input tensors need to be on the same GPU, but found some tensors to not be on a GPU:\n {[(t.shape, t.device) for t in tensors]}",
        )

    if len(gpu_ids) > 1:
        raise RuntimeError(
            f"Input tensors need to be on the same GPU, but found the following tensor and device combinations:\n {[(t.shape, t.device) for t in tensors]}",
        )
    return on_gpu


def _get_tensor_stream(tensor: Tensor) -> ct.c_void_p:
    # We use the raw stream for performance reasons.
    return ct.c_void_p(torch._C._cuda_getCurrentRawStream(tensor.device.index))


def get_ptr(A: Optional[Tensor]) -> Optional[ct.c_void_p]:
    """Gets the memory address of the first element of a tenso

    Args:
        A (`Optional[Tensor]`): A PyTorch tensor.

    Returns:
        `Optional[ct.c_void_p]`: A pointer to the underlying tensor data.
    """
    if A is None:
        return None

    return ct.c_void_p(A.data_ptr())


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def estimate_quantiles(
    A: Tensor,
    out: Optional[torch.Tensor] = None,
    offset: float = 1 / 512,
    num_quantiles=256,
) -> Tensor:
    """
    Estimates 256 equidistant quantiles on the input tensor eCDF.

    Uses SRAM-Quantiles algorithm to quickly estimate 256 equidistant quantiles
    via the eCDF of the input tensor `A`. This is a fast but approximate algorithm
    and the extreme quantiles close to 0 and 1 have high variance / large estimation
    errors. These large errors can be avoided by using the offset variable which trims
    the distribution. The default offset value of 1/512 ensures minimum entropy encoding -- it
    trims 1/512 = 0.2% from each side of the distrivution. An offset value of 0.01 to 0.02
    usually has a much lower error but is not a minimum entropy encoding. Given an offset
    of 0.02 equidistance points in the range [0.02, 0.98] are used for the quantiles.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor. Any shape.
    out : torch.Tensor
        Tensor with the 256 estimated quantiles.
    offset : float
        The offset for the first and last quantile from 0 and 1. Default: 1/(2*num_quantiles)
    num_quantiles : int
        The number of equally spaced quantiles.

    Returns
    -------
    torch.Tensor:
        The 256 quantiles in float32 datatype.
    """
    if A.numel() < 256:
        raise NotImplementedError(
            f"Quantile estimation needs at least 256 values in the Tensor, but Tensor had only {A.numel()} values.",
        )
    if num_quantiles > 256:
        raise NotImplementedError(
            f"Currently only a maximum of 256 equally spaced quantiles are supported, but the argument num_quantiles={num_quantiles}",
        )
    if num_quantiles < 256 and offset == 1 / (512):
        # override default arguments
        offset = 1 / (2 * num_quantiles)

    if out is None:
        out = torch.zeros((256,), dtype=torch.float32, device=A.device)

    with _cuda_device_of(A):
        is_on_gpu([A, out])

        if A.dtype == torch.float32:
            lib.cestimate_quantiles_fp32(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
        elif A.dtype == torch.float16:
            lib.cestimate_quantiles_fp16(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
        else:
            raise NotImplementedError(f"Not supported data type {A.dtype}")

    if num_quantiles < 256:
        step = round(256 / num_quantiles)
        idx = torch.linspace(0, 255, num_quantiles).long().to(A.device)
        out = out[idx]

    return out


class QuantState:
    """container for quantization state components to work with Params4bit and similar classes"""

    valid_quant_types = ("fp4", "nf4")
    valid_qs_type_keys = [f"bitsandbytes__{x}" for x in valid_quant_types]
    valid_qs_keys = [
        "absmax",
        "quant_map",
        "nested_absmax",
        "nested_quant_map",
        "quant_state",
        "quant_type",
        "blocksize",
        "dtype",
        "shape",
        "nested_blocksize",
        "nested_dtype",
        "nested_offset",
    ]

    def __init__(
        self,
        absmax,
        shape=None,
        code=None,
        blocksize=None,
        quant_type=None,
        dtype=None,
        offset=None,
        state2=None,
    ):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.offset = offset
        self.state2 = state2
        self.nested = state2 is not None

    def __getitem__(self, idx):
        """
        ensures compatibility with older quant state scheme with nested lists.
        assumes the following layout:
        state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
        state2 = [absmax, input_shape, A.dtype, blocksize, None, quant_type]
        """
        if self.nested:
            list_repr = [
                self.absmax,
                self.shape,
                self.dtype,
                self.blocksize,
                [self.offset, self.state2],
                self.quant_type,
            ]
        else:
            list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, None, self.quant_type]
        return list_repr[idx]

    @classmethod
    def from_dict(cls, qs_dict: dict[str, Any], device: torch.device) -> "QuantState":
        """
        unpacks components of state_dict into QuantState
        where necessary, convert into strings, torch.dtype, ints, etc.

        qs_dict: based on state_dict, with only relevant keys, striped of prefixes.

        item with key `quant_state.bitsandbytes__[nf4/fp4]` may contain minor and non-tensor quant state items.
        """

        # unpacking tensor with non-tensor components
        qs_key = [k for k, v in qs_dict.items() if "quant_state" in k and isinstance(v, torch.Tensor)]
        if not len(qs_key) and "quant_type" not in qs_dict:
            raise ValueError("Expected packed or unpacked quant_state items, found neither")
        elif len(qs_key) != 1 or qs_key[0].split(".")[-1] not in cls.valid_qs_type_keys:
            raise ValueError(
                f"There should be exactly one `quant_state` item with ending from {cls.valid_qs_type_keys}.\nDetected {qs_key}.",
            )

        # unpacking minor and non-tensor quant state items if necessary
        if len(qs_key) == 1:
            first_qs_key = qs_key[0]
            qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))

        qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}  # strip prefixes
        assert set(qs_dict.keys()).issubset(cls.valid_qs_keys)

        if "nested_absmax" in qs_dict:
            offset = torch.tensor(float(qs_dict["nested_offset"])).to(device)
            state2 = cls(
                absmax=qs_dict["nested_absmax"].to(device),
                blocksize=qs_dict["nested_blocksize"],
                code=qs_dict["nested_quant_map"].to(device),
                dtype=getattr(torch, qs_dict["nested_dtype"]),
            )
        else:
            offset, state2 = None, None

        quant_state = cls(
            quant_type=qs_dict["quant_type"],
            absmax=qs_dict["absmax"].to(device),
            blocksize=qs_dict["blocksize"],
            code=qs_dict["quant_map"].to(device),
            dtype=getattr(torch, qs_dict["dtype"]),
            shape=torch.Size(qs_dict["shape"]) if qs_dict["shape"] is not None else None,
            offset=offset,
            state2=state2,
        )
        return quant_state

    def as_dict(self, packed=False):
        """
        returns dict of tensors and strings to use in serialization via _save_to_state_dict()
        param: packed -- returns dict[str, torch.Tensor] for state_dict fit for safetensors saving
        """
        qs_dict = {
            "quant_type": self.quant_type,
            "absmax": self.absmax,
            "blocksize": self.blocksize,
            "quant_map": self.code,
            "dtype": str(self.dtype).strip("torch."),
            "shape": tuple(self.shape),
        }
        if self.nested:
            qs_dict.update(
                {
                    "nested_absmax": self.state2.absmax,
                    "nested_blocksize": self.state2.blocksize,
                    "nested_quant_map": self.state2.code.clone(),  # un-shared to avoid restoring it after shared tensors are removed by safetensors
                    "nested_dtype": str(self.state2.dtype).strip("torch."),
                    "nested_offset": self.offset.item(),
                },
            )
        if not packed:
            return qs_dict

        # packed format allows serialization of non-tensor components, critical for saving in safetensors format
        qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
        non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
        qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = pack_dict_to_tensor(non_tensor_dict)
        return qs_packed_dict

    def to(self, device):
        # make sure the quantization state is on the right device
        self.code = self.code.to(device)
        self.absmax = self.absmax.to(device)
        if self.nested:
            self.offset = self.offset.to(device)
            self.state2.absmax = self.state2.absmax.to(device)
            self.state2.code = self.state2.code.to(device)

    def __eq__(self, other):
        if not isinstance(other, QuantState):
            return False

        return (
            torch.allclose(self.absmax, other.absmax, atol=1e-6)
            and self.shape == other.shape
            and torch.allclose(self.code, other.code, atol=1e-6)
            and self.dtype == other.dtype
            and self.blocksize == other.blocksize
            and self.quant_type == other.quant_type
            and (
                self.offset == other.offset
                if self.offset is not None and other.offset is not None
                else self.offset is other.offset
            )
            and (
                self.state2 == other.state2
                if self.state2 is not None and other.state2 is not None
                else self.state2 is other.state2
            )
        )


def quantize_blockwise(
    A: torch.Tensor,
    code: Optional[torch.Tensor] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=4096,
    nested=False,
) -> tuple[torch.Tensor, QuantState]:
    """Quantize a tensor in blocks of values.

    The input tensor is quantized by dividing it into blocks of `blocksize` values.
    The the absolute maximum value within these blocks is calculated for scaling
    the non-linear quantization.

    Args:
        A (`torch.Tensor`): The input tensor. Supports `float16`, `bfloat16`, or `float32` datatypes.
        code (`torch.Tensor`, *optional*):
            A mapping describing the low-bit data type. Defaults to a signed 8-bit dynamic type.
            For more details, see  (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561].
        absmax (`torch.Tensor`, *optional*): A tensor to use to store the absmax values.
        out (`torch.Tensor`, *optional*): A tensor to use to store the result.
        blocksize (`int`, *optional*):
            The size of the blocks. Defaults to 4096.
            Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
        nested (`bool`, *optional*): Whether to additionally quantize the absmax values. Defaults to False.

    Raises:
        ValueError: Raised when the input data type is not supported.

    Returns:
        `Tuple[torch.Tensor, QuantState]`: A tuple containing the quantization results.
        - `torch.Tensor`: The quantized tensor.
        - [`QuantState`]: The state object used to undo the quantization.
    """

    if code is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]

    _out, _absmax = torch.ops.bitsandbytes.quantize_blockwise.default(
        A,
        code.to(A.device),
        blocksize,
    )

    if nested:
        offset = _absmax.mean()
        _absmax -= offset
        qabsmax, state2 = quantize_blockwise(_absmax, blocksize=blocksize, nested=False)
        quant_state = QuantState(
            absmax=qabsmax,
            code=code,
            blocksize=blocksize,
            dtype=A.dtype,
            offset=offset,
            state2=state2,
        )
    else:
        quant_state = QuantState(absmax=_absmax, code=code.to(A.device), blocksize=blocksize, dtype=A.dtype)

    # TODO(matthewdouglas): Deprecate out kwarg
    out = out.copy_(_out) if out is not None else _out

    # TODO(matthewdouglas): Deprecate absmax kwarg
    if absmax is not None:
        quant_state.absmax = absmax.copy_(quant_state.absmax)

    return out, quant_state


def dequantize_blockwise(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 4096,
    nested=False,
) -> torch.Tensor:
    """Dequantize a tensor in blocks of values.

    The input tensor is dequantized by dividing it into blocks of `blocksize` values.
    The the absolute maximum value within these blocks is used for scaling
    the non-linear dequantization.

    Args:
        A (`torch.Tensor`): The quantized input tensor.
        quant_state ([`QuantState`], *optional*):
            The quantization state as returned by [`quantize_blockwise`].
            Required if `absmax` is not provided.
        absmax (`torch.Tensor`, *optional*):
            A tensor containing the scaling values.
            Required if `quant_state` is not provided and ignored otherwise.
        code (`torch.Tensor`, *optional*):
            A mapping describing the low-bit data type. Defaults to a signed 8-bit dynamic type.
            For more details, see  (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561].
            Ignored when `quant_state` is provided.
        out (`torch.Tensor`, *optional*): A tensor to use to store the result.
        blocksize (`int`, *optional*):
            The size of the blocks. Defaults to 4096.
            Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
            Ignored when `quant_state` is provided.

    Raises:
        ValueError: Raised when the input data type is not supported.

    Returns:
        `torch.Tensor`:
            The dequantized tensor. The datatype is indicated by `quant_state.dtype` and defaults to `torch.float32`.
    """

    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]

    if quant_state is None:
        quant_state = QuantState(absmax=absmax, code=code, blocksize=blocksize, dtype=torch.float32)

    absmax = quant_state.absmax
    if quant_state.nested:
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()

    if out is not None:
        torch.ops.bitsandbytes.dequantize_blockwise.out(
            A,
            absmax,
            code.to(A.device),
            blocksize,
            quant_state.dtype,
            out=out,
        )
        return out

    return torch.ops.bitsandbytes.dequantize_blockwise.default(
        A,
        absmax,
        quant_state.code.to(A.device),
        quant_state.blocksize,
        quant_state.dtype,
    )


def get_4bit_type(typename, device=None, blocksize=64):
    if device is None:
        device = "cuda"
    data = None
    if typename == "nf4":
        """ Implements the NF4 data type.

            Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
            is normalized into the range [-1, 1].

            For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)

            Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in
            the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.
        """
        data = [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ]
    elif typename == "fp4":
        # 0b000 = 0
        # 0b001 = 0.0625
        # 0b010 = 8
        # 0b011 = 12
        # 0b100 = 4
        # 0b101 = 6
        # 0b110 = 2
        # 0b111 = 3
        # can also be created with bnb.functional.create_fp8_map(signed=True, exponent_bits=2, precision_bits=1, total_bits=4)
        data = [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0]
    elif typename == "int4":
        data = [7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7]
    elif typename == "af4":
        # Taken from: NF4 Isn't Information Theoretically Optimal (and that's Good)
        # https://arxiv.org/abs/2306.06965
        if blocksize == 64:
            data = [
                -1.0,
                -0.69441008,
                -0.51243739,
                -0.3736951,
                -0.25607552,
                -0.14982478,
                -0.04934812,
                0.0,
                0.04273164,
                0.12934483,
                0.21961274,
                0.31675666,
                0.42563882,
                0.55496234,
                0.72424863,
                1.0,
            ][::-1]
        else:
            raise NotImplementedError("4-bit AbnormalFloats currently only support blocksize 64.")

    if data is None:
        raise NotImplementedError(f"Typename {typename} not supported")

    data = torch.tensor(data, device=device)
    data.div_(data.abs().max())

    assert data.numel() == 16

    return data


def quantize_fp4(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=64,
    compress_statistics=False,
    quant_storage=torch.uint8,
):
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "fp4", quant_storage)


def quantize_nf4(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=64,
    compress_statistics=False,
    quant_storage=torch.uint8,
):
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "nf4", quant_storage)


def quantize_4bit(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize=64,
    compress_statistics=False,
    quant_type="fp4",
    quant_storage=torch.uint8,
) -> tuple[torch.Tensor, QuantState]:
    """Quantize tensor A in blocks of 4-bit values.

    Quantizes tensor A by dividing it into blocks which are independently quantized.

    Args:
        A (`torch.Tensor`): The input tensor. Supports `float16`, `bfloat16`, or `float32` datatypes.
        absmax (`torch.Tensor`, *optional*): A tensor to use to store the absmax values.
        out (`torch.Tensor`, *optional*): A tensor to use to store the result.
        blocksize (`int`, *optional*):
            The size of the blocks. Defaults to 64.
            Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
        compress_statistics (`bool`, *optional*): Whether to additionally quantize the absmax values. Defaults to False.
        quant_type (`str`, *optional*): The data type to use: `nf4` or `fp4`. Defaults to `fp4`.
        quant_storage (`torch.dtype`, *optional*): The dtype of the tensor used to store the result. Defaults to `torch.uint8`.

    Raises:
        ValueError: Raised when the input data type is not supported.

    Returns:
        Tuple[`torch.Tensor`, `QuantState`]: A tuple containing the quantization results.
        - `torch.Tensor`: The quantized tensor with packed 4-bit values.
        - [`QuantState`]: The state object used to undo the quantization.
    """
    input_shape = A.shape

    _out, _absmax = torch.ops.bitsandbytes.quantize_4bit.default(
        A,
        blocksize,
        quant_type,
        quant_storage,
    )

    code = get_4bit_type(quant_type, device=A.device)

    if compress_statistics:
        offset = _absmax.mean()
        qabsmax, state2 = quantize_blockwise(_absmax - offset, blocksize=256)
        del _absmax
        state = QuantState(
            absmax=qabsmax,
            shape=input_shape,
            dtype=A.dtype,
            blocksize=blocksize,
            code=code,
            quant_type=quant_type,
            offset=offset,
            state2=state2,
        )
    else:
        state = QuantState(
            absmax=_absmax,
            shape=input_shape,
            dtype=A.dtype,
            blocksize=blocksize,
            code=code,
            quant_type=quant_type,
        )

    # TODO(matthewdouglas): Deprecate out kwarg
    out = out.copy_(_out) if out is not None else _out

    # TODO(matthewdouglas): Deprecate absmax kwarg
    if absmax is not None:
        state.absmax = absmax.copy_(state.absmax)

    return out, state


def dequantize_fp4(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 64,
) -> torch.Tensor:
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "fp4")


def dequantize_nf4(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 64,
) -> torch.Tensor:
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "nf4")


def dequantize_4bit(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 64,
    quant_type="fp4",
) -> torch.Tensor:
    """Dequantizes a packed 4-bit quantized tensor.

    The input tensor is dequantized by dividing it into blocks of `blocksize` values.
    The the absolute maximum value within these blocks is used for scaling
    the non-linear dequantization.

    Args:
        A (`torch.Tensor`): The quantized input tensor.
        quant_state ([`QuantState`], *optional*):
            The quantization state as returned by [`quantize_4bit`].
            Required if `absmax` is not provided.
        absmax (`torch.Tensor`, *optional*):
            A tensor containing the scaling values.
            Required if `quant_state` is not provided and ignored otherwise.
        out (`torch.Tensor`, *optional*): A tensor to use to store the result.
        blocksize (`int`, *optional*):
            The size of the blocks. Defaults to 64.
            Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
        quant_type (`str`, *optional*): The data type to use: `nf4` or `fp4`. Defaults to `fp4`.

    Raises:
        ValueError: Raised when the input data type or blocksize is not supported.

    Returns:
        `torch.Tensor`: The dequantized tensor.
    """
    if quant_state is None:
        assert absmax is not None and out is not None

        quant_state = QuantState(
            absmax=absmax,
            shape=out.shape,
            dtype=out.dtype,
            blocksize=blocksize,
            quant_type=quant_type,
        )

    else:
        absmax = quant_state.absmax

    if quant_state.nested:
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()

    if out is not None:
        torch.ops.bitsandbytes.dequantize_4bit.out(
            A, absmax, quant_state.blocksize, quant_state.quant_type, quant_state.shape, quant_state.dtype, out=out
        )
    else:
        out = torch.ops.bitsandbytes.dequantize_4bit.default(
            A,
            absmax,
            quant_state.blocksize,
            quant_state.quant_type,
            quant_state.shape,
            quant_state.dtype,
        )

    if A.shape[0] == 1:  # is transposed, transpose back
        return out.t()
    return out


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def quantize(
    A: Tensor,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    if code is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]
        code = code.to(A.device)

    absmax = torch.abs(A).max()
    if absmax.dtype != torch.float32:
        absmax = absmax.float()
    inp = A / absmax
    out = quantize_no_absmax(inp, code, out)
    return out, (absmax, code)


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def dequantize(
    A: Tensor,
    state: Optional[tuple[Tensor, Tensor]] = None,
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tensor:
    assert state is not None or absmax is not None
    if code is None and state is None:
        if "dynamic" not in name2qmap:
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]
        code = code.to(A.device)

    if state is None:
        state = (absmax, code)
    out = dequantize_no_absmax(A, state[1], out)
    return out * state[0]


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def quantize_no_absmax(A: Tensor, code: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    """
    Quantizes input tensor to 8-bit.

    Quantizes the 32-bit input tensor `A` to the 8-bit output tensor
    `out` using the quantization map `code`.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor, optional
        The output tensor. Needs to be of type byte.

    Returns
    -------
    torch.Tensor:
        Quantized 8-bit tensor.
    """
    with _cuda_device_of(A):
        if out is None:
            out = torch.zeros_like(A, dtype=torch.uint8)
        is_on_gpu([A, out])
        lib.cquantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))

    return out


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def dequantize_no_absmax(A: Tensor, code: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    """
    Dequantizes the 8-bit tensor to 32-bit.

    Dequantizes the 8-bit tensor `A` to the 32-bit tensor `out` via
    the quantization map `code`.

    Parameters
    ----------
    A : torch.Tensor
        The 8-bit input tensor.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor
        The 32-bit output tensor.

    Returns
    -------
    torch.Tensor:
        32-bit output tensor.
    """
    with _cuda_device_of(A):
        if out is None:
            out = torch.zeros_like(A, dtype=torch.float32)
        is_on_gpu([code, A, out])
        stream = _get_tensor_stream(A)
        lib.cdequantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()), stream)

    return out


def optimizer_update_32bit(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    beta1: float,
    eps: float,
    step: int,
    lr: float,
    state2: Optional[torch.Tensor] = None,
    beta2: float = 0.0,
    beta3: float = 0.0,
    alpha: float = 0.0,
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Optional[torch.Tensor] = None,
    max_unorm: float = 0.0,
    skip_zeros=False,
) -> None:
    """
    Performs an inplace optimizer update with one or two optimizer states.

    Universal optimizer update for 32-bit state and 32/16-bit gradients/weights.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer: {adam}.
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Optimizer state 1.
    beta1 : float
        Optimizer beta1.
    eps : float
        Optimizer epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    state2 : torch.Tensor
        Optimizer state 2.
    beta2 : float
        Optimizer beta2.
    beta3 : float
        Optimizer beta3.
    alpha : float
        Optimizer alpha.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    unorm_vec : torch.Tensor
        The tensor for the update norm.
    max_unorm : float
        The maximum update norm relative to the weight norm.
    skip_zeros : bool
        Whether to skip zero-valued gradients or not (default: False).
    """

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    optim_func = None
    if g.dtype == torch.float32:
        optim_func = str2optimizer32bit[optimizer_name][0]
    elif g.dtype == torch.float16:
        optim_func = str2optimizer32bit[optimizer_name][1]
    elif g.dtype == torch.bfloat16 and len(str2optimizer32bit[optimizer_name]) == 3:
        optim_func = str2optimizer32bit[optimizer_name][2]
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}",
        )

    is_on_gpu([g, p, state1, state2, unorm_vec])

    with _cuda_device_of(g):
        optim_func(
            get_ptr(g),
            get_ptr(p),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(beta3),
            ct.c_float(alpha),
            ct.c_float(eps),
            ct.c_float(weight_decay),
            ct.c_int32(step),
            ct.c_float(lr),
            ct.c_float(gnorm_scale),
            ct.c_bool(skip_zeros),
            ct.c_int32(g.numel()),
        )


@deprecated(
    "This function is deprecated and will be removed in a future release. "
    "Please use optimizer_update_8bit_blockwise instead. ",
    category=FutureWarning,
)
def optimizer_update_8bit(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    state2: Optional[torch.Tensor],
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    lr: float,
    qmap1: Tensor,
    qmap2: Optional[torch.Tensor],
    max1: Tensor,
    max2: Optional[torch.Tensor],
    new_max1: Tensor,
    new_max2: Optional[torch.Tensor],
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Optional[torch.Tensor] = None,
    max_unorm: float = 0.0,
) -> None:
    """
    Performs an inplace Adam update.

    Universal Adam update for 32/8-bit state and 32/16-bit gradients/weights.
    Uses AdamW formulation if weight decay > 0.0.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer. Choices {adam, momentum}
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Adam state 1.
    state2 : torch.Tensor
        Adam state 2.
    beta1 : float
        Adam beta1.
    beta2 : float
        Adam beta2.
    eps : float
        Adam epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    qmap1 : torch.Tensor
        Quantization map for first Adam state.
    qmap2 : torch.Tensor
        Quantization map for second Adam state.
    max1 : torch.Tensor
        Max value for first Adam state update.
    max2 : torch.Tensor
        Max value for second Adam state update.
    new_max1 : torch.Tensor
        Max value for the next Adam update of the first state.
    new_max2 : torch.Tensor
        Max value for the next Adam update of the second state.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    unorm_vec : torch.Tensor
        The tensor for the update norm.
    max_unorm : float
        The maximum update norm relative to the weight norm.
    """

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    with _cuda_device_of(g):
        is_on_gpu([g, p, state1, state2, unorm_vec, qmap1, qmap2, max1, max2, new_max1, new_max2])
        if g.dtype == torch.float32 and state1.dtype == torch.uint8:
            str2optimizer8bit[optimizer_name][0](
                get_ptr(p),
                get_ptr(g),
                get_ptr(state1),
                get_ptr(state2),
                get_ptr(unorm_vec),
                ct.c_float(max_unorm),
                ct.c_float(param_norm),
                ct.c_float(beta1),
                ct.c_float(beta2),
                ct.c_float(eps),
                ct.c_int32(step),
                ct.c_float(lr),
                get_ptr(qmap1),
                get_ptr(qmap2),
                get_ptr(max1),
                get_ptr(max2),
                get_ptr(new_max1),
                get_ptr(new_max2),
                ct.c_float(weight_decay),
                ct.c_float(gnorm_scale),
                ct.c_int32(g.numel()),
            )
        elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
            str2optimizer8bit[optimizer_name][1](
                get_ptr(p),
                get_ptr(g),
                get_ptr(state1),
                get_ptr(state2),
                get_ptr(unorm_vec),
                ct.c_float(max_unorm),
                ct.c_float(param_norm),
                ct.c_float(beta1),
                ct.c_float(beta2),
                ct.c_float(eps),
                ct.c_int32(step),
                ct.c_float(lr),
                get_ptr(qmap1),
                get_ptr(qmap2),
                get_ptr(max1),
                get_ptr(max2),
                get_ptr(new_max1),
                get_ptr(new_max2),
                ct.c_float(weight_decay),
                ct.c_float(gnorm_scale),
                ct.c_int32(g.numel()),
            )
        else:
            raise ValueError(
                f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}",
            )


def optimizer_update_8bit_blockwise(
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    state2: Optional[torch.Tensor],
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    eps: float,
    step: int,
    lr: float,
    qmap1: Tensor,
    qmap2: Optional[torch.Tensor],
    absmax1: Tensor,
    absmax2: Optional[torch.Tensor],
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    skip_zeros=False,
) -> None:
    optim_func = None

    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        optim_func = str2optimizer8bit_blockwise[optimizer_name][0]
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        optim_func = str2optimizer8bit_blockwise[optimizer_name][1]
    elif (
        g.dtype == torch.bfloat16
        and state1.dtype == torch.uint8
        and len(str2optimizer8bit_blockwise[optimizer_name]) == 3
    ):
        optim_func = str2optimizer8bit_blockwise[optimizer_name][2]
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}",
        )

    is_on_gpu([p, g, state1, state2, qmap1, qmap2, absmax1, absmax2])

    with _cuda_device_of(g):
        optim_func(
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(beta3),
            ct.c_float(alpha),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(absmax1),
            get_ptr(absmax2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_bool(skip_zeros),
            ct.c_int32(g.numel()),
        )


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def percentile_clipping(grad: Tensor, gnorm_vec: Tensor, step: int, percentile: int = 5):
    """Applies percentile clipping

    grad: torch.Tensor
        The gradient tensor.
    gnorm_vec: torch.Tensor
        Vector of gradient norms. 100 elements expected.
    step: int
        The current optimization steps (number of past gradient norms).

    """
    with _cuda_device_of(grad):
        is_on_gpu([grad, gnorm_vec])
        if grad.dtype == torch.float32:
            lib.cpercentile_clipping_g32(
                get_ptr(grad),
                get_ptr(gnorm_vec),
                ct.c_int32(step),
                ct.c_int32(grad.numel()),
            )
        elif grad.dtype == torch.float16:
            lib.cpercentile_clipping_g16(
                get_ptr(grad),
                get_ptr(gnorm_vec),
                ct.c_int32(step),
                ct.c_int32(grad.numel()),
            )
        else:
            raise ValueError(f"Gradient type {grad.dtype} not supported!")

    current_gnorm = torch.sqrt(gnorm_vec[step % 100])
    vals, idx = torch.sort(gnorm_vec)
    clip_value = torch.sqrt(vals[percentile])
    gnorm_scale = 1.0

    if current_gnorm > clip_value:
        gnorm_scale = clip_value / current_gnorm

    return current_gnorm, clip_value, gnorm_scale


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def histogram_scatter_add_2d(histogram: Tensor, index1: Tensor, index2: Tensor, source: Tensor):
    assert len(histogram.shape) == 2
    assert histogram.dtype == torch.float32
    assert source.dtype == torch.float32
    assert index1.dtype == torch.int32
    assert index2.dtype == torch.int32

    assert histogram.device.type == "cuda"
    assert index1.device.type == "cuda"
    assert index2.device.type == "cuda"
    assert source.device.type == "cuda"

    maxdim1 = ct.c_int32(histogram.shape[0])
    n = ct.c_int32(index1.numel())
    is_on_gpu([histogram, index1, index2, source])
    lib.chistogram_scatter_add_2d(get_ptr(histogram), get_ptr(index1), get_ptr(index2), get_ptr(source), maxdim1, n)


def check_matmul(A, B, out, transposed_A, transposed_B, expected_type=torch.int8):
    if not torch.cuda.is_initialized():
        torch.cuda.init()
    if A.dtype != expected_type or B.dtype != expected_type:
        raise TypeError(f"Expected torch.int8 input tensors A and B, but got {A.dtype} and {B.dtype}")

    sA = A.shape
    sB = B.shape
    tA = transposed_A
    tB = transposed_B

    correct = True

    if len(sA) == 2 and len(sB) == 2:
        if not tA and not tB and A.shape[1] != B.shape[0]:
            correct = False
        elif tA and not tB and A.shape[0] != B.shape[0]:
            correct = False
        elif tA and tB and A.shape[0] != B.shape[1]:
            correct = False
        elif not tA and tB and A.shape[1] != B.shape[1]:
            correct = False
    elif len(sA) == 3 and len(sB) == 2:
        if not tA and not tB and A.shape[2] != B.shape[0]:
            correct = False
        elif tA and not tB and A.shape[1] != B.shape[0]:
            correct = False
        elif tA and tB and A.shape[1] != B.shape[1]:
            correct = False
        elif not tA and tB and A.shape[2] != B.shape[1]:
            correct = False
    elif len(sA) == 3 and len(sB) == 3:
        if not tA and not tB and A.shape[2] != B.shape[1]:
            correct = False
        elif tA and not tB and A.shape[1] != B.shape[1]:
            correct = False
        elif tA and tB and A.shape[1] != B.shape[2]:
            correct = False
        elif not tA and tB and A.shape[2] != B.shape[2]:
            correct = False

    if out is not None:
        sout = out.shape
        # special case common in backprop
        if not correct and len(sA) == 3 and len(sB) == 3:
            if sout[0] == sA[2] and sout[1] == sB[2] and sA[0] == sB[0] and sA[1] == sB[1]:
                correct = True
    else:
        if len(sA) == 2 and len(sB) == 2:
            if not tA and not tB:
                sout = (sA[0], sB[1])
            elif tA and tB:
                sout = (sA[1], sB[0])
            elif tA and not tB:
                sout = (sA[1], sB[1])
            elif not tA and tB:
                sout = (sA[0], sB[0])
        elif len(sA) == 3 and len(sB) == 2:
            if not tA and not tB:
                sout = (sA[0], sA[1], sB[1])
            elif tA and tB:
                sout = (sA[0], sA[2], sB[0])
            elif tA and not tB:
                sout = (sA[0], sA[2], sB[1])
            elif not tA and tB:
                sout = (sA[0], sA[1], sB[0])
        elif len(sA) == 3 and len(sB) == 3:
            if not tA and not tB:
                sout = (sA[0], sA[1], sB[2])
            elif tA and tB:
                sout = (sA[0], sA[2], sB[1])
            elif tA and not tB:
                sout = (sA[0], sA[2], sB[2])
            elif not tA and tB:
                sout = (sA[0], sA[1], sB[1])

    if not correct:
        raise ValueError(
            f"Tensor dimensions incorrect for matrix mulitiplication: A x B: {sA} x {sB} with transpose for A x B: {tA} x {tB}.",
        )

    return sout


def gemv_4bit(
    A: Tensor,
    B: Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A=False,
    transposed_B=False,
    state=None,
):
    if state is None:
        raise ValueError("state cannot be None. gemv_4bit() requires the state from quantize_4bit()")

    absmax = state.absmax
    if state.nested:
        absmax = dequantize_blockwise(absmax, state.state2) + state.offset

    if out is not None:
        torch.ops.bitsandbytes.gemv_4bit.out(
            A,
            B,
            state.shape,
            absmax,
            state.code,
            state.blocksize,
            out=out,
        )
        return out

    return torch.ops.bitsandbytes.gemv_4bit.default(
        A,
        B,
        state.shape,
        absmax,
        state.code,
        state.blocksize,
    )


def igemm(
    A: Tensor,
    B: Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A=False,
    transposed_B=False,
):
    sout = check_matmul(A, B, out, transposed_A, transposed_B)
    if out is None:
        out = torch.zeros(size=sout, dtype=torch.int32, device=A.device)
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1]:
            return batched_igemm(A, B, out)

    sA = A.shape
    sB = B.shape
    if transposed_A and len(sA) == 2:
        sA = (sA[1], sA[0])
    elif transposed_A and len(sA) == 3:
        sA = (sA[0], sA[2], sA[0])
    if transposed_B and len(sB) == 2:
        sB = (sB[1], sB[0])
    elif transposed_B and len(sB) == 3:
        sB = (sB[0], sB[2], sB[0])
    # this is a mess: cuBLAS expect column major, but PyTorch is row major.
    # So to perform the matrix multiplication, we have to treat A, B, and C matrices
    # (transpose of row major is column major)
    # This means we compute B^T A^T = C^T and we explicitly switch the dimensions of each of these

    # matrices in the input arguments for cuBLAS
    # column major: A @ B = C: [m, k] @ [k, n] = [m, n]
    # row major: B^T @ A^T = C^T: [m, k] @ [k, n] = [m, n]
    # column major with row major layout: B^T @ A^T = C^T: [k, m] @ [n, k] = [n, m]
    if len(sB) == 2:
        if B.stride()[0] == B.shape[1]:
            transposed_B = False
        elif B.stride()[1] == B.shape[0]:
            transposed_B = True
        if len(A.shape) == 2:
            if A.stride()[0] == A.shape[1]:
                transposed_A = False
            elif A.stride()[1] == A.shape[0]:
                transposed_A = True
        else:
            if A.stride()[1] == A.shape[2]:
                transposed_A = False
            elif A.stride()[2] == A.shape[1]:
                transposed_A = True

        if len(sA) == 2:
            n = sA[0]
            ldb = A.stride()[1 if transposed_A else 0]
        elif len(sA) == 3 and len(sB) == 2:
            n = sA[0] * sA[1]
            ldb = sA[2]

        m = sB[1]
        k = sB[0]
        lda = B.stride()[(1 if transposed_B else 0)]
        ldc = sB[1]
    elif len(sB) == 3:
        # special case
        assert len(sA) == 3
        if not (sA[0] == sB[0] and sA[1] == sB[1]):
            raise ValueError(
                f"Only bsi,bso->io supported for tensor contractions, but dims for A x B were: {sA} x {sB}",
            )

        transposed_A = True
        transposed_B = False

        m = sB[2]
        n = sA[2]
        k = sB[0] * sB[1]

        lda = m
        ldb = sA[2]
        ldc = m

    ptr = CUBLAS_Context.get_instance().get_context(A.device)

    # B^T @ A^T = C^T
    # [km, nk -> mn]
    is_on_gpu([B, A, out])
    lib.cigemm(
        ptr,
        ct.c_bool(transposed_B),
        ct.c_bool(transposed_A),
        ct.c_int32(m),
        ct.c_int32(n),
        ct.c_int32(k),
        get_ptr(B),
        get_ptr(A),
        get_ptr(out),
        ct.c_int32(lda),
        ct.c_int32(ldb),
        ct.c_int32(ldc),
    )
    return out


def batched_igemm(
    A: Tensor,
    B: Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A=False,
    transposed_B=False,
):
    if not len(A.shape) == 3 or not len(B.shape) == 3:
        raise ValueError(f"Expected 3-dimensional tensors for bmm, but got shapes A and B: {A.shape} and {B.shape}")
    sout = check_matmul(A, B, out, transposed_A, transposed_B)
    if out is None:
        out = torch.zeros(size=sout, dtype=torch.int32, device=A.device)

    if B.is_contiguous():
        lda = B.stride()[1]
        transposed_A = False
    else:
        s = B.stride()
        if s[0] != B.shape[0]:
            B = B.contiguous()
            lda = B.stride()[1]
        elif s[2] == B.shape[1]:
            transposed_A = True
            lda = B.stride()[2]
        else:
            if s[2] == 1:
                B = B.contiguous()
                lda = B.stride()[1]
            elif s[1] == 1:
                B = B.contiguous()
                lda = B.stride()[1]
            else:
                B = B.contiguous()
                lda = B.stride()[1]

    if A.is_contiguous():
        ldb = A.stride()[1]
        transposed_B = False
    else:
        s = A.stride()
        if s[0] != A.shape[0]:
            A = A.contiguous()
            ldb = A.stride()[1]
            transposed_B = False
        elif s[2] == A.shape[1]:
            ldb = A.stride()[2]
            transposed_B = True
        else:
            A = A.contiguous()
            ldb = A.stride()[1]
            transposed_B = False

    # this is a mess: cuBLAS expect column major, but PyTorch is row major.
    # So to perform the matrix multiplication, we have to treat A, B, and C matrices
    # (transpose of row major is column major)
    # This means we compute B^T A^T = C^T and we explicitly switch the dimensions of each of these
    # matrices in the input arguments for cuBLAS

    # column major: A @ B = C: [batch, m, k] @ [batch, k, n] = [batch, m, n]
    # row major: B^T @ A^T = C^T: [batch, m, k] @ [batch, k, n] = [batch, m, n]
    # column major with row major layout: B^T @ A^T = C^T: [batch, k, m] @ [batch, n, k] = [batch, n, m]
    num_batch = A.shape[0]
    n = A.shape[1]
    m = B.shape[2]
    k = B.shape[1]

    ldc = m

    strideA = B.shape[1] * B.shape[2]
    strideB = A.shape[1] * A.shape[2]
    strideC = A.shape[1] * B.shape[2]

    ptr = CUBLAS_Context.get_instance().get_context(A.device)

    is_on_gpu([B, A, out])
    lib.cbatched_igemm(
        ptr,
        ct.c_bool(transposed_B),
        ct.c_bool(transposed_A),
        ct.c_int32(m),
        ct.c_int32(n),
        ct.c_int32(k),
        get_ptr(B),
        get_ptr(A),
        get_ptr(out),
        ct.c_int32(lda),
        ct.c_int32(ldb),
        ct.c_int32(ldc),
        ct.c_long(strideA),
        ct.c_long(strideB),
        ct.c_long(strideC),
        ct.c_uint32(num_batch),
    )
    return out


def int8_linear_matmul(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32):
    """Performs an 8-bit integer matrix multiplication.

    A linear transformation is applied such that `out = A @ B.T`. When possible, integer tensor core hardware is
    utilized to accelerate the operation.

    Args:
        A (`torch.Tensor`): The first matrix operand with the data type `torch.int8`.
        B (`torch.Tensor`): The second matrix operand with the data type `torch.int8`.
        out (`torch.Tensor`, *optional*): A pre-allocated tensor used to store the result.
        dtype (`torch.dtype`, *optional*): The expected data type of the output. Defaults to `torch.int32`.

    Raises:
        `NotImplementedError`: The operation is not supported in the current environment.
        `RuntimeError`: Raised when the cannot be completed for any other reason.

    Returns:
        `torch.Tensor`: The result of the operation.
    """
    if out is not None:
        torch.ops.bitsandbytes.int8_linear_matmul.out(A, B, out)
        return out

    return torch.ops.bitsandbytes.int8_linear_matmul.default(A, B)


def int8_mm_dequant(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
):
    """Performs dequantization on the result of a quantized int8 matrix multiplication.

    Args:
        A (`torch.Tensor` with dtype `torch.int32`): The result of a quantized int8 matrix multiplication.
        row_stats (`torch.Tensor`): The row-wise quantization statistics for the lhs operand of the matrix multiplication.
        col_stats (`torch.Tensor`): The column-wise quantization statistics for the rhs operand of the matrix multiplication.
        out (`torch.Tensor`, *optional*): A pre-allocated tensor to store the output of the operation.
        bias (`torch.Tensor`, *optional*): An optional bias vector to add to the result.

    Returns:
        `torch.Tensor`: The dequantized result with an optional bias, with dtype `torch.float16`.
    """
    result = torch.ops.bitsandbytes.int8_mm_dequant.default(A, row_stats, col_stats, dtype=torch.float16, bias=bias)

    # TODO(matthewdouglas): Deprecate out kwarg
    if out is not None:
        return out.copy_(result)

    return result


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def get_colrow_absmax(
    A: torch.Tensor,
    row_stats: Optional[torch.Tensor] = None,
    col_stats: Optional[torch.Tensor] = None,
    nnz_block_ptr: Optional[torch.Tensor] = None,
    threshold=0.0,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """ "Determine the quantization statistics for input matrix `A` in accordance to the `LLM.int8()` algorithm.

    The row-wise and column-wise absmax values are determined.

    For more information, see the [LLM.int8() paper](https://arxiv.org/abs/2208.07339).

    <Tip>
    This function is useful for training, but for inference it is advised to use [`get_row_absmax`] instead.
    The column-wise quantization scales are not typically needed in inference scenarios.
    </Tip>

    Args:
        A (`torch.Tensor` with dtype `torch.float16`): Input tensor.
        row_stats (`torch.Tensor`, *optional*): If provided, calculation of row statistics is skipped.
        col_stats (`torch.Tensor`, *optional*): If provided, calculation of column statistics is skipped.
        nnz_block_ptr (`torch.Tensor`, *optional*): Not used.
        threshold (`float`, *optional*):
            An optional threshold for sparse decomposition of outlier features.
            No outliers are held back when 0.0. Defaults to 0.0.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`: A tuple containing quantization statistics.
        - `torch.Tensor` with dtype `torch.float32`: The row-wise quantization statistics.
        - `torch.Tensor` with dtype `torch.float32`: The column-wise quantization statistics.
        - `torch.Tensor` with dtype `torch.bool`, *optional*: A mask indicating the locations of outliers in the input tensor.
    """
    assert A.is_floating_point()

    outlier_mask = None

    if row_stats is None or col_stats is None:
        absA = A.abs().view(-1, A.shape[-1])

        if threshold > 0.0:
            # Filter outliers from stats when enabled
            outlier_mask = absA >= threshold
            absA.masked_fill_(outlier_mask, 0.0)

        if row_stats is None:
            # shape [rows]; unsqueeze(-1) gives [rows,1]
            # We have a CUDA kernel for row max, but not yet for cols.
            row_stats = get_row_absmax(A, threshold)

        if col_stats is None:
            # shape [cols]; unsqueeze(0) gives [1,cols]
            col_stats = absA.amax(dim=0, keepdim=False).float()

    return row_stats, col_stats, outlier_mask


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def get_row_absmax(A: torch.Tensor, threshold=0.0):
    """Determine the quantization statistics for input matrix `A` in accordance to the `LLM.int8()` algorithm.

    For more information, see the [LLM.int8() paper](https://arxiv.org/abs/2208.07339).

    Args:
        A (`torch.Tensor` with dtype `torch.float16`): The input matrix.
        threshold (`float`, *optional*):
            An optional threshold for sparse decomposition of outlier features.
            No outliers are held back when 0.0. Defaults to 0.0.

    Returns:
        `torch.Tensor` with dtype `torch.float32`: The absolute maximum value for each row, with outliers ignored.
    """

    assert A.dtype == torch.float16

    rows = prod(A.shape[:-1])
    cols = A.shape[-1]

    row_stats = torch.empty((rows,), dtype=torch.float32, device=A.device)

    is_on_gpu([A])

    with _cuda_device_of(A):
        lib.cget_row_stats(
            get_ptr(A),
            get_ptr(row_stats),
            ct.c_float(threshold),
            ct.c_int32(rows),
            ct.c_int32(cols),
            _get_tensor_stream(A),
        )

    return row_stats


class COOSparseTensor:
    def __init__(
        self, rows: int, cols: int, nnz: int, rowidx: torch.Tensor, colidx: torch.Tensor, values: torch.Tensor
    ):
        assert rowidx.dtype == torch.int32
        assert colidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colidx.numel() == nnz

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowidx = rowidx
        self.colidx = colidx
        self.values = values


class CSRSparseTensor:
    def __init__(self, rows, cols, nnz, rowptr, colidx, values):
        assert rowptr.dtype == torch.int32
        assert colidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert colidx.numel() == nnz
        assert rowptr.numel() == rows + 1

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.rowptr = rowptr
        self.colidx = colidx
        self.values = values


class CSCSparseTensor:
    def __init__(self, rows, cols, nnz, colptr, rowidx, values):
        assert colptr.dtype == torch.int32
        assert rowidx.dtype == torch.int32
        assert values.dtype == torch.float16
        assert values.numel() == nnz
        assert rowidx.numel() == nnz
        assert colptr.numel() == cols + 1

        self.rows = rows
        self.cols = cols
        self.nnz = nnz
        self.colptr = colptr
        self.rowidx = rowidx
        self.values = values


def coo2csr(cooA):
    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    values.add_(1)
    rowptr = torch.zeros((cooA.rows + 1,), dtype=torch.int32, device=cooA.rowidx.device)
    rowptr.scatter_(index=values.long(), src=counts.int(), dim=0)
    rowptr.cumsum_(0)
    return CSRSparseTensor(cooA.rows, cooA.cols, cooA.nnz, rowptr, cooA.colidx, cooA.values)


def coo2csc(cooA):
    val, col2rowidx = torch.sort(cooA.colidx)
    rowidx = cooA.rowidx[col2rowidx]
    values = cooA.values[col2rowidx]
    colvalues, counts = torch.unique(val, return_counts=True)
    colvalues.add_(1)
    colptr = torch.zeros((cooA.cols + 1,), dtype=torch.int32, device=cooA.colidx.device)
    colptr.scatter_(index=colvalues.long(), src=counts.int(), dim=0)
    colptr.cumsum_(0)
    return CSCSparseTensor(cooA.rows, cooA.cols, cooA.nnz, colptr, rowidx, values)


def coo_zeros(rows, cols, nnz, device, dtype=torch.half):
    rowidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
    colidx = torch.zeros((nnz,), dtype=torch.int32, device=device)
    values = torch.zeros((nnz,), dtype=dtype, device=device)
    return COOSparseTensor(rows, cols, nnz, rowidx, colidx, values)


def int8_double_quant(
    A: torch.Tensor,
    col_stats: Optional[torch.Tensor] = None,
    row_stats: Optional[torch.Tensor] = None,
    out_col: Optional[torch.Tensor] = None,
    out_row: Optional[torch.Tensor] = None,
    threshold=0.0,
):
    """Determine the quantization statistics for input matrix `A` in accordance to the `LLM.int8()` algorithm.

    The statistics are determined both row-wise and column-wise (transposed).

    For more information, see the [LLM.int8() paper](https://arxiv.org/abs/2208.07339).

    <Tip>
    This function is useful for training, but for inference it is advised to use [`int8_vectorwise_quant`] instead.
    This implementation performs additional column-wise transposed calculations which are not optimized.
    </Tip>

    Args:
        A (`torch.Tensor` with dtype `torch.float16`): The input matrix.
        col_stats (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the column-wise quantization scales.
        row_stats (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the row-wise quantization scales.
        out_col (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the column-wise quantized data.
        out_row (`torch.Tensor`, *optional*): A pre-allocated tensor to hold the row-wise quantized data.
        threshold (`float`, *optional*):
            An optional threshold for sparse decomposition of outlier features.

            No outliers are held back when 0.0. Defaults to 0.0.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`: A tuple containing the quantized tensor and relevant statistics.
        - `torch.Tensor` with dtype `torch.int8`: The row-wise quantized data.
        - `torch.Tensor` with dtype `torch.int8`: The column-wise quantized data.
        - `torch.Tensor` with dtype `torch.float32`: The row-wise quantization scales.
        - `torch.Tensor` with dtype `torch.float32`: The column-wise quantization scales.
        - `torch.Tensor` with dtype `torch.int32`, *optional*: A list of column indices which contain outlier features.
    """

    if row_stats is not None:
        raise ValueError("row_stats must be None. int8_double_quant() does not support pre-allocated row_stats.")
    if col_stats is not None:
        raise ValueError("col_stats must be None. int8_double_quant() does not support pre-allocated col_stats.")
    if out_col is not None:
        raise ValueError("out_col must be None. int8_double_quant() does not support pre-allocated out_col.")
    if out_row is not None:
        raise ValueError("out_row must be None. int8_double_quant() does not support pre-allocated out_row.")

    return torch.ops.bitsandbytes.int8_double_quant.default(A, threshold=threshold)


def int8_vectorwise_dequant(A: torch.Tensor, stats: torch.Tensor):
    """Dequantizes a tensor with dtype `torch.int8` to `torch.float32`.

    Args:
        A (`torch.Tensor` with dtype `torch.int8`): The quantized int8 tensor.
        stats (`torch.Tensor` with dtype `torch.float32`): The row-wise quantization statistics.

    Returns:
        `torch.Tensor` with dtype `torch.float32`: The dequantized tensor.
    """
    # To dequantize we divide by 127, or multiply by the reciprocal.
    return torch.ops.bitsandbytes.int8_vectorwise_dequant.default(A, stats)


def int8_vectorwise_quant(A: torch.Tensor, threshold=0.0):
    """Quantizes a tensor with dtype `torch.float16` to `torch.int8` in accordance to the `LLM.int8()` algorithm.

    For more information, see the [LLM.int8() paper](https://arxiv.org/abs/2208.07339).

    Args:
        A (`torch.Tensor` with dtype `torch.float16`): The input tensor.
        threshold (`float`, *optional*):
            An optional threshold for sparse decomposition of outlier features.

            No outliers are held back when 0.0. Defaults to 0.0.

    Returns:
        `Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`: A tuple containing the quantized tensor and relevant statistics.
        - `torch.Tensor` with dtype `torch.int8`: The quantized data.
        - `torch.Tensor` with dtype `torch.float32`: The quantization scales.
        - `torch.Tensor` with dtype `torch.int32`, *optional*: A list of column indices which contain outlier features.
    """
    return torch.ops.bitsandbytes.int8_vectorwise_quant.default(A, threshold)


def spmm_coo(
    cooA: Union[COOSparseTensor, torch.Tensor],
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
):
    if not isinstance(cooA, COOSparseTensor):
        assert cooA.is_sparse and cooA.layout == torch.sparse_coo, (
            "Tensor must be `COOSparseTensor or a PyTorch COO tensor."
        )

        # Convert to custom COOSparseTensor
        cooA = COOSparseTensor(
            rows=cooA.shape[0],
            cols=cooA.shape[1],
            nnz=cooA._nnz(),
            rowidx=cooA.indices()[0].int(),
            colidx=cooA.indices()[1].int(),
            values=cooA.values(),
        )

    if out is None:
        out = torch.empty((cooA.rows, B.shape[1]), device=B.device, dtype=B.dtype)
    nnz = cooA.nnz
    assert cooA.rowidx.numel() == nnz
    assert cooA.colidx.numel() == nnz
    assert cooA.values.numel() == nnz
    assert cooA.cols == B.shape[0]

    transposed_B = False if B.is_contiguous() else True

    ldb = B.stride()[(1 if transposed_B else 0)]
    ldc = B.shape[1]

    ptr = Cusparse_Context.get_instance().context

    ptrRowidx = get_ptr(cooA.rowidx)
    ptrColidx = get_ptr(cooA.colidx)
    ptrValues = get_ptr(cooA.values)
    ptrB = get_ptr(B)
    ptrC = get_ptr(out)
    cnnz = ct.c_int32(cooA.nnz)
    crowsA = ct.c_int32(cooA.rows)
    ccolsA = ct.c_int32(cooA.cols)
    ccolsB = ct.c_int32(B.shape[1])
    cldb = ct.c_int32(ldb)
    cldc = ct.c_int32(ldc)

    is_on_gpu([cooA.rowidx, cooA.colidx, cooA.values, B, out])
    lib.cspmm_coo(
        ptr,
        ptrRowidx,
        ptrColidx,
        ptrValues,
        cnnz,
        crowsA,
        ccolsA,
        ccolsB,
        cldb,
        ptrB,
        cldc,
        ptrC,
        ct.c_bool(transposed_B),
    )

    return out


def spmm_coo_very_sparse(cooA, B, dequant_stats=None, out=None):
    if out is None:
        out = torch.zeros((cooA.rows, B.shape[1]), device=B.device, dtype=cooA.values.dtype)
    nnz = cooA.nnz

    assert cooA.rowidx.numel() == nnz
    assert cooA.colidx.numel() == nnz
    assert cooA.values.numel() == nnz
    assert cooA.cols == B.shape[0], f"{cooA.cols} vs {B.shape}"

    transposed_B = False if B.is_contiguous() else True

    ldb = B.stride()[(1 if transposed_B else 0)]
    ldc = B.shape[1]

    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    offset = counts.cumsum(0).int()
    max_count, max_idx = torch.sort(counts, descending=True)
    max_idx = max_idx.int()
    max_count = max_count.int()
    assert max_count[0] <= 32, f"Current max count per row is 8 but found {max_count[0]}."
    assert B.dtype in [torch.float16, torch.int8]
    ptrOffset = get_ptr(offset)
    ptrMaxCount = get_ptr(max_count)
    ptrMaxIdx = get_ptr(max_idx)

    ptrRowidx = get_ptr(cooA.rowidx)
    ptrColidx = get_ptr(cooA.colidx)
    ptrValues = get_ptr(cooA.values)
    ptrB = get_ptr(B)
    ptrC = get_ptr(out)
    ptrDequantStats = get_ptr(dequant_stats)
    cnnz_rows = ct.c_int32(counts.numel())
    cnnz = ct.c_int32(cooA.nnz)
    crowsA = ct.c_int32(cooA.rows)
    ccolsA = ct.c_int32(cooA.cols)
    crowsB = ct.c_int32(B.shape[1])
    ccolsB = ct.c_int32(B.shape[1])
    cldb = ct.c_int32(ldb)
    cldc = ct.c_int32(ldc)

    with _cuda_device_of(B):
        is_on_gpu([cooA.rowidx, cooA.colidx, cooA.values, B, out, dequant_stats])
        if B.dtype == torch.float16:
            lib.cspmm_coo_very_sparse_naive_fp16(
                ptrMaxCount,
                ptrMaxIdx,
                ptrOffset,
                ptrRowidx,
                ptrColidx,
                ptrValues,
                ptrB,
                ptrC,
                ptrDequantStats,
                cnnz_rows,
                cnnz,
                crowsA,
                crowsB,
                ccolsB,
            )
        elif B.dtype == torch.int8:
            lib.cspmm_coo_very_sparse_naive_int8(
                ptrMaxCount,
                ptrMaxIdx,
                ptrOffset,
                ptrRowidx,
                ptrColidx,
                ptrValues,
                ptrB,
                ptrC,
                ptrDequantStats,
                cnnz_rows,
                cnnz,
                crowsA,
                crowsB,
                ccolsB,
            )
        # else: assertion error

    return out


C = 127.0


@deprecated(
    "This function is deprecated and will be removed in a future release. "
    "Consider using `int8_vectorwise_quant` instead.",
    category=FutureWarning,
)
def vectorwise_quant(x, dim=1, quant_type="vector"):
    if quant_type == "linear":
        max1 = torch.abs(x).max().float()
        xq = torch.round(x / max1 * 127).to(torch.int8)
        return xq, max1
    elif quant_type in ["vector", "row"]:
        max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        xq = torch.round(x * (C / max1)).to(torch.int8)
        return xq, max1
    elif quant_type == "zeropoint":
        dtype = x.dtype
        x = x.float()
        dyna = x.max() - x.min()
        if dyna == 0:
            dyna = 1
        qx = 255.0 / dyna
        minx = x.min()
        zpx = torch.round(minx * qx)
        x = torch.round(qx * x - zpx) + zpx
        return x, qx
    elif quant_type in ["vector-zeropoint", "row-zeropoint"]:
        dtype = x.dtype
        x = x.float()
        dyna = torch.amax(x, dim=dim, keepdim=True) - torch.amin(x, dim=dim, keepdim=True)
        dyna[dyna == 0] = 1
        qx = 255.0 / dyna
        minx = torch.amin(x, dim=dim, keepdim=True)
        zpx = torch.round(minx * qx)
        x = torch.round(qx * x - zpx) + zpx
        return x, qx
    elif quant_type == "truncated-vector":
        with torch.no_grad():
            absx = torch.abs(x)
            max1 = torch.amax(absx, dim=dim, keepdim=True)
            max1 = max1 * 0.7
            idx = absx > max1.expand_as(absx)
            sign = torch.sign(x[idx])
            x[idx] = max1.expand_as(absx)[idx] * sign
            xq = torch.round(x / max1 * C).to(torch.int8)
        return xq, max1
    else:
        return None


@deprecated(
    "This function is deprecated and will be removed in a future release.",
    category=FutureWarning,
)
def vectorwise_mm_dequant(xq, S1, S2, dtype=torch.half, quant_type="vector"):
    if quant_type == "linear":
        norm = S1 * S2 / (C * C)
        # double cast needed to prevent overflows
        return (xq.float() * norm).to(dtype)
    elif quant_type == "zeropoint":
        norm = 1.0 / (S1 * S2)
        return (xq.float() * norm).to(dtype)
    elif quant_type == "row-zeropoint":
        norm = 1.0 / (S1 * S2)
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= norm
        else:
            x *= norm
        return x.to(dtype)
    elif quant_type == "vector-zeropoint":
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= 1.0 / S1
        else:
            x *= 1.0 / S1
        x *= 1.0 / S2.t()
        return x.to(dtype)
    elif quant_type == "row":
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= S1 * S2 / (C * C)
        else:
            x *= S1 * S2 / (C * C)
        return x.to(dtype)
    elif quant_type in ["truncated-vector", "vector"]:
        x = xq.float()
        if len(S1.shape) == 3 and len(x.shape) == 2:
            S1 = S1.squeeze(0)
        if len(S2.shape) == 3 and len(x.shape) == 2:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= S1 / C
        else:
            x *= S1 / C
        x *= S2 / C
        return x.to(dtype)
    else:
        return None
'''


from collections.abc import Iterable, Sequence 
import ctypes as ct
import itertools
from functools import reduce 
import operator 
from math import prod as math_prod 
from typing import Any, Optional, Union, Tuple

import numpy as np
import torch
from torch import Tensor
from typing_extensions import deprecated 

from .cextension import lib, HIP_ENVIRONMENT, BNB_HIP_VERSION 


def prod_compat(iterable):
    return reduce(operator.mul, iterable, 1)


try:
    from math import prod as math_prod
except ImportError:
    math_prod = prod_compat


name2qmap = {}

str2optimizer32bit = {}
str2optimizer8bit_blockwise = {}

str2optimizer8bit = {}


if lib and getattr(lib, 'compiled_with_cuda', False):
    str2optimizer32bit = {
        "adam": (
            lib.cadam32bit_grad_fp32,
            lib.cadam32bit_grad_fp16,
            lib.cadam32bit_grad_bf16,
        ),
        "momentum": (
            lib.cmomentum32bit_grad_32, 
            lib.cmomentum32bit_grad_16, 
                                    
        ),
        "rmsprop": (
            lib.crmsprop32bit_grad_32,
            lib.crmsprop32bit_grad_16,
        ),
        "lion": (
            lib.clion32bit_grad_fp32,
            lib.clion32bit_grad_fp16,
            lib.clion32bit_grad_bf16,
        ),
        "adagrad": (
            lib.cadagrad32bit_grad_32,
            lib.cadagrad32bit_grad_16,
        ),
        "lamb": ( 
            lib.cadam32bit_grad_fp32, 
            lib.cadam32bit_grad_fp16,
            lib.cadam32bit_grad_bf16,
        ),

        "ademamix": (
            lib.cademamix32bit_grad_fp32,
            lib.cademamix32bit_grad_fp16,
            lib.cademamix32bit_grad_bf16,
        ),
    }

    str2optimizer8bit_blockwise = {
        "adam": (
            lib.cadam_8bit_blockwise_grad_fp32,
            lib.cadam_8bit_blockwise_grad_fp16,
            lib.cadam_8bit_blockwise_grad_bf16,
        ),
        "momentum": (
            lib.cmomentum_8bit_blockwise_grad_fp32,
            lib.cmomentum_8bit_blockwise_grad_fp16,
            lib.cmomentum_8bit_blockwise_grad_bf16,
        ),
        "rmsprop": (
            lib.crmsprop_8bit_blockwise_grad_fp32,
            lib.crmsprop_8bit_blockwise_grad_fp16,
            lib.crmsprop_8bit_blockwise_grad_bf16,
        ),
        "lion": (
            lib.clion_8bit_blockwise_grad_fp32,
            lib.clion_8bit_blockwise_grad_fp16,
            lib.clion_8bit_blockwise_grad_bf16,
        ),
        "adagrad": (
            lib.cadagrad_8bit_blockwise_grad_fp32,
            lib.cadagrad_8bit_blockwise_grad_fp16,
            lib.cadagrad_8bit_blockwise_grad_bf16,
        ),

        "ademamix": (
            lib.cademamix_8bit_blockwise_grad_fp32,
            lib.cademamix_8bit_blockwise_grad_fp16,
            lib.cademamix_8bit_blockwise_grad_bf16,
        ),
    }

    str2optimizer8bit = {
        "adam": (
            lib.cadam_static_8bit_grad_32,
            lib.cadam_static_8bit_grad_16,
        ),
        "momentum": (
            lib.cmomentum_static_8bit_grad_32,
            lib.cmomentum_static_8bit_grad_16,
        ),
        "rmsprop": (
            lib.crmsprop_static_8bit_grad_32,
            lib.crmsprop_static_8bit_grad_16,
        ),
        "lion": (
            lib.clion_static_8bit_grad_32,
            lib.clion_static_8bit_grad_16,
        ),
        "lamb": (
            lib.cadam_static_8bit_grad_32, 
            lib.cadam_static_8bit_grad_16,
        ),
        "lars": ( 
            lib.cmomentum_static_8bit_grad_32, 
            lib.cmomentum_static_8bit_grad_16,
        ),
    }


class GlobalPageManager: 
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.paged_tensors = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def prefetch_all(self, to_cpu=False):
        for t in self.paged_tensors[::-1]:
            prefetch_tensor(t, to_cpu)


class CUBLAS_Context: 
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.context = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def get_context(self, device: torch.device):
        if device.index not in self.context:
            prev_device_idx = torch.cuda.current_device()
            torch.cuda.set_device(device)
            self.context[device.index] = ct.c_void_p(lib.get_context())
            torch.cuda.set_device(prev_device_idx)
        return self.context[device.index]


class Cusparse_Context: # As in Main PDF, for cuSPARSE/hipSPARSE
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        global HIP_ENVIRONMENT
        if HIP_ENVIRONMENT:
            if hasattr(lib, 'get_hipsparse'):
                self.context = ct.c_void_p(lib.get_hipsparse())
            else: 
                logger.warning("lib.get_hipsparse not found. Sparse operations might fail on ROCm.")
                self.context = None
        else:
            if hasattr(lib, 'get_cusparse'):
                self.context = ct.c_void_p(lib.get_cusparse())
            else: 
                logger.warning("lib.get_cusparse not found. Sparse operations might fail on CUDA.")
                self.context = None


    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

dtype2bytes = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.uint8: 1,
    torch.int8: 1,
}


FIRST_CUDA_DEVICE = torch.device("cuda", index=0) 

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    def _cuda_device_of(a: torch.Tensor): 
        return torch.cuda.device_of(a)
else:
    import contextlib
    def _cuda_device_of(a: torch.Tensor):
        return contextlib.nullcontext()


def get_paged(*shape, dtype=torch.float32, device=FIRST_CUDA_DEVICE): 
    num_bytes = dtype2bytes.get(dtype, 0) * math_prod(shape)
    if num_bytes == 0 and dtype not in dtype2bytes:
        raise ValueError(f"Unsupported dtype for paged tensor: {dtype}")
    cuda_ptr = lib.cget_managed_ptr(ct.c_size_t(num_bytes))
    c_ptr = ct.cast(cuda_ptr, ct.POINTER(ct.c_byte)) 

    if dtype == torch.float32: ctype_dtype = ct.c_float
    elif dtype == torch.float16: ctype_dtype = ct.c_uint16 
    elif dtype == torch.bfloat16: ctype_dtype = ct.c_uint16 
    elif dtype == torch.uint8: ctype_dtype = ct.c_uint8
    elif dtype == torch.int8: ctype_dtype = ct.c_int8
    else: raise TypeError(f"Unsupported dtype for get_paged: {dtype}")

    c_typed_ptr = ct.cast(cuda_ptr, ct.POINTER(ctype_dtype))
       
    buffer_size = num_bytes // dtype2bytes[dtype] # Number of elements
    c_array_type = ctype_dtype * buffer_size
    actual_c_array = c_array_type.from_address(cuda_ptr)
    
    out = torch.from_numpy(np.frombuffer(actual_c_array, dtype=dtype.name).reshape(shape))
    np_dtype = dtype.name # e.g. 'float32'
    typed_cuda_ptr = ct.cast(cuda_ptr, ct.POINTER(ctype_dtype))
    np_array = np.ctypeslib.as_array(typed_cuda_ptr, shape=(math_prod(shape),)) 
    out = torch.from_numpy(np_array).reshape(shape)
    out = out.to(device)


    out.is_paged = True # Mark as paged
    out.page_deviceid = device.index
    return out


def prefetch_tensor(A: torch.Tensor, to_cpu=False): # As in Main PDF
    assert getattr(A, 'is_paged', False), "Only paged tensors can be prefetched!"
    if to_cpu:
        deviceid = -1 # Standard CUDA code for host
    else:
        deviceid = A.page_deviceid

    num_bytes = A.nbytes # Use tensor's nbytes property
    lib.cprefetch(get_ptr(A), ct.c_size_t(num_bytes), ct.c_int32(deviceid))


def elementwise_func(func_name, A, B, value, prefetch=True): 
    func = None
    cvalue = None # Initialize cvalue
    if A.dtype == torch.float32:
        func = getattr(lib, f"c{func_name}_fp32", None)
        cvalue = ct.c_float(value)
    elif A.dtype == torch.uint8: # Main PDF example
        func = getattr(lib, f"c{func_name}_uint8", None)
        cvalue = ct.c_uint8(value)


    if func is None:
        raise NotImplementedError(f"Elementwise function '{func_name}' not implemented for dtype {A.dtype}")

    is_managed = getattr(A, "is_managed", False)
    is_paged_A = getattr(A, "is_paged", False)
    is_paged_B = getattr(B, "is_paged", False) if B is not None else False

    if (is_paged_A or is_managed) and prefetch: # Check is_paged
        prefetch_tensor(A)
    if B is not None and is_paged_B and prefetch:
        prefetch_tensor(B)

    func(get_ptr(A), get_ptr(B) if B is not None else None, cvalue, ct.c_int64(A.numel()))

    if is_paged_A or is_paged_B:
        torch.cuda.synchronize()


def fill(A, value, device=None, prefetch=True): 
    elementwise_func("fill", A, None, value, prefetch=prefetch)


def _mul(A, B, device=None):
    elementwise_func("_mul", A, B, 0, prefetch=True) 


def create_linear_map(signed=True, total_bits=8, add_zero=True): 
    sign_val = -1.0 if signed else 0.0
    total_values = 2**total_bits
    if add_zero or total_bits < 8:
        total_values = 2**total_bits if not signed else 2**total_bits - 1
    
    values = torch.linspace(sign_val, 1.0, total_values)
    gap = 256 - values.numel()
    if gap == 0:
        return values
    else:

        mid_idx = values.numel() // 2
        return torch.tensor(values[:mid_idx].tolist() + [0.0] * gap + values[mid_idx:].tolist(), dtype=torch.float32)


def create_normal_map(offset=0.9677083, use_extra_value=True):
    try:
        from scipy.stats import norm
    except ImportError as ie:
        raise ImportError(
            "Scipy is required for 'create_normal_map'. Install 'bitsandbytes' with the [test] extra, e.g. pip install bitsandbytes[test]",
        ) from ie

    if use_extra_value:
        v1 = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1]).tolist()
        v2 = [0.0] * (256 - 15) # 15 non-zero values
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist() 
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, 8)[:-1]).tolist() 
        v2 = [0.0] * (256 - 14) # 14 non-zero values
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist() 

    v = sorted(v1 + v2 + v3) # Ensure sorted order
    values = torch.tensor(v, dtype=torch.float32)
    max_abs_val = values.abs().max()
    if max_abs_val > 0:
        values /= max_abs_val
    
    assert values.numel() == 256
    return values


def create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8): 
    e = exponent_bits
    p = precision_bits
    has_sign_bit = 1 if signed else 0 # Renamed for clarity
    if e + p != total_bits - has_sign_bit:
         raise ValueError(f"Exponent bits ({e}) + Precision bits ({p}) must equal Total bits ({total_bits}) - Sign bit ({has_sign_bit})")

    values = []
    
    exp_bias = 2**(e - 1) - 1

    for exp_pattern in range(2**e):
        for prec_pattern in range(2**p):
            significand = 0.0
            if exp_pattern == 0: 
                # Significand is 0.fraction
                for i in range(p):
                    if (prec_pattern >> (p - 1 - i)) & 1: 
                        significand += 2**-(i + 1)
                actual_exp = 1 - exp_bias
            else: 
                significand = 1.0
                for i in range(p):
                    if (prec_pattern >> (p - 1 - i)) & 1:
                        significand += 2**-(i + 1)
                actual_exp = exp_pattern - exp_bias
            
            value = significand * (2**actual_exp)
            values.append(value)
            if signed and value != 0: 
                values.append(-value)
            elif signed and value == 0 and len(values) < 2**total_bits : 
                pass



    values = sorted(list(set(values)))

    if total_bits < 8:
        num_current_values = len(values)
        gap = 256 - num_current_values
        if gap > 0:
            values.extend([0.0] * gap)
            values = sorted(list(set(values))) 
    if total_bits == 8:
        assert len(values) <= 2**total_bits, f"Generated more values ({len(values)}) than expected ({2**total_bits}) for fp{total_bits}"

    code = torch.tensor(values, dtype=torch.float32)
    

    max_abs = code.abs().max()
    if max_abs > 0:
        code /= max_abs
    return code


def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8): 
    data = []
    non_sign_bits = total_bits - (1 if signed else 0) 
    for i in range(max_exponent_bits):
        bits_for_fraction_at_this_level = (i + non_sign_bits - max_exponent_bits)
        fraction_items = 2**bits_for_fraction_at_this_level + 1
        
        boundaries = torch.linspace(0.1, 1.0, int(fraction_items), dtype=torch.float32) 
        if len(boundaries) < 2: # Need at least 2 boundaries to make means
            means = boundaries # or skip if no means can be formed
        else:
            means = (boundaries[:-1] + boundaries[1:]) / 2.0
        current_exponent_scale = 10**(-(max_exponent_bits - 1) + i)
        
        data.extend((current_exponent_scale * means).tolist())
        if signed:
            data.extend((-current_exponent_scale * means).tolist())

    if non_sign_bits > max_exponent_bits:
        bits_for_finest_fraction = non_sign_bits - max_exponent_bits
        additional_items_count = 2**bits_for_finest_fraction 
        
        num_additional_linspace_points = 2**(non_sign_bits - max_exponent_bits) 
                                                                           
        if num_additional_linspace_points -1 > 0: # Check if we can form means
            boundaries_additional = torch.linspace(0.1, 1.0, int(num_additional_linspace_points), dtype=torch.float32)
            if len(boundaries_additional) >=2:
                means_additional = (boundaries_additional[:-1] + boundaries_additional[1:]) / 2.0
                smallest_exp_scale = 10**(-(max_exponent_bits - 1))
                data.extend((smallest_exp_scale * means_additional).tolist())
                if signed:
                    data.extend((-smallest_exp_scale * means_additional).tolist())


    data.append(0.0) # Add zero

    data = sorted(list(set(data)))

    # Pad to 256 if total_bits < 8
    if total_bits < 8:
        current_len = len(data)
        gap = 256 - current_len
        if gap > 0:
            data.extend([0.0] * gap)
        data = sorted(list(set(data))) 
    if len(data) > 256: data = data[:256] 
    while len(data) < 256: data.append(0.0) 
    data = sorted(data)

    final_map = torch.tensor(data, dtype=torch.float32)
    
  
    max_abs = final_map.abs().max()
    if max_abs > 0:
        final_map /= max_abs
        
    return final_map


@deprecated("This function is deprecated and will be removed in a future release.", category=FutureWarning)
def create_quantile_map(A, total_bits=8): # As in Main PDF
    num_q = 2**total_bits -1
    q_values = estimate_quantiles(A, num_quantiles=num_q)
    
    q_list = q_values.tolist()
    q_list.append(0.0) # Add 0

    gap = 256 - len(q_list)
    if gap > 0:
        q_list.extend([0.0] * gap)
    
    q_list = sorted(list(set(q_list))) 

   
    if len(q_list) > 256: q_list = q_list[:256]
    while len(q_list) < 256: q_list.append(0.0) 
    q_list = sorted(q_list)

    q_tensor = torch.tensor(q_list, dtype=torch.float32)
    
    max_abs = q_tensor.abs().max()
    if max_abs > 0:
        q_tensor /= max_abs
        
    return q_tensor

class QuantState:
    """Container for quantization state components."""
    def __init__(
        self,
        absmax: Optional[torch.Tensor] = None,
        shape: Optional[torch.Size] = None, 
        code: Optional[torch.Tensor] = None,
        blocksize: Optional[int] = None,
        quant_type: Optional[str] = None, 
        dtype: Optional[torch.dtype] = None,
        offset: Optional[torch.Tensor] = None, 
        state2: Optional['QuantState'] = None, 

    ):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize
        self.quant_type = quant_type 
        self.offset = offset
        self.state2 = state2
        self.nested = state2 is not None

    def __getitem__(self, idx: int):
        """
        Ensures compatibility with older quant state scheme which was a nested list:
        state = [absmax, input_shape, A.dtype, blocksize, [offset, state2] or None, quant_type]
        """
        if self.nested:
            nested_content = [self.offset, self.state2]
            list_repr = [
                self.absmax,
                self.shape, 
                self.dtype,
                self.blocksize,
                nested_content,
                self.quant_type,
            ]
        else:
            list_repr = [
                self.absmax,
                self.shape,
                self.dtype,
                self.blocksize,
                None, # No nested state
                self.quant_type,
            ]

        if self.nested:
            # Ensure all components are present if creating this list representation
            _shape = self.shape if self.shape is not None else torch.Size([])
            _dtype = self.dtype if self.dtype is not None else torch.float32
            _blocksize = self.blocksize if self.blocksize is not None else 0
            _quant_type = self.quant_type if self.quant_type is not None else ""
            
            constructed_list = [
                self.absmax, _shape, _dtype, _blocksize,
                [self.offset, self.state2], # Nested part
                _quant_type
            ]
        else:
            _shape = self.shape if self.shape is not None else torch.Size([])
            _dtype = self.dtype if self.dtype is not None else torch.float32
            _blocksize = self.blocksize if self.blocksize is not None else 0
            _quant_type = self.quant_type if self.quant_type is not None else ""
            constructed_list = [
                self.absmax, _shape, _dtype, _blocksize,
                None, 
                _quant_type
            ]
        return constructed_list[idx]


    try:
        from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
    except ImportError:
   
        def pack_dict_to_tensor(d): raise NotImplementedError("pack_dict_to_tensor not found")
        def unpack_tensor_to_dict(t): raise NotImplementedError("unpack_tensor_to_dict not found")

    valid_quant_types_static = ("fp4", "nf4") # For as_dict/from_dict key construction
    valid_qs_type_keys_static = [f"bitsandbytes__{x}" for x in valid_quant_types_static] # Note: refactored used "bitsandbytes_{x}"
    valid_qs_keys_static = [ # From refactored
        "absmax", "quant_map", "nested_absmax", "nested_quant_map",
        "quant_state", "quant_type", "blocksize", "dtype", "shape",
        "nested_blocksize", "nested_dtype", "nested_offset",
    ]


    @classmethod
    def from_dict(cls, qs_dict: dict[str, Any], device: Optional[torch.device] = None) -> "QuantState":
  
        qs_dict_copy = qs_dict.copy()

        packed_state_key = None
        for k in qs_dict_copy.keys():
            if "quant_state.bitsandbytes__" in k and isinstance(qs_dict_copy[k], torch.Tensor):
                packed_state_key = k
                break
        
        if packed_state_key:
            unpacked_items = unpack_tensor_to_dict(qs_dict_copy.pop(packed_state_key))
            qs_dict_copy.update(unpacked_items)

        cleaned_qs_dict = {}
        for k, v in qs_dict_copy.items():
            cleaned_key = k.split(".")[-1] # Takes the last part after any dots
            cleaned_qs_dict[cleaned_key] = v

        target_device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        nested_state2 = None
        nested_offset_tensor = None
        if "nested_absmax" in cleaned_qs_dict:
            nested_offset_val = float(cleaned_qs_dict.get("nested_offset", 0.0))
            nested_offset_tensor = torch.tensor(nested_offset_val, device=target_device, dtype=torch.float32)
            
            nested_state2 = cls(
                absmax=cleaned_qs_dict["nested_absmax"].to(target_device),
                blocksize=int(cleaned_qs_dict["nested_blocksize"]),
                code=cleaned_qs_dict["nested_quant_map"].to(target_device), 
                dtype=getattr(torch, cleaned_qs_dict["nested_dtype"]),
                quant_type=cleaned_qs_dict.get("quant_type"),
                shape=torch.Size(cleaned_qs_dict.get("shape", [])), 
            )
            
     
        shape_data = cleaned_qs_dict.get("shape")
        final_shape = None
        if shape_data is not None:
            if isinstance(shape_data, (list, tuple)):
                final_shape = torch.Size(shape_data)
            elif isinstance(shape_data, torch.Size):
                final_shape = shape_data
            else:
                try: 
                    final_shape = torch.Size(eval(str(shape_data)))
                except:
                    logger.warning(f"Could not parse shape '{shape_data}' for QuantState, leaving as None.")


        quant_state_obj = cls(
            absmax=cleaned_qs_dict["absmax"].to(target_device),
            shape=final_shape,
            code=cleaned_qs_dict["quant_map"].to(target_device), 
            blocksize=int(cleaned_qs_dict["blocksize"]),
            quant_type=str(cleaned_qs_dict["quant_type"]),
            dtype=getattr(torch, str(cleaned_qs_dict["dtype"])),
            offset=nested_offset_tensor,
            state2=nested_state2,
        )
        return quant_state_obj

    def as_dict(self, packed: bool = False) -> dict[str, Any]:
        qs_dict_data = {
            "quant_type": str(self.quant_type),
            "absmax": self.absmax,
            "blocksize": int(self.blocksize) if self.blocksize is not None else 0,
            "quant_map": self.code, 
            "dtype": str(self.dtype).replace("torch.", ""), 
            "shape": list(self.shape) if self.shape is not None else [], 
        }

        if self.nested and self.state2 is not None and self.offset is not None:
            qs_dict_data.update({
                "nested_absmax": self.state2.absmax,
                "nested_blocksize": int(self.state2.blocksize) if self.state2.blocksize is not None else 0,
                "nested_quant_map": self.state2.code.clone(), 
                "nested_dtype": str(self.state2.dtype).replace("torch.", ""),
                "nested_offset": float(self.offset.item()) if isinstance(self.offset, torch.Tensor) else float(self.offset),
            })

        if not packed:
            return qs_dict_data

        final_packed_dict = {}
        non_tensor_items = {}
        for k, v in qs_dict_data.items():
            if isinstance(v, torch.Tensor):
                final_packed_dict[k] = v
            else:
                non_tensor_items[k] = v
        

        if non_tensor_items:
            serialization_quant_type = self.quant_type if self.quant_type in self.valid_quant_types_static else "other"
            packed_non_tensor_key = f"quant_state.bitsandbytes__{serialization_quant_type}"
            
            final_packed_dict[packed_non_tensor_key] = pack_dict_to_tensor(non_tensor_items)
            
        return final_packed_dict

    def to(self, device: Union[str, torch.device]): 
        self.absmax = self.absmax.to(device)
        if self.code is not None:
            self.code = self.code.to(device)
        if self.nested:
            if self.offset is not None:
                self.offset = self.offset.to(device)
            if self.state2 is not None:
                self.state2.to(device) 
        return self 

    def __eq__(self, other) -> bool: 
        if not isinstance(other, QuantState):
            return False
      
        if not (self.shape == other.shape and \
                self.dtype == other.dtype and \
                self.blocksize == other.blocksize and \
                self.quant_type == other.quant_type):
            return False

        if not torch.allclose(self.absmax, other.absmax, atol=1e-6): return False
        if self.code is not None and other.code is not None:
            if not torch.allclose(self.code, other.code, atol=1e-6): return False
        elif self.code is not other.code: # One is None, the other isn't
            return False
            
        if self.nested != other.nested: return False
        if self.nested: # Both are nested
            if (self.offset is not None and other.offset is not None):
                if isinstance(self.offset, torch.Tensor) and isinstance(other.offset, torch.Tensor):
                    if not torch.allclose(self.offset, other.offset, atol=1e-6): return False
                elif self.offset != other.offset : return False 
            elif self.offset is not other.offset: 
                 return False

            if not (self.state2 == other.state2): return False 
        
        return True

import logging
logger = logging.getLogger(__name__) 

def is_on_gpu(tensors: Iterable[Optional[torch.Tensor]]):
    """
    Verifies that the input tensors are all on the same GPU device.
    A tensor might be None or marked as 'paged', in which case its device placement is handled differently.
    """
    on_gpu_device = True 
    gpu_device_indices = set()
    first_device_type = None

    for t in tensors:
        if t is None:
            continue
        
        if getattr(t, "is_paged", False):
            if first_device_type is None and t.page_deviceid is not None : 
                 first_device_type = "cuda" 
            continue

        if t.device.type not in ["cuda"]: 
            on_gpu_device = False
            break
        
        if first_device_type is None:
            first_device_type = t.device.type
        elif t.device.type != first_device_type:
            on_gpu_device = False 
            break
        
        gpu_device_indices.add(t.device.index)

    if not on_gpu_device:
        tensor_info = []
        for t_info in tensors:
            if t_info is not None:
                tensor_info.append( (str(t_info.shape), str(t_info.device), getattr(t_info, 'is_paged', False)) )
            else:
                tensor_info.append( (None, None, None) )
        raise RuntimeError(
            f"All input tensors intended for GPU operations need to be on a 'cuda' device (NVIDIA or ROCm), "
            f"but found mixed or non-GPU devices: {tensor_info}",
        )

    if len(gpu_device_indices) > 1:
        tensor_info = []
        for t_info in tensors:
            if t_info is not None and not getattr(t_info, 'is_paged', False): 
                tensor_info.append( (str(t_info.shape), str(t_info.device)) )
        raise RuntimeError(
            f"Input tensors need to be on the same GPU, but found combinations on different GPU indices: {tensor_info}",
        )
    return True 


def get_tensor_stream(tensor: Tensor) -> ct.c_void_p: 
    """Gets the raw CUDA/HIP stream of a tensor."""
    if tensor.device.type not in ["cuda"]:
        raise TypeError(f"Tensor must be on a 'cuda' (GPU) device to get its stream, but got {tensor.device.type}")
    return ct.c_void_p(torch._C._cuda_getCurrentRawStream(tensor.device.index))


def get_ptr(A: Optional[Tensor]) -> Optional[ct.c_void_p]:
    """Gets the memory address of the first element of a tensor."""
    if A is None:
        return None
    return ct.c_void_p(A.data_ptr())


@deprecated("This function is deprecated. Use torch.ops.bitsandbytes.estimate_quantiles or specific backend version.", category=FutureWarning)
def estimate_quantiles( 
    A: Tensor,
    out: Optional[torch.Tensor] = None,
    offset: float = 1 / 512,
    num_quantiles: int = 256,
) -> Tensor:
    if A.numel() < 256: # Min elements for meaningful quantile estimation
        raise NotImplementedError(
            f"Quantile estimation needs at least 256 values, but tensor had only {A.numel()} values."
        )
    if num_quantiles > 256:
        raise NotImplementedError(
            f"Currently only a maximum of 256 equally spaced quantiles are supported, got {num_quantiles}."
        )
    
    if num_quantiles < 256 and offset == 1 / 512: 
        offset = 1 / (2 * num_quantiles)

    if out is None:
        out = torch.zeros((256,), dtype=torch.float32, device=A.device)
    else:
        if out.numel() < 256 :
             raise ValueError(f"Output tensor `out` must have at least 256 elements, got {out.numel()}")
        if out.device != A.device:
             out = out.to(A.device)
        if out.dtype != torch.float32:
             out = out.to(torch.float32)


    with _cuda_device_of(A): 
        is_on_gpu([A, out]) 

        if A.dtype == torch.float32:
            lib.cestimate_quantiles_fp32(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
        elif A.dtype == torch.float16:
            lib.cestimate_quantiles_fp16(get_ptr(A), get_ptr(out), ct.c_float(offset), ct.c_int(A.numel()))
        else:
            raise NotImplementedError(f"Quantile estimation not supported for data type {A.dtype}")

    if num_quantiles < 256:
        idx = torch.linspace(0, 255, num_quantiles, device=A.device).long()
        out = out[idx]
    
    return out


def quantize_blockwise(
    A: torch.Tensor,
    code: Optional[torch.Tensor] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 4096, 
    nested: bool = False,
) -> tuple[torch.Tensor, QuantState]:
    """Quantize a tensor in blocks of values (typically 8-bit dynamic quantization)."""
    if code is None:
        if "dynamic" not in name2qmap: # Cache the map
            name2qmap["dynamic"] = create_dynamic_map().to(A.device) 
        code = name2qmap["dynamic"]
    else: # Ensure provided code is on the correct device
        code = code.to(A.device)

    n = A.numel()
    num_blocks = (n + blocksize - 1) // blocksize 

    if absmax is None:
        absmax = torch.empty((num_blocks,), device=A.device, dtype=torch.float32)
    elif absmax.numel() != num_blocks:
        raise ValueError(f"Provided absmax tensor has incorrect size. Expected {num_blocks}, got {absmax.numel()}")

    if out is None:
        out = torch.empty_like(A, dtype=torch.uint8) 
    elif out.dtype != torch.uint8:
        raise ValueError(f"Output tensor `out` must be of dtype torch.uint8, got {out.dtype}")

    supported_blocksizes_cuda = [4096, 2048, 1024, 512, 256, 128, 64]
    supported_blocksizes_hip = [4096, 2048, 1024, 512, 256, 128] # 64 might be less optimal or not supported on some HIP
    
    current_device_type = A.device.type
    if current_device_type == "cuda": # Covers both NVIDIA and AMD ROCm via PyTorch device type
        is_on_gpu([A, code, absmax, out]) # Ensure all tensors are on GPU
        with _cuda_device_of(A): # Set device context
            c_blocksize_arg = ct.c_int32(blocksize) # For GPU kernels
            
            if HIP_ENVIRONMENT:
                if blocksize not in supported_blocksizes_hip:
                    logger.warning(
                        f"Blocksize {blocksize} for quantize_blockwise on ROCm is not in the typical supported list: {supported_blocksizes_hip}. "
                        "Kernel might not be optimized or available."
                    )
            else: # CUDA
                if blocksize not in supported_blocksizes_cuda:
                     logger.warning(
                        f"Blocksize {blocksize} for quantize_blockwise on CUDA is not in the typical supported list: {supported_blocksizes_cuda}. "
                        "Kernel might not be optimized or available."
                    )

            if A.dtype == torch.float32:
                lib.cquantize_blockwise_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), c_blocksize_arg, ct.c_int(n))
            elif A.dtype == torch.float16:
                lib.cquantize_blockwise_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), c_blocksize_arg, ct.c_int(n))
            elif A.dtype == torch.bfloat16:
                lib.cquantize_blockwise_bf16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), c_blocksize_arg, ct.c_int(n))
            else:
                raise ValueError(f"Blockwise quantization only supports float16/bfloat16/float32, but got {A.dtype}")

    elif current_device_type == "cpu":
        if A.dtype != torch.float32: # CPU kernel in main PDF was specific to fp32 input
            raise ValueError(f"CPU quantize_blockwise currently only supports float32 input, got {A.dtype}")
        if code.device.type != "cpu": code = code.cpu()
        if absmax.device.type != "cpu": absmax = absmax.cpu()
        if out.device.type != "cpu": out = out.cpu()
        lib.cquantize_blockwise_cpu_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_longlong(blocksize), ct.c_longlong(n))
    else:
        raise NotImplementedError(f"quantize_blockwise not implemented for device type {current_device_type}")

    if nested:

        offset = absmax.mean() 
        absmax_to_quantize = absmax - offset
        qabsmax, state2 = quantize_blockwise(absmax_to_quantize, code=code, blocksize=blocksize, nested=False) 
        
        quant_state = QuantState(
            absmax=qabsmax, code=code, blocksize=blocksize, dtype=A.dtype, 
            offset=offset, state2=state2, shape=A.shape, quant_type="dynamic_nested"
        )
    else:
        quant_state = QuantState(
            absmax=absmax, code=code, blocksize=blocksize, dtype=A.dtype, 
            shape=A.shape, quant_type="dynamic"
        )
    
    return out, quant_state


def dequantize_blockwise( 
    A: torch.Tensor, 
    quant_state: Optional[QuantState] = None,

    absmax: Optional[torch.Tensor] = None, 
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None, 
) -> torch.Tensor:
    """Dequantize a tensor that was quantized by quantize_blockwise."""
    if quant_state is None:
        if absmax is None or code is None or blocksize is None:
            raise ValueError("If quant_state is None, then absmax, code, and blocksize must be provided.")
        _dtype = torch.float32
        _shape = A.shape # Output shape will be same as quantized if not further info

        quant_state = QuantState(absmax=absmax, code=code, blocksize=blocksize, dtype=_dtype, shape=_shape, quant_type="dynamic_legacy")
    
    current_absmax = quant_state.absmax
    current_code = quant_state.code
    current_blocksize = quant_state.blocksize
    target_dtype = quant_state.dtype
    target_shape = quant_state.shape if quant_state.shape is not None else A.shape

    if quant_state.nested:
        dequantized_absmax = dequantize_blockwise(current_absmax, quant_state.state2)
        current_absmax = dequantized_absmax + quant_state.offset # Add back the offset
    
    if current_absmax.dtype != torch.float32: # Ensure absmax is float32 for dequant kernel
        current_absmax = current_absmax.float()
    if current_code.device != A.device: # Ensure code is on same device as data
        current_code = current_code.to(A.device)

    if out is None:
        out = torch.empty(target_shape, dtype=target_dtype, device=A.device)
    elif out.shape != target_shape or out.dtype != target_dtype:
        raise ValueError(
            f"Output tensor `out` has incorrect shape or dtype. Expected shape {target_shape}, dtype {target_dtype}, "
            f"but got shape {out.shape}, dtype {out.dtype}."
        )

    n = out.numel() # Number of elements in the output tensor

    current_device_type = A.device.type
    if current_device_type == "cuda": 
        is_on_gpu([A, current_absmax, current_code, out])
        with _cuda_device_of(A):
            c_blocksize_arg = ct.c_int(current_blocksize) 

            supported_blocksizes_cuda = [4096, 2048, 1024, 512, 256, 128, 64]
            supported_blocksizes_hip = [4096, 2048, 1024, 512, 256, 128]
            if HIP_ENVIRONMENT:
                if current_blocksize not in supported_blocksizes_hip:
                    logger.warning(f"Blocksize {current_blocksize} for dequantize_blockwise on ROCm not in typical list.")
            else:
                if current_blocksize not in supported_blocksizes_cuda:
                    logger.warning(f"Blocksize {current_blocksize} for dequantize_blockwise on CUDA not in typical list.")
            
            if target_dtype == torch.float32:
                lib.cdequantize_blockwise_fp32(get_ptr(current_code), get_ptr(A), get_ptr(current_absmax), get_ptr(out), c_blocksize_arg, ct.c_int(n))
            elif target_dtype == torch.float16:
                lib.cdequantize_blockwise_fp16(get_ptr(current_code), get_ptr(A), get_ptr(current_absmax), get_ptr(out), c_blocksize_arg, ct.c_int(n))
            elif target_dtype == torch.bfloat16:
                lib.cdequantize_blockwise_bf16(get_ptr(current_code), get_ptr(A), get_ptr(current_absmax), get_ptr(out), c_blocksize_arg, ct.c_int(n))
            else:
                raise ValueError(f"Blockwise dequantization only supports float16/bfloat16/float32 output, but got {target_dtype}")
    
    elif current_device_type == "cpu":
        if target_dtype != torch.float32: 
            raise ValueError(f"CPU dequantize_blockwise currently only supports float32 output, got {target_dtype}")
        if current_code.device.type != "cpu": current_code = current_code.cpu()
        if current_absmax.device.type != "cpu": current_absmax = current_absmax.cpu()
        if A.device.type != "cpu": A_cpu = A.cpu() 
        else: A_cpu = A
        if out.device.type != "cpu": 
            pass

        lib.cdequantize_blockwise_cpu_fp32(get_ptr(current_code), get_ptr(A_cpu), get_ptr(current_absmax), get_ptr(out), ct.c_longlong(current_blocksize), ct.c_longlong(n))
    else:
        raise NotImplementedError(f"dequantize_blockwise not implemented for device type {current_device_type}")
        
    return out

def get_4bit_type(typename: str, device: Optional[Union[str, torch.device]] = None, blocksize: int = 64) -> torch.Tensor: # As in Main PDF (page 41)
    """
    Returns the quantization map for a given 4-bit data type (nf4, fp4, etc.).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    data = None
    if typename == "nf4":
        data = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ]
    elif typename == "fp4": # Values from Main PDF
        data = [
            0.0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, 
            -0.0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0 # Ensure negative zero if distinct, or just 0.
        ]
    elif typename == "int4": # From Main PDF
        data = [7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7] # Symmetric int4
        data = [float(x) for x in data] # Convert to float for tensor
    elif typename == "af4": # Abnormal Float 4-bit (Main PDF)
        if blocksize == 64: # AF4 in main PDF was blocksize specific
            data = [ # Values are reversed in main PDF, check intended order
                -1.0, -0.69441008, -0.51243739, -0.3736951,
                -0.25607552, -0.14982478, -0.04934812, 0.0,
                0.04273164, 0.12934483, 0.21961274, 0.31675666,
                0.42563882, 0.55496234, 0.72424863, 1.0,
            ] 
        else:
            raise NotImplementedError(f"4-bit Abnormal Floats (af4) currently only support blocksize 64, got {blocksize}.")
    
    if data is None:
        raise NotImplementedError(f"4-bit type '{typename}' not supported.")

    map_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    max_abs = map_tensor.abs().max()
    if max_abs > 0:
        map_tensor.div_(max_abs)
    
    if map_tensor.numel() != 16:
        raise ValueError(f"Generated 4-bit map for '{typename}' has {map_tensor.numel()} elements, expected 16.")
        
    return map_tensor


def quantize_4bit( 
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None, 
    out: Optional[torch.Tensor] = None,    
    blocksize: Optional[int] = None,
    compress_statistics: bool = False, 
    quant_type: str = "fp4", # "nf4" or "fp4"
    quant_storage: torch.dtype = torch.uint8, 
) -> tuple[torch.Tensor, QuantState]:
    """Quantize tensor A to 4-bit using blockwise quantization."""
    
    if blocksize is None: 
        blocksize = 64 if not HIP_ENVIRONMENT else 128

    input_shape = A.shape
    n = A.numel()
    num_blocks = (n + blocksize - 1) // blocksize

    current_device_type = A.device.type
    if current_device_type == "cuda": 
        is_on_gpu([A]) 
        

        supported_blocksizes_cuda = [4096, 2048, 1024, 512, 256, 128, 64]
        supported_blocksizes_hip = [4096, 2048, 1024, 512, 256, 128] # Example for HIP

        if HIP_ENVIRONMENT:
            if blocksize not in supported_blocksizes_hip:
                logger.warning(f"Blocksize {blocksize} for quantize_4bit on ROCm not in typical list: {supported_blocksizes_hip}")
        else: # CUDA
            if blocksize not in supported_blocksizes_cuda:
                logger.warning(f"Blocksize {blocksize} for quantize_4bit on CUDA not in typical list: {supported_blocksizes_cuda}")

        num_output_elements_packed = (n + 1) // (quant_storage.itemsize * 8 // 4) # General for any itemsize

        _out_q = torch.empty((num_output_elements_packed, 1), dtype=quant_storage, device=A.device)
        _absmax_q = torch.empty((num_blocks,), device=A.device, dtype=torch.float32)

        with _cuda_device_of(A):
            c_args = (
                None, 
                get_ptr(A),
                get_ptr(_absmax_q),
                get_ptr(_out_q),
                ct.c_int32(blocksize),
                ct.c_int(n), 
            )
            if A.dtype == torch.bfloat16:
                if quant_type == "fp4": lib.cquantize_blockwise_bf16_fp4(*c_args)
                else: lib.cquantize_blockwise_bf16_nf4(*c_args) # nf4
            elif A.dtype == torch.float16:
                if quant_type == "fp4": lib.cquantize_blockwise_fp16_fp4(*c_args)
                else: lib.cquantize_blockwise_fp16_nf4(*c_args) # nf4
            elif A.dtype == torch.float32:
                if quant_type == "fp4": lib.cquantize_blockwise_fp32_fp4(*c_args)
                else: lib.cquantize_blockwise_fp32_nf4(*c_args) # nf4
            else:
                raise ValueError(f"4-bit quantization only supports bfloat16/float16/float32, got {A.dtype}")
    
    elif current_device_type == "cpu":
        if quant_type != "nf4": # Main PDF CPU ops only showed NF4 for 4-bit
            raise NotImplementedError(f"CPU 4-bit quantization currently only supports nf4, got {quant_type}")
        if A.dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError(f"CPU 4-bit quantization only supports float16/32, got {A.dtype}")
        if n % blocksize != 0: # CPU check from main PDF
             raise ValueError(f"CPU 4-bit quantization: n ({n}) must be divisible by blocksize ({blocksize})")

        blocks_reshaped = A.reshape(-1, blocksize)
        _absmax_q = blocks_reshaped.abs().max(dim=1).values.float() 
        scaled = blocks_reshaped / _absmax_q.unsqueeze(-1)
        
  
        _NF4_QUANT_TABLE_CPU = get_4bit_type("nf4", device='cpu')

        quantized_indices = torch.argmin(torch.abs(scaled.reshape(-1, 1) - _NF4_QUANT_TABLE_CPU), dim=-1).to(torch.uint8)

        if quantized_indices.numel() % 2 != 0: # Pad if odd number of elements
            quantized_indices = torch.cat([quantized_indices, torch.zeros(1, dtype=torch.uint8, device='cpu')])

        _out_q_packed_list = (quantized_indices[::2] << 4) | quantized_indices[1::2]
        _out_q = _out_q_packed_list.unsqueeze(1) 

        if quant_storage != torch.uint8: 
             _out_q = _out_q.squeeze().view(quant_storage).unsqueeze(1)

    else:
        raise NotImplementedError(f"quantize_4bit not implemented for device type {current_device_type}")


    if out is not None: out.copy_(_out_q)
    else: out = _out_q
    if absmax is not None: absmax.copy_(_absmax_q)
    code_map = get_4bit_type(quant_type, device=A.device) # Get the 4-bit map for the state
    
    final_absmax_for_state = _absmax_q
    final_offset_for_state = None
    final_state2_for_state = None

    if compress_statistics: # Nested quantization for absmax
        offset_val = _absmax_q.mean()
        absmax_to_quantize_nested = _absmax_q - offset_val
        qabsmax_nested, state2_nested = quantize_blockwise(absmax_to_quantize_nested, blocksize=256, nested=False)
        
        final_absmax_for_state = qabsmax_nested
        final_offset_for_state = offset_val
        final_state2_for_state = state2_nested
        
    quant_state_obj = QuantState(
        absmax=final_absmax_for_state,
        shape=input_shape,
        dtype=A.dtype,
        blocksize=blocksize,
        code=code_map, # The 4-bit map (e.g., NF4 values)
        quant_type=quant_type,
        offset=final_offset_for_state,
        state2=final_state2_for_state

    )
    
    return out, quant_state_obj


def dequantize_4bit( 
    A: torch.Tensor, 
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None, 
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None, 
    quant_type: Optional[str] = None, 
) -> torch.Tensor:
    """Dequantizes a 4-bit quantized tensor."""

    if quant_state is None:

        if absmax is None or blocksize is None or quant_type is None or out is None: 
            raise ValueError("If quant_state is None, then absmax, blocksize, quant_type, and a template `out` tensor (for shape/dtype) must be provided.")
        _quant_state_temp = QuantState(
            absmax=absmax, shape=out.shape, dtype=out.dtype, 
            blocksize=blocksize, quant_type=quant_type,
            code=get_4bit_type(quant_type, device=A.device) # Generate code map
        )

        current_quant_state = _quant_state_temp
    else:
        current_quant_state = quant_state


    current_absmax = current_quant_state.absmax
    current_blocksize = current_quant_state.blocksize
    current_quant_type = current_quant_state.quant_type
    target_shape = current_quant_state.shape
    target_dtype = current_quant_state.dtype
   

    if current_quant_state.nested:

        dequantized_absmax_stats = dequantize_blockwise(current_absmax, current_quant_state.state2) # Uses 8-bit dequant
        current_absmax = dequantized_absmax_stats + current_quant_state.offset
    
    if current_absmax.dtype != torch.float32: 
        current_absmax = current_absmax.float()


    if out is None:
        out_tensor = torch.empty(target_shape, dtype=target_dtype, device=A.device)
    else: 
        if out.shape != target_shape or out.dtype != target_dtype or out.device != A.device:
            raise ValueError(f"Provided `out` tensor is incompatible. Expected shape {target_shape}, dtype {target_dtype}, device {A.device}. "
                             f"Got shape {out.shape}, dtype {out.dtype}, device {out.device}.")
        out_tensor = out
        
    num_output_elements = out_tensor.numel()

    current_device_type = A.device.type
    if current_device_type == "cuda": 
        is_on_gpu([A, current_absmax, out_tensor])
        with _cuda_device_of(A):
            c_blocksize_arg = ct.c_int(current_blocksize)
            stream_ptr = get_tensor_stream(A) 

            supported_blocksizes_cuda = [4096, 2048, 1024, 512, 256, 128, 64]
            supported_blocksizes_hip = [4096, 2048, 1024, 512, 256, 128]
            if HIP_ENVIRONMENT:
                if current_blocksize not in supported_blocksizes_hip:
                    logger.warning(f"Blocksize {current_blocksize} for dequantize_4bit on ROCm not in typical list.")
            else: 
                if current_blocksize not in supported_blocksizes_cuda:
                    logger.warning(f"Blocksize {current_blocksize} for dequantize_4bit on CUDA not in typical list.")
            
            c_args = (
                None, 
                get_ptr(A), 
                get_ptr(current_absmax), 
                get_ptr(out_tensor),   
                c_blocksize_arg,
                ct.c_int(num_output_elements), 
                stream_ptr 
            )


            if target_dtype == torch.bfloat16:
                if current_quant_type == "fp4": lib.cdequantize_blockwise_bf16_fp4(*c_args)
                else: lib.cdequantize_blockwise_bf16_nf4(*c_args) # nf4
            elif target_dtype == torch.float16:
                if current_quant_type == "fp4": lib.cdequantize_blockwise_fp16_fp4(*c_args)
                else: lib.cdequantize_blockwise_fp16_nf4(*c_args) # nf4
            elif target_dtype == torch.float32:
                if current_quant_type == "fp4": lib.cdequantize_blockwise_fp32_fp4(*c_args)
                else: lib.cdequantize_blockwise_fp32_nf4(*c_args) # nf4
            else:
                raise ValueError(f"4-bit dequantization only supports bfloat16/float16/float32 output, got {target_dtype}")

    elif current_device_type == "cpu":

        if current_quant_type != "nf4":
            raise NotImplementedError(f"CPU 4-bit dequantization currently only supports nf4, got {current_quant_type}")
        if target_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError(f"CPU 4-bit dequantization only supports float16/32 output, got {target_dtype}")
        if A.dtype != torch.uint8: 
            raise ValueError(f"CPU 4-bit dequantization expects uint8 input, got {A.dtype}")

        A_cpu = A.cpu() if A.device.type != "cpu" else A
        current_absmax_cpu = current_absmax.cpu() if current_absmax.device.type != "cpu" else current_absmax
        if out_tensor.device.type != "cpu": out_tensor = out_tensor.to("cpu") 

        A_flat = A_cpu.view(-1)
        upper_nibbles = (A_flat >> 4).to(torch.int64) # Higher 4 bits
        lower_nibbles = (A_flat & 0x0F).to(torch.int64) # Lower 4 bits
        
        indices = torch.empty(num_output_elements, dtype=torch.int64, device='cpu')
        num_packed_bytes = A_flat.numel()
        indices[:num_packed_bytes*2:2] = upper_nibbles[:num_packed_bytes]
        if num_output_elements % 2 == 0 : 
             indices[1:num_packed_bytes*2:2] = lower_nibbles[:num_packed_bytes]
        else: 
             if num_packed_bytes > 0: 
                indices[1:(num_packed_bytes-1)*2+1:2] = lower_nibbles[:num_packed_bytes-1]          
                unpacked_values = []
                for byte_val in A_flat:
                    unpacked_values.append( (byte_val.item() >> 4) & 0x0F )
                    unpacked_values.append( byte_val.item() & 0x0F )     
                indices = torch.tensor(unpacked_values[:num_output_elements], dtype=torch.int64, device='cpu')

        _NF4_QUANT_TABLE_CPU = get_4bit_type("nf4", device='cpu')
        dequantized_blocks_scaled = _NF4_QUANT_TABLE_CPU[indices]
        dequantized_blocks_scaled = dequantized_blocks_scaled.reshape(-1, current_blocksize)
        dequantized_final = dequantized_blocks_scaled * current_absmax_cpu.unsqueeze(-1)
        

        out_tensor.copy_(dequantized_final.reshape(target_shape).to(target_dtype))

    else:
        raise NotImplementedError(f"dequantize_4bit not implemented for device type {current_device_type}")
        
    return out_tensor

def quantize_fp4(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None, # Made optional to use env-aware default
    compress_statistics: bool = False,
    quant_storage: torch.dtype = torch.uint8,
):
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "fp4", quant_storage)

def quantize_nf4(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None, # Made optional
    compress_statistics: bool = False,
    quant_storage: torch.dtype = torch.uint8,
):
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "nf4", quant_storage)

def dequantize_fp4(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None, 
):
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "fp4")

def dequantize_nf4(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None, 
):
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "nf4")


@deprecated("This function is deprecated. Use blockwise quantization (quantize_blockwise) or 4-bit quantization.", category=FutureWarning)
def quantize(
    A: Tensor,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[Tensor, tuple[Tensor, Tensor]]: 
    if code is None:
        if "dynamic" not in name2qmap: 
            name2qmap["dynamic"] = create_dynamic_map().to(A.device)
        code = name2qmap["dynamic"]
    else:
        code = code.to(A.device)

    current_absmax = torch.abs(A).max()
    if current_absmax.dtype != torch.float32:
        current_absmax = current_absmax.float()
    
    if current_absmax == 0: 
        inp_scaled = torch.zeros_like(A)
    else:
        inp_scaled = A / current_absmax
    

    quantized_out = quantize_no_absmax(inp_scaled, code, out)
    
    return quantized_out, (current_absmax, code)


@deprecated("This function is deprecated. Use blockwise dequantization (dequantize_blockwise) or 4-bit dequantization.", category=FutureWarning)
def dequantize(
    A: Tensor, 
    state: Optional[tuple[Tensor, Tensor]] = None, 
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tensor:
    current_absmax, current_code = None, None
    if state is not None:
        current_absmax, current_code = state
    elif absmax is not None and code is not None:
        current_absmax, current_code = absmax, code
    else:
        raise ValueError("Either 'state' or both 'absmax' and 'code' must be provided for dequantization.")

    if current_code is None: 
         raise ValueError("Quantization map 'code' is missing.")
    current_code = current_code.to(A.device) 


    dequantized_scaled_out = dequantize_no_absmax(A, current_code, out)
    

    if current_absmax is not None:
        return dequantized_scaled_out * current_absmax
    else: # Should not happen if absmax was required
        return dequantized_scaled_out


@deprecated("This function is deprecated.", category=FutureWarning)
def quantize_no_absmax(A: Tensor, code: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    """Quantizes input tensor (assumed to be scaled to [-1, 1]) to 8-bit using 'code' map."""
    current_device_type = A.device.type
    if out is None:
        out = torch.empty_like(A, dtype=torch.uint8) 
    
    if current_device_type == "cuda":
        is_on_gpu([A, code, out])
        with _cuda_device_of(A):
            # lib.cquantize is the C function for this operation
            lib.cquantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()))
    elif current_device_type == "cpu":
        logger.warning("quantize_no_absmax CPU path might be slow or not fully implemented without a dedicated C kernel.")
        A_flat = A.flatten()
        out_flat = out.flatten()
        for i in range(A_flat.shape[0]):
            val = A_flat[i]
            insertion_point = torch.searchsorted(code, val.to(code.device)) 
            idx = insertion_point.clamp(0, code.shape[0] - 1)
            if idx > 0 and (val - code[idx-1]).abs() < (val - code[idx]).abs():
                idx = idx -1
            out_flat[i] = idx.to(torch.uint8)

    else:
        raise NotImplementedError(f"quantize_no_absmax not implemented for device type {current_device_type}")
    return out


@deprecated("This function is deprecated.", category=FutureWarning)
def dequantize_no_absmax(A: Tensor, code: Tensor, out: Optional[torch.Tensor] = None) -> Tensor:
    """Dequantizes 8-bit tensor to 32-bit float using 'code' map."""
    current_device_type = A.device.type
    if out is None:
        out = torch.empty_like(A, dtype=torch.float32) # Output is float32

    if current_device_type == "cuda": 
        is_on_gpu([A, code, out])
        with _cuda_device_of(A):
            stream = get_tensor_stream(A) 
            # lib.cdequantize is the C function
            lib.cdequantize(get_ptr(code), get_ptr(A), get_ptr(out), ct.c_int(A.numel()), stream)
    elif current_device_type == "cpu":
 
        A_cpu = A.cpu() if A.device.type != "cpu" else A
        code_cpu = code.cpu() if code.device.type != "cpu" else code
        if out.device.type != "cpu": out = out.to("cpu")
        
        out.copy_(code_cpu[A_cpu.long()])
    else:
        raise NotImplementedError(f"dequantize_no_absmax not implemented for device type {current_device_type}")
    return out

from bitsandbytes.backends import backends, ensure_backend_is_available


def optimizer_update_32bit( 
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor,
    beta1: float,
    eps: float,
    step: int,
    lr: float, 
    state2: Optional[torch.Tensor] = None, 
    beta2: float = 0.0,
    beta3: float = 0.0, 
    alpha: float = 0.0, 
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Optional[torch.Tensor] = None, 
    max_unorm: float = 0.0, 
    skip_zeros: bool = False,
) -> None:
    """
    Performs an inplace 32-bit optimizer update.
    Dispatches to the appropriate backend (CPU, CUDA, ROCm).
    """
    ensure_backend_is_available(g.device.type)
    backends[g.device.type.lower()].optimizer_update_32bit(
        optimizer_name=optimizer_name,
        g=g,
        p=p,
        state1=state1,
        beta1=beta1,
        eps=eps,
        step=step,
        lr=lr, 
        state2=state2,
        beta2=beta2,
        weight_decay=weight_decay,
        gnorm_scale=gnorm_scale,
        unorm_vec=unorm_vec,
        max_unorm=max_unorm,
        skip_zeros=skip_zeros,
    )

@deprecated(
    "This function is deprecated. Use optimizer_update_8bit_blockwise instead.",
    category=FutureWarning,
)
def optimizer_update_8bit( 
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor, 
    state2: Optional[torch.Tensor], 
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    lr: float, 
    qmap1: Tensor, 
    qmap2: Optional[torch.Tensor], 
    max1: Tensor, 
    max2: Optional[torch.Tensor], 
    new_max1: Tensor, 
    new_max2: Optional[torch.Tensor], 
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Optional[torch.Tensor] = None,
    max_unorm: float = 0.0,
) -> None:
    """Deprecated static 8-bit optimizer update. GPU only."""
    if g.device.type != 'cuda':
        raise NotImplementedError("Static 8-bit optimizer is only implemented for CUDA/ROCm devices.")

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    with _cuda_device_of(g):
        is_on_gpu([g, p, state1, state2, unorm_vec, qmap1, qmap2, max1, max2, new_max1, new_max2]) # Defined in Part 2
        
        optim_func_idx = 0 # Default to float32 gradient version
        if g.dtype == torch.float16: optim_func_idx = 1

        if optimizer_name not in str2optimizer8bit or \
           len(str2optimizer8bit[optimizer_name]) <= optim_func_idx:
            raise ValueError(f"Static 8-bit optimizer {optimizer_name} not found or no kernel for grad dtype {g.dtype}")

        c_optim_func = str2optimizer8bit[optimizer_name][optim_func_idx]
        
        c_optim_func(
            get_ptr(p), get_ptr(g), get_ptr(state1), get_ptr(state2),
            get_ptr(unorm_vec), ct.c_float(max_unorm), ct.c_float(param_norm),
            ct.c_float(beta1), ct.c_float(beta2), ct.c_float(eps), ct.c_int32(step),
            ct.c_float(lr), get_ptr(qmap1), get_ptr(qmap2),
            get_ptr(max1), get_ptr(max2), get_ptr(new_max1), get_ptr(new_max2),
            ct.c_float(weight_decay), ct.c_float(gnorm_scale), ct.c_int32(g.numel()),
        )


def optimizer_update_8bit_blockwise( 
    optimizer_name: str,
    g: Tensor,
    p: Tensor,
    state1: Tensor, 
    state2: Optional[torch.Tensor], 
    beta1: float,
    beta2: float,
    beta3: float, 
    alpha: float, 
    eps: float,
    step: int,
    lr: float,
    qmap1: Tensor, 
    qmap2: Optional[torch.Tensor], 
    absmax1: Tensor, # Blockwise absmax for state1
    absmax2: Optional[torch.Tensor], 
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    skip_zeros: bool = False,
) -> None:
    """
    Performs an inplace 8-bit blockwise optimizer update.
    Dispatches to the appropriate backend.
    """
    ensure_backend_is_available(g.device.type)
    backends[g.device.type.lower()].optimizer_update_8bit_blockwise(
        optimizer_name=optimizer_name,
        g=g,
        p=p,
        state1=state1,
        state2=state2,
        beta1=beta1,
        beta2=beta2,
        # beta3 and alpha are not passed to backend method currently.
        # If needed, backend method signature would need update, or specific handling here.
        eps=eps,
        step=step,
        lr=lr,
        qmap1=qmap1,
        qmap2=qmap2,
        absmax1=absmax1,
        absmax2=absmax2,
        weight_decay=weight_decay,
        gnorm_scale=gnorm_scale,
        skip_zeros=skip_zeros,
    )


@deprecated("This function is deprecated and GPU-specific.", category=FutureWarning)
def percentile_clipping(grad: Tensor, gnorm_vec: Tensor, step: int, percentile: int = 5): 
    """Applies percentile clipping. GPU only."""
    if grad.device.type != 'cuda':
        raise NotImplementedError("percentile_clipping is only implemented for CUDA/ROCm devices.")
    
    with _cuda_device_of(grad):
        is_on_gpu([grad, gnorm_vec])
        c_func = None
        if grad.dtype == torch.float32:
            c_func = lib.cpercentile_clipping_g32
        elif grad.dtype == torch.float16:
            c_func = lib.cpercentile_clipping_g16
        else:
            raise ValueError(f"Gradient type {grad.dtype} not supported for percentile_clipping!")
        
        c_func(
            get_ptr(grad), get_ptr(gnorm_vec),
            ct.c_int32(step), ct.c_int32(grad.numel())
        )
    
    current_gnorm = torch.sqrt(gnorm_vec[step % 100]) 
    vals_sorted, _ = torch.sort(gnorm_vec) 
    clip_value_sq = vals_sorted[percentile] 
    clip_value = torch.sqrt(clip_value_sq) 
    
    gnorm_scale = 1.0
    if current_gnorm > clip_value:
        gnorm_scale = clip_value / current_gnorm
        
    return current_gnorm, clip_value, gnorm_scale


@deprecated("This function is deprecated and GPU-specific.", category=FutureWarning)
def histogram_scatter_add_2d(histogram: Tensor, index1: Tensor, index2: Tensor, source: Tensor):
    """Performs a 2D histogram scatter add. GPU only."""
    if histogram.device.type != 'cuda':
        raise NotImplementedError("histogram_scatter_add_2d is only implemented for CUDA/ROCm devices.")

  
    assert len(histogram.shape) == 2
    assert histogram.dtype == torch.float32
    assert source.dtype == torch.float32
    assert index1.dtype == torch.int32
    assert index2.dtype == torch.int32
    
    is_on_gpu([histogram, index1, index2, source])
    
    maxdim1 = ct.c_int32(histogram.shape[0])
    n_elements = ct.c_int32(index1.numel()) 
    
    with _cuda_device_of(histogram):
        lib.chistogram_scatter_add_2d(
            get_ptr(histogram), get_ptr(index1), get_ptr(index2), get_ptr(source),
            maxdim1, n_elements
        )

def check_matmul(A, B, out, transposed_A, transposed_B, expected_type=torch.int8):
    """Utility to check matrix multiplication dimensions."""

    if torch.cuda.is_available() and not torch.cuda.is_initialized(): 
        torch.cuda.init()

    if A.dtype != expected_type or B.dtype != expected_type:
        raise TypeError(f"Expected {expected_type} input tensors A and B, but got {A.dtype} and {B.dtype}")

    sA = A.shape
    sB = B.shape
    correct = True

    if len(sA) == 2 and len(sB) == 2:
        if not transposed_A and not transposed_B and A.shape[1] != B.shape[0]: correct = False
        elif transposed_A and not transposed_B and A.shape[0] != B.shape[0]: correct = False
        elif transposed_A and transposed_B and A.shape[0] != B.shape[1]: correct = False
        elif not transposed_A and transposed_B and A.shape[1] != B.shape[1]: correct = False
    elif len(sA) == 3 and len(sB) == 2:
        if not transposed_A and not transposed_B and A.shape[2] != B.shape[0]: correct = False
        elif transposed_A and not transposed_B and A.shape[1] != B.shape[0]: correct = False
        elif transposed_A and transposed_B and A.shape[1] != B.shape[1]: correct = False
        elif not transposed_A and transposed_B and A.shape[2] != B.shape[1]: correct = False
    elif len(sA) == 3 and len(sB) == 3: 
        if sA[0] != sB[0]: correct = False 
        if not transposed_A and not transposed_B and A.shape[2] != B.shape[1]: correct = False
        elif transposed_A and not transposed_B and A.shape[1] != B.shape[1]: correct = False
        elif transposed_A and transposed_B and A.shape[1] != B.shape[2]: correct = False
        elif not transposed_A and transposed_B and A.shape[2] != B.shape[2]: correct = False
    else: 
        correct = False

  
    sout_expected = None
    if correct:
        if len(sA) == 2 and len(sB) == 2:
            sout_expected = (sA[0] if not transposed_A else sA[1], sB[1] if not transposed_B else sB[0])
        elif len(sA) == 3 and len(sB) == 2: 
            sout_expected = (sA[0], sA[1] if not transposed_A else sA[2], sB[1] if not transposed_B else sB[0])
        elif len(sA) == 3 and len(sB) == 3: 
            sout_expected = (sA[0], sA[1] if not transposed_A else sA[2], sB[2] if not transposed_B else sB[1])

    if out is not None:
        sout_actual = out.shape
        if not correct and len(sA) == 3 and len(sB) == 3:
            if sout_actual[0] == sA[2] and sout_actual[1] == sB[2] and sA[0] == sB[0] and sA[1] == sB[1]:
                correct = True 
                sout_expected = sout_actual 
        elif sout_expected is not None and sout_actual != sout_expected:
            correct = False 
    elif not correct and sout_expected is None : 
         pass 

    if not correct:
        raise ValueError(
            f"Tensor dimensions incorrect for matrix multiplication: A x B: {sA} x {sB} "
            f"with transpose_A={transposed_A}, transpose_B={transposed_B}. "
            f"Expected output shape: {sout_expected if sout_expected else ' undetermined'}, "
            f"Actual output shape (if out provided): {out.shape if out is not None else 'N/A'}."
        )
    return sout_expected if out is None else out.shape 

def gemv_4bit( 
    A: Tensor, 
    B: Tensor, 
    out: Optional[torch.Tensor] = None, 
    transposed_A: bool = False, 
    transposed_B: bool = False, 
                               
    state: Optional[QuantState] = None, 
):
    """Performs 4-bit GEMV: out = A @ B_dequantized."""
    if state is None:
        raise ValueError("QuantState 'state' for the 4-bit matrix B cannot be None for gemv_4bit.")
    
    ensure_backend_is_available(A.device.type)
    return backends[A.device.type.lower()].gemv_4bit(
        A=A, B=B, out=out,
        transposed_A=transposed_A, 
        transposed_B=transposed_B, 
        state=state
    )


def igemm( # Integer GEMM from Main PDF (page 55)
    A: Tensor, 
    B: Tensor, 
    out: Optional[torch.Tensor] = None, # Output int32
    transposed_A: bool = False,
    transposed_B: bool = False,
):
    """Performs int8 matrix multiplication: C = A @ B (with optional transposes)."""
    if A.device.type != 'cuda':
        raise NotImplementedError("igemm is currently only implemented for CUDA/ROCm devices via C extensions.")

    sout_shape = check_matmul(A, B, out, transposed_A, transposed_B, expected_type=torch.int8)
    if out is None:
        out = torch.empty(sout_shape, dtype=torch.int32, device=A.device) # Note: torch.empty, not zeros
    elif out.dtype != torch.int32:
        raise ValueError(f"Output tensor for igemm must be int32, got {out.dtype}")

    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] == B.shape[0] and \
           (A.shape[2] if not transposed_A else A.shape[1]) == \
           (B.shape[1] if not transposed_B else B.shape[2]): # k_A == k_B for batch
            return batched_igemm(A, B, out, transposed_A, transposed_B) # Call batched version

    sA_effective = A.shape
    sB_effective = B.shape
    
    
    m_dim = A.shape[0] if not transposed_A else A.shape[1]
    k_dim_A = A.shape[1] if not transposed_A else A.shape[0]
    
    k_dim_B = B.shape[0] if not transposed_B else B.shape[1]
    n_dim = B.shape[1] if not transposed_B else B.shape[0]

    if k_dim_A != k_dim_B:
        raise ValueError(f"Inner dimensions for igemm mismatch: A_k={k_dim_A}, B_k={k_dim_B} "
                         f"(A shape {A.shape}, tA={transposed_A}; B shape {B.shape}, tB={transposed_B})")
    k_dim = k_dim_A


    ptr = CUBLAS_Context.get_instance().get_context(A.device)
    is_on_gpu([A, B, out]) # Validation


    logger.warning("functional.igemm direct C call is complex and error-prone; prefer torch.ops.bitsandbytes.int8_linear_matmul.")
      
    rows_B_op = B.shape[0] if not transposed_B else B.shape[1]
    cols_B_op = B.shape[1] if not transposed_B else B.shape[0] # This is K_eff

    rows_A_op = A.shape[0] if not transposed_A else A.shape[1] # This must match K_eff
    cols_A_op = A.shape[1] if not transposed_A else A.shape[0]

    if cols_B_op != rows_A_op:
        raise ValueError(f"Inner dimension mismatch for B_op @ A_op. B_op cols: {cols_B_op}, A_op rows: {rows_A_op}")

    m_for_lib = rows_B_op
    n_for_lib = cols_A_op
    k_for_lib = cols_B_op # = rows_A_op

    if out is not None:
        torch.ops.bitsandbytes.int8_linear_matmul.out(A if not transposed_A else A.t(),
                                                      B if not transposed_B else B.t(), 
                                                      out) 
        return out
    else:
        raise NotImplementedError("Direct igemm C call logic is too complex to safely replicate here. Use torch.ops or backend methods.")

    return out


def batched_igemm( # Main PDF (page 56)
    A: Tensor, # int8, (batch, m, k)
    B: Tensor, # int8, (batch, k, n)
    out: Optional[torch.Tensor] = None, # Output int32, (batch, m, n)
    transposed_A: bool = False, # Transpose for A within each batch item
    transposed_B: bool = False, # Transpose for B within each batch item
):
    """Performs batched int8 matrix multiplication."""
    if A.device.type != 'cuda':
        raise NotImplementedError("batched_igemm is currently only implemented for CUDA/ROCm devices.")

    if A.shape[0] != B.shape[0]:
        raise ValueError(f"Batch dimensions for batched_igemm must match: A batch={A.shape[0]}, B batch={B.shape[0]}")


    sout_shape_item = check_matmul(A[0], B[0], out[0] if out is not None else None, transposed_A, transposed_B, expected_type=torch.int8)
    sout_shape_batch = (A.shape[0], *sout_shape_item)

    if out is None:
        out = torch.empty(sout_shape_batch, dtype=torch.int32, device=A.device)
    elif out.dtype != torch.int32:
        raise ValueError(f"Output tensor for batched_igemm must be int32, got {out.dtype}")

    logger.warning("functional.batched_igemm direct C call is complex; prefer torch.ops or backend methods if available for int8 bmm.")

    raise NotImplementedError("Direct batched_igemm C call logic is too complex. Use torch.ops or backend methods.")
    
    return out


def int8_linear_matmul(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor] = None, dtype=torch.int32): # Main PDF (page 58)
    """
    Performs int8_linear_matmul (A @ B.T). Uses torch.ops.
    This is already backend-aware due to torch.library dispatch.
    """
    if A.device.type == "cpu" and B.device.type == "cpu":
        # Main PDF cpu/ops.py has a direct torch._int_mm for torch >= 2.6
        if hasattr(torch, '_int_mm') and torch.__version__ >= "2.6": # Check torch version appropriately
             # This assumes A is activations, B is weights (transposed in _int_mm call)
             # A: (..., K), B: (N, K) for A @ B.T -> (..., N)
             # torch._int_mm(X, W.T()) where X is (M,K), W is (N,K)
             # So, B is the weight matrix here.
             if out is not None:
                 # torch._int_mm doesn't have an out argument in this simple form.
                 # Perform and then copy.
                 res = torch._int_mm(A.reshape(-1, A.shape[-1]), B.t()).reshape(*A.shape[:-1], B.shape[0])
                 out.copy_(res)
                 return out
             else:
                 return torch._int_mm(A.reshape(-1, A.shape[-1]), B.t()).reshape(*A.shape[:-1], B.shape[0])
        # Fall through to torch.ops if specific CPU version not met or not preferred.

    if out is not None:
        # Ensure out has correct dtype if provided, op might not check this before C call.
        if out.dtype != dtype:
            # This is tricky. The op expects out to be properly initialized.
            # It's better if the op itself handles dtype or if user provides correct out.
            # For now, assume op handles it or user is correct.
            pass
        torch.ops.bitsandbytes.int8_linear_matmul.out(A, B, out=out) # Pass B as is, op expects A @ B.T
        return out
    else:
        return torch.ops.bitsandbytes.int8_linear_matmul.default(A, B) # Pass B as is


def int8_mm_dequant( # Main PDF (page 59)
    A: torch.Tensor, # int32 result from int8_linear_matmul
    row_stats: torch.Tensor, # absmax for rows of original LHS matrix of matmul
    col_stats: torch.Tensor, # absmax for columns of original RHS matrix of matmul (which was transposed)
    out: Optional[torch.Tensor] = None, # Output tensor (float16 typically)
    bias: Optional[torch.Tensor] = None,
):
    """
    Dequantizes the int32 result of an int8 matrix multiplication. Uses torch.ops.
    Result = (A_int32 * row_stats * col_stats / (127*127)) + bias
    """
    # torch.ops.bitsandbytes.int8_mm_dequant.default(A, row_stats, col_stats, dtype=torch.float16, bias=bias)
    # The `dtype` in the op call is the *output* dtype.
    # The `out` argument in this Python function is for pre-allocated output.

    # Default output dtype if not specified by `out` tensor
    output_dtype = torch.float16 if out is None else out.dtype
    
    # The torch.ops definition from main PDF _ops.py:
    # "bitsandbytes::int8_mm_dequant", "(Tensor A, Tensor row_stats, Tensor col_stats, ScalarType? dtype=None, Tensor? bias=None) -> Tensor"
    # Here, `dtype` is an optional argument to the op itself, specifying the desired output type.

    # If `out` is provided, we should use its dtype for the op, and the op should write into `out`.
    # However, the standard torch.ops pattern doesn't usually have an `out` variant for functions that return a new tensor
    # AND also take a `dtype` for that new tensor.
    # The main PDF's cuda/ops.py for int8_mm_dequant creates `out_cuda = torch.empty_like(A, dtype=torch.float16)`
    # then calls lib.cdequant_mm_int32_fp16. This implies the kernel might be fp16 specific.
    # The torch.library.define for it does take `ScalarType? dtype`.

    if out is not None:
        # If out is provided, the op should ideally write to it.
        # Let's assume the op's `dtype` argument controls the computation/output type,
        # and if `out` is given, a copy might occur if dtypes don't match what op produces.
        # This is a bit ambiguous. A dedicated .out variant of the op would be clearer.
        # For now, call the default op and copy if `out` is provided.
        result = torch.ops.bitsandbytes.int8_mm_dequant.default(A, row_stats, col_stats, dtype=out.dtype, bias=bias)
        out.copy_(result)
        return out
    else:
        # If out is not provided, call the op and let it allocate. Default to float16.
        return torch.ops.bitsandbytes.int8_mm_dequant.default(A, row_stats, col_stats, dtype=torch.float16, bias=bias)


@deprecated("This function is deprecated. Use int8_vectorwise_quant or int8_double_quant.", category=FutureWarning)
def get_colrow_absmax( # Main PDF (page 59)
    A: torch.Tensor, # float16 input
    row_stats: Optional[torch.Tensor] = None, # Output for row absmax
    col_stats: Optional[torch.Tensor] = None, # Output for col absmax
    nnz_block_ptr: Optional[torch.Tensor] = None, # Not used in main PDF's get_colrow_absmax C call
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """GPU-only. Determines row-wise and column-wise absmax values for LLM.int8()."""
    if A.device.type != 'cuda':
        raise NotImplementedError("get_colrow_absmax is only implemented for CUDA/ROCm devices.")
    if A.dtype != torch.float16:
        # The C kernel cget_col_row_stats in main PDF was likely for fp16.
        raise ValueError(f"get_colrow_absmax expects float16 input, got {A.dtype}")

    rows_A = A.shape[0] if len(A.shape) == 2 else A.shape[0] * A.shape[1] # Flatten leading dims
    cols_A = A.shape[-1]

    if row_stats is None:
        row_stats = torch.empty(rows_A, dtype=torch.float32, device=A.device)
    if col_stats is None:
        col_stats = torch.empty(cols_A, dtype=torch.float32, device=A.device)
    
    # nnz_block_ptr was for sparse outliers, not directly used by cget_col_row_stats in main PDF's functional.py
    # but was mentioned in the CUDABackend refactored version.
    # The C function cget_col_row_stats in functional.py (main) did not take nnz_block_ptr.
    # It was `lib.cget_col_row_stats(ptrA, ptrRowStats, ptrColStats, ct.c_float(threshold), rows, cols)`
    # The `nnz_row_ptr` in refactored CUDABackend.double_quant was for `cdouble_rowcol_quant`.
    # For get_colrow_absmax, the main PDF's functional.py implementation was simpler:
    
    # Simpler version from main PDF functional.py (page 60), which uses Python for col_stats if not provided.
    # This seems more aligned with the original functional.py structure.
    # It also had a different C call `lib.cget_row_stats` for just row_stats.
    
    # The version on page 59 of main PDF using Python for outlier mask and col_stats:
    outlier_mask = None
    absA = A.abs() # Keep original dtype for now
    if len(A.shape) == 3: absA = absA.view(-1, A.shape[-1]) # Flatten if 3D

    if threshold > 0.0:
        outlier_mask = absA >= threshold
        # Create a view for masked_fill_ to avoid modifying original A if it's not desired by caller
        absA_masked = absA.clone() if outlier_mask.any() else absA
        absA_masked.masked_fill_(outlier_mask, 0.0)
        current_absA_for_stats = absA_masked
    else:
        current_absA_for_stats = absA

    if row_stats is None or True: # Always calculate if called, or fill provided.
                                  # Main PDF's get_row_absmax used a C kernel.
        # Python way:
        # calculated_row_stats = current_absA_for_stats.amax(dim=-1, keepdim=False).float()
        # row_stats.copy_(calculated_row_stats)
        # Using the C kernel from get_row_absmax (main PDF page 60) for row_stats:
        _row_stats_calc = torch.empty(rows_A, dtype=torch.float32, device=A.device)
        with _cuda_device_of(A):
            is_on_gpu([A, _row_stats_calc])
            lib.cget_row_stats( # Assumes A is contiguous or kernel handles strides
                get_ptr(A.contiguous()), get_ptr(_row_stats_calc), # Pass original A for kernel
                ct.c_float(threshold), # Kernel handles thresholding for row stats
                ct.c_int32(rows_A), ct.c_int32(cols_A),
                get_tensor_stream(A)
            )
        row_stats.copy_(_row_stats_calc)


    if col_stats is None or True:
        # Python way for col_stats (as in main PDF page 60 for get_colrow_absmax)
        calculated_col_stats = current_absA_for_stats.amax(dim=0, keepdim=False).float()
        col_stats.copy_(calculated_col_stats)
        
    return row_stats, col_stats, outlier_mask


@deprecated("This function is deprecated. Use int8_vectorwise_quant.", category=FutureWarning)
def get_row_absmax(A: torch.Tensor, threshold: float = 0.0): # Main PDF (page 60)
    """GPU-only. Determine row-wise absmax for LLM.int8()."""
    if A.device.type != 'cuda':
        raise NotImplementedError("get_row_absmax is only implemented for CUDA/ROCm devices.")
    if A.dtype != torch.float16:
        raise ValueError(f"get_row_absmax expects float16 input, got {A.dtype}")

    rows = A.shape[0] if len(A.shape) == 2 else A.shape[0] * A.shape[1]
    cols = A.shape[-1]
    
    row_stats_out = torch.empty(rows, dtype=torch.float32, device=A.device)
    
    with _cuda_device_of(A):
        is_on_gpu([A, row_stats_out])
        # Ensure A is contiguous if the kernel expects it.
        A_contig = A.contiguous() if not A.is_contiguous() else A
        lib.cget_row_stats(
            get_ptr(A_contig), get_ptr(row_stats_out),
            ct.c_float(threshold),
            ct.c_int32(rows), ct.c_int32(cols),
            get_tensor_stream(A) # Get stream for the operation
        )
    return row_stats_out


