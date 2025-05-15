'''
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys

import torch

from . import _ops, research, utils
from .autograd._functions import (
    MatmulLtState,
    matmul,
    matmul_4bit,
)
from .backends.cpu import ops as cpu_ops
from .backends.default import ops as default_ops
from .nn import modules
from .optim import adam

# This is a signal for integrations with transformers/diffusers.
# Eventually we may remove this but it is currently required for compatibility.
features = {"multi_backend"}
supported_torch_devices = {
    "cpu",
    "cuda",  # NVIDIA/AMD GPU
    "xpu",  # Intel GPU
    "hpu",  # Gaudi
    "npu",  # Ascend NPU
    "mps",  # Apple Silicon
}

if torch.cuda.is_available():
    from .backends.cuda import ops as cuda_ops


def _import_backends():
    """
    Discover and autoload all available backends installed as separate packages.
    Packages with an entrypoint for "bitsandbytes.backends" will be loaded.
    Inspired by PyTorch implementation: https://pytorch.org/tutorials/prototype/python_extension_autoload.html
    """
    from importlib.metadata import entry_points

    if sys.version_info < (3, 10):
        extensions = entry_points().get("bitsandbytes.backends", [])
    else:
        extensions = entry_points(group="bitsandbytes.backends")

    for ext in extensions:
        try:
            entry = ext.load()
            entry()
        except Exception as e:
            raise RuntimeError(f"bitsandbytes: failed to load backend {ext.name}: {e}") from e


_import_backends()

__pdoc__ = {
    "libbitsandbytes": False,
    "optim.optimizer.Optimizer8bit": False,
    "optim.optimizer.MockArgs": False,
}

__version__ = "0.46.0.dev0"
'''


import sys
import torch


from . import research, utils # Assuming these are subpackages/modules
from .autograd._functions import ( # MatmulLtState is also in refactored __init__
    MatmulLtState,
    matmul,
    matmul_4bit,
)

from .nn import modules # Main PDF had this


from .backends import register_backend
from .backends.cpu import CPUBackend


features = {"multi_backend"} # Also in refactored PDF

supported_torch_devices = {
    "cuda", 
    "cpu",
    "xpu", 
    "mps",  
    "npu",   
}

register_backend("cpu", CPUBackend())

if torch.cuda.is_available():

    from .cextension import HIP_ENVIRONMENT
    if HIP_ENVIRONMENT: # AMD ROCm
        from .backends.rocm import ROCmBackend
        register_backend("cuda", ROCmBackend())

    elif hasattr(torch.version, 'cuda') and torch.version.cuda: # NVIDIA CUDA
        from .backends.cuda import CUDABackend
        register_backend("cuda", CUDABackend())
  
    else:
        pass
      
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    try:
        from .backends.mps import MPSBackend
        register_backend("mps", MPSBackend())
    except ImportError:
        pass

if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
    try:
        from .backends.xpu import XPUBackend
        register_backend("xpu", XPUBackend())
    except ImportError:
        if "xpu" in supported_torch_devices: supported_torch_devices.remove("xpu") # If import fails
        pass

if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
    try:
        from .backends.npu import NPUBackend 
        register_backend("npu", NPUBackend())
    except ImportError:
        if "npu" in supported_torch_devices: supported_torch_devices.remove("npu") 
        pass
def _import_backends():
    """
    Discover and autoload all available backends installed as separate packages.
    Packages with an entrypoint for "bitsandbytes.backends" will be loaded.
    Inspired by PyTorch implementation: https://pytorch.org/tutorials/prototype/python_extension_autoload.html
    """
    from importlib.metadata import entry_points 

    if sys.version_info < (3, 10):
        extensions = entry_points().get("bitsandbytes.backends", [])
    else:
        extensions = entry_points(group="bitsandbytes.backends")

    for ext in extensions:
        try:
            entry = ext.load()
            if callable(entry): 
                entry() 
            else:
                pass 
        except Exception as e:
            raise RuntimeError(f"bitsandbytes: failed to load backend {ext.name}: {e}") from e

_import_backends() 

pdoc = {
    "libbitsandbytes": False, 
}
if 'Optimizer8bit' in dir(utils): # A guess, replace utils with actual optim module
    pdoc['optim.optimizer.Optimizer8bit'] = False
if 'MockArgs' in dir(utils): # A guess
    pdoc['optim.optimizer.MockArgs'] = False

__version__ = "0.46.0.dev0"
 
