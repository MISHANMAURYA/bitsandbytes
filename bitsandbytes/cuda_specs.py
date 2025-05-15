'''
import dataclasses
from functools import lru_cache
from typing import Optional

import torch


@dataclasses.dataclass(frozen=True)
class CUDASpecs:
    highest_compute_capability: tuple[int, int]
    cuda_version_string: str
    cuda_version_tuple: tuple[int, int]

    @property
    def has_imma(self) -> bool:
        return torch.version.hip or self.highest_compute_capability >= (7, 5)


def get_compute_capabilities() -> list[tuple[int, int]]:
    return sorted(torch.cuda.get_device_capability(torch.cuda.device(i)) for i in range(torch.cuda.device_count()))


@lru_cache(None)
def get_cuda_version_tuple() -> Optional[tuple[int, int]]:
    """Get CUDA/HIP version as a tuple of (major, minor)."""
    try:
        if torch.version.cuda:
            version_str = torch.version.cuda
        elif torch.version.hip:
            version_str = torch.version.hip
        else:
            return None

        parts = version_str.split(".")
        if len(parts) >= 2:
            return tuple(map(int, parts[:2]))
        return None
    except (AttributeError, ValueError, IndexError):
        return None


def get_cuda_version_string() -> Optional[str]:
    """Get CUDA/HIP version as a string."""
    version_tuple = get_cuda_version_tuple()
    if version_tuple is None:
        return None
    major, minor = version_tuple
    return f"{major * 10 + minor}"


def get_cuda_specs() -> Optional[CUDASpecs]:
    """Get CUDA/HIP specifications."""
    if not torch.cuda.is_available():
        return None

    try:
        compute_capabilities = get_compute_capabilities()
        if not compute_capabilities:
            return None

        version_tuple = get_cuda_version_tuple()
        if version_tuple is None:
            return None

        version_string = get_cuda_version_string()
        if version_string is None:
            return None

        return CUDASpecs(
            highest_compute_capability=compute_capabilities[-1],
            cuda_version_string=version_string,
            cuda_version_tuple=version_tuple,
        )
    except Exception:
        return None
'''


import dataclasses
import logging 
import re 
import subprocess 
from functools import lru_cache 
from typing import List, Optional, Tuple 

import torch


try:
    from bitsandbytes.cextension import HIP_ENVIRONMENT, BNB_HIP_VERSION
except ImportError:
   
    HIP_ENVIRONMENT = hasattr(torch.version, 'hip') and torch.version.hip is not None
    if HIP_ENVIRONMENT:
        hip_major_temp, hip_minor_temp = map(int, torch.version.hip.split(".")[:2])
        BNB_HIP_VERSION = hip_major_temp * 100 + hip_minor_temp
    else:
        BNB_HIP_VERSION = 0


logger = logging.getLogger(__name__) # Added from refactored

@dataclasses.dataclass(frozen=True)
class CUDASpecs: 
    highest_compute_capability: Tuple[int, int] 
    cuda_version_string: str 
    cuda_version_tuple: Tuple[int, int] 

    @property
    def has_cublaslt(self) -> bool: 
        """
        Checks if the GPU environment supports operations typically accelerated by cuBLASLt or hipBLASLt.
        For CUDA, this usually means Compute Capability >= 7.5.
        For ROCm, this depends on the ROCm version and GPU architecture.
        """
        global HIP_ENVIRONMENT, BNB_HIP_VERSION # Access globals

        if HIP_ENVIRONMENT:
            return BNB_HIP_VERSION >= 601
        else: # CUDA
            return self.highest_compute_capability >= (7, 5)

  
    def has_cublaslt_override(self, override_version_str: str) -> bool:
        """
        Checks if a given CUDA version string (from override) would support cuBLASLt.
        This is primarily for NVIDIA CUDA.
        """
        if HIP_ENVIRONMENT: 
            return BNB_HIP_VERSION >= 601
        return self.highest_compute_capability >= (7, 5)


def get_compute_capabilities() -> List[Tuple[int, int]]:
    """Gets the compute capabilities of all available CUDA devices."""
    if not torch.cuda.is_available() or (hasattr(torch.version, 'hip') and torch.version.hip is not None):
        if HIP_ENVIRONMENT:
            return [(0,0)]                           
        return [] 

    try:
        return sorted(torch.cuda.get_device_capability(torch.cuda.device(i)) for i in range(torch.cuda.device_count()))
    except RuntimeError as e:
        logger.warning(f"Could not get CUDA compute capabilities: {e}. This might happen on WSL.")
        return [(0,0)] 


@lru_cache(None) 
def get_cuda_version_tuple() -> Optional[Tuple[int, int]]:
    """Get CUDA/HIP version as a tuple of (major, minor)."""
    try:
        version_str = None
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            version_str = torch.version.cuda
        elif hasattr(torch.version, 'hip') and torch.version.hip:
            version_str = torch.version.hip
        else:
            return None

        if version_str:
            parts = version_str.split(".")
            if len(parts) >= 2:
                return tuple(map(int, parts[:2]))
        return None
    except (AttributeError, ValueError, IndexError) as e:
        logger.warning(f"Could not parse CUDA/HIP version: {e}")
        return None

def get_cuda_version_string() -> Optional[str]:
    """
    Get CUDA/HIP version as a string (e.g., "118" for 11.8, "57" for 5.7).
    This string is used in library naming.
    """
    version_tuple = get_cuda_version_tuple()
    if version_tuple is None:
        return None
    major, minor = version_tuple
    return f"{major}{minor}"


def get_rocm_gpu_arch() -> str:
    """Detects the ROCm GPU architecture (e.g., gfx90a, gfx942)."""

    global HIP_ENVIRONMENT
    if not HIP_ENVIRONMENT:
        return "unknown"
    try:
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            match = re.search(r"Name:\s+gfx([a-zA-Z\d]+)", result.stdout)
            if match:
                return "gfx" + match.group(1)
        logger.warning(f"rocminfo command failed or did not return GPU name. stdout: {result.stdout}, stderr: {result.stderr}")
        return "unknown"
    except FileNotFoundError:
        logger.warning("rocminfo command not found. Cannot detect ROCm GPU architecture.")
        return "unknown"
    except Exception as e:
        logger.error(f"An error occurred while trying to detect ROCm GPU architecture: {e}")
        return "unknown"


@lru_cache(None)
def get_cuda_specs() -> Optional[CUDASpecs]:
    """Get CUDA/HIP specifications."""
    if not torch.cuda.is_available(): 
        return None

    try:
        compute_caps = get_compute_capabilities()
        if not compute_caps:
             return None
        highest_cc = compute_caps[-1]

        version_tuple = get_cuda_version_tuple()
        if version_tuple is None:
            return None

        version_string = get_cuda_version_string() 
        if version_string is None:
            return None

        return CUDASpecs(
            highest_compute_capability=highest_cc,
            cuda_version_string=version_string,
            cuda_version_tuple=version_tuple,
        )
    except Exception as e:
        logger.error(f"Error gathering GPU specs: {e}", exc_info=True)
        return None


