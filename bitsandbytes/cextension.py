'''
import ctypes as ct
import logging
import os
from pathlib import Path
import re

import torch

from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX, PACKAGE_DIR
from bitsandbytes.cuda_specs import CUDASpecs, get_cuda_specs

logger = logging.getLogger(__name__)


def get_cuda_bnb_library_path(cuda_specs: CUDASpecs) -> Path:
    """
    Get the disk path to the CUDA BNB native library specified by the
    given CUDA specs, taking into account the `BNB_CUDA_VERSION` override environment variable.

    The library is not guaranteed to exist at the returned path.
    """

    prefix = "rocm" if torch.version.hip else "cuda"
    library_name = f"libbitsandbytes_{prefix}{cuda_specs.cuda_version_string}{DYNAMIC_LIBRARY_SUFFIX}"

    override_value = os.environ.get("BNB_CUDA_VERSION")
    if override_value:
        library_name = re.sub(r"cuda\d+", f"cuda{override_value}", library_name, count=1)
        logger.warning(
            f"WARNING: BNB_CUDA_VERSION={override_value} environment variable detected; loading {library_name}.\n"
            "This can be used to load a bitsandbytes version that is different from the PyTorch CUDA version.\n"
            "If this was unintended set the BNB_CUDA_VERSION variable to an empty string: export BNB_CUDA_VERSION=\n"
            "If you use the manual override make sure the right libcudart.so is in your LD_LIBRARY_PATH\n"
            "For example by adding the following to your .bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_cuda_dir/lib64\n",
        )

    return PACKAGE_DIR / library_name


class BNBNativeLibrary:
    _lib: ct.CDLL
    compiled_with_cuda = False

    def __init__(self, lib: ct.CDLL):
        self._lib = lib

    def __getattr__(self, item):
        return getattr(self._lib, item)

    def __getitem__(self, item):
        return getattr(self._lib, item)


class CudaBNBNativeLibrary(BNBNativeLibrary):
    compiled_with_cuda = True

    def __init__(self, lib: ct.CDLL):
        super().__init__(lib)
        lib.get_context.restype = ct.c_void_p
        lib.get_cusparse.restype = ct.c_void_p
        lib.cget_managed_ptr.restype = ct.c_void_p


def get_native_library() -> BNBNativeLibrary:
    binary_path = PACKAGE_DIR / f"libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}"
    cuda_specs = get_cuda_specs()
    if cuda_specs:
        cuda_binary_path = get_cuda_bnb_library_path(cuda_specs)
        if cuda_binary_path.exists():
            binary_path = cuda_binary_path
        else:
            logger.warning("Could not find the bitsandbytes CUDA binary at %r", cuda_binary_path)
    logger.debug(f"Loading bitsandbytes native library from: {binary_path}")
    dll = ct.cdll.LoadLibrary(str(binary_path))

    if hasattr(dll, "get_context"):  # only a CUDA-built library exposes this
        return CudaBNBNativeLibrary(dll)

    logger.warning(
        "The installed version of bitsandbytes was compiled without GPU support. "
        "8-bit optimizers and GPU quantization are unavailable.",
    )
    return BNBNativeLibrary(dll)


try:
    lib = get_native_library()
except Exception as e:
    lib = None
    logger.error(f"Could not load bitsandbytes native library: {e}", exc_info=True)
    if torch.cuda.is_available():
        logger.warning(
            """
CUDA Setup failed despite CUDA being available. Please run the following command to get more information:

python -m bitsandbytes

Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
and open an issue at: https://github.com/bitsandbytes-foundation/bitsandbytes/issues
""",
        )
'''


import ctypes as ct
import logging
import os
from pathlib import Path
import re # Added from refactored
from typing import Optional 

import torch

from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX, PACKAGE_DIR

from bitsandbytes.cuda_specs import CUDASpecs, get_cuda_specs, get_cuda_version_tuple, get_rocm_gpu_arch

logger = logging.getLogger(__name__)


HIP_ENVIRONMENT = False
BNB_HIP_VERSION = 0
BNB_HIP_VERSION_SHORT = ""
BNB_BACKEND = "CUDA" 
ROCM_GPU_ARCH = "unknown"


def get_cuda_bnb_library_path(cuda_specs: CUDASpecs) -> Path:
    """
    Get the disk path to the CUDA or ROCm BNB native library specified by the
    given GPU specs, taking into account override environment variables.
    The library is not guaranteed to exist at the returned path.
    """
    global HIP_ENVIRONMENT, BNB_HIP_VERSION_SHORT, BNB_BACKEND 

    is_hip_environment = hasattr(torch.version, 'hip') and torch.version.hip is not None

    if is_hip_environment:
        
        current_bnb_hip_version = 0
        if BNB_HIP_VERSION: 
            current_bnb_hip_version = BNB_HIP_VERSION
        elif hasattr(torch.version, 'hip') and torch.version.hip:
            hip_major_local, hip_minor_local = map(int, torch.version.hip.split(".")[:2])
            current_bnb_hip_version = hip_major_local * 100 + hip_minor_local

        if current_bnb_hip_version > 0 and current_bnb_hip_version < 601:
            library_name = f"libbitsandbytes_rocm{cuda_specs.cuda_version_string}_nohipblaslt{DYNAMIC_LIBRARY_SUFFIX}"
        else:
            library_name = f"libbitsandbytes_rocm{cuda_specs.cuda_version_string}{DYNAMIC_LIBRARY_SUFFIX}"

        override_value = os.environ.get("BNB_ROCM_VERSION") or os.environ.get("BNB_CUDA_VERSION") 
        if override_value:
 
            library_name_stem, _, library_name_ext = library_name.rpartition(".")
            library_name_stem = library_name_stem.replace(f"rocm{cuda_specs.cuda_version_string}", f"rocm{override_value}")
            library_name = f"{library_name_stem}.{library_name_ext}"
            logger.warning(
                f"WARNING: BNB_ROCM_VERSION (or BNB_CUDA_VERSION as fallback)={override_value} environment variable detected; loading {library_name}.\n"
  
            )
        return PACKAGE_DIR / library_name

    else: 
        library_name = f"libbitsandbytes_cuda{cuda_specs.cuda_version_string}"
    
        if not cuda_specs.has_cublaslt: 
            library_name += "_nocublaslt"
        library_name = f"{library_name}{DYNAMIC_LIBRARY_SUFFIX}"

        override_value = os.environ.get("BNB_CUDA_VERSION")
        if override_value:
         
            library_name_stem, _, library_name_ext = library_name.rpartition(".") 
            current_version_str = cuda_specs.cuda_version_string 
            base_stem = library_name_stem.replace(f"cuda{current_version_str}", "cuda") 
            if "_nocublaslt" in base_stem:
                 final_stem = base_stem.replace("_nocublaslt", "") + override_value + "_nocublaslt" \
                              if not cuda_specs.has_cublaslt_override(override_value) else base_stem.replace("_nocublaslt", "") + override_value
            else:
                 final_stem = base_stem + override_value

            library_name = f"{final_stem}.{library_name_ext}"

            logger.warning(
                f"WARNING: BNB_CUDA_VERSION={override_value} environment variable detected; loading {library_name}.\n"
                "This can be used to load a bitsandbytes version that is different from the PyTorch CUDA version.\n"
                "If this was unintended set the BNB_CUDA_VERSION variable to an empty string: export BNB_CUDA_VERSION=\n"
                "If you use the manual override make sure the right libcudart.so is in your LD_LIBRARY_PATH\n"
                "For example by adding the following to your .bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_cuda_dir/lib64\n",
            )
        return PACKAGE_DIR / library_name


class BNBNativeLibrary:
    _lib: ct.CDLL
    compiled_with_cuda = False # More accurately, compiled_with_gpu

    def __init__(self, lib: ct.CDLL):
        self._lib = lib

    def __getattr__(self, name):
 
        def throw_on_call(*args, **kwargs):
            if hasattr(self._lib, name): # Check if the underlying C library has the method
                return getattr(self._lib, name)(*args, **kwargs)
   
            raise RuntimeError(
                f"Method '{name}' not available in CPU-only or incorrectly loaded version of bitsandbytes.\n"
                "If you intended to use GPU features, reinstall with GPU support or check CUDA/ROCm setup."
            )
        return throw_on_call

    def __getitem__(self, item): 
        return self.__getattr__(item)


class CudaBNBNativeLibrary(BNBNativeLibrary):
    compiled_with_cuda = True

    def __init__(self, lib: ct.CDLL):
        super().__init__(lib)
        lib.get_context.restype = ct.c_void_p 
        lib.cget_managed_ptr.restype = ct.c_void_p

        global HIP_ENVIRONMENT
        if HIP_ENVIRONMENT: 
            if hasattr(lib, 'get_hipsparse'):
                 lib.get_hipsparse.restype = ct.c_void_p
            else:
                 logger.warning("hipSPARSE handle function 'get_hipsparse' not found in the loaded library.")
        else: # CUDA
            if hasattr(lib, 'get_cusparse'): 
                lib.get_cusparse.restype = ct.c_void_p
            else:
                logger.warning("cuSPARSE handle function 'get_cusparse' not found in the loaded library.")


def get_available_cuda_binary_versions() -> list[str]: 
    """Get formatted CUDA/ROCm versions from existing library files."""

    lib_pattern_cuda = f"libbitsandbytes_cuda*{DYNAMIC_LIBRARY_SUFFIX}"
    lib_pattern_rocm = f"libbitsandbytes_rocm*{DYNAMIC_LIBRARY_SUFFIX}"
    versions = []
    for lib_pattern in [lib_pattern_cuda, lib_pattern_rocm]:
        for lib_path in Path(__file__).parent.glob(lib_pattern):
            # For CUDA: libbitsandbytes_cuda118.so -> 11.8
            match_cuda = re.search(r"cuda(\d{3})", lib_path.name)
            if match_cuda:
                ver_code = int(match_cuda.group(1))
                major = ver_code // 10
                minor = ver_code % 10
                versions.append(f"CUDA {major}.{minor}")

            match_rocm = re.search(r"rocm(\d{2,3})", lib_path.name) 
            if match_rocm:
                ver_str = match_rocm.group(1)
                if len(ver_str) == 2: # e.g. "57"
                    major = int(ver_str[0])
                    minor = int(ver_str[1])
                elif len(ver_str) == 3: 
                    major = int(ver_str[0]) 
                    minor = int(ver_str[1:]) 
                                             
                    if len(ver_str) == 3 and ver_str.startswith('6'): 
                        minor = int(ver_str[1]) 
                    else: # for 118 style
                        major = int(ver_str[:2]) if len(ver_str) == 3 else int(ver_str[0])
                        minor = int(ver_str[2]) if len(ver_str) == 3 else int(ver_str[1])


                versions.append(f"ROCm {major}.{minor}")
    return sorted(list(set(versions)))

def parse_cuda_version(version_str: str, is_rocm: bool) -> str: 
    """Convert raw version string (e.g. '118' or '57') to formatted version (e.g. '11.8' or '5.7')."""
    if version_str.isdigit():
        if is_rocm: 
            if len(version_str) == 2:
                 return f"{version_str[0]}.{version_str[1]}"
            elif len(version_str) == 3: 
                 return f"{version_str[0]}.{version_str[1]}" 
        else: # CUDA e.g. "118" -> "11.8"
            if len(version_str) == 3:
                return f"{version_str[:2]}.{version_str[2]}"
    return version_str  


class ErrorHandlerMockBNBNativeLibrary(BNBNativeLibrary):

    def __init__(self, error_msg: str):
        self.error_msg = error_msg
        self.user_gpu_version_tuple = get_cuda_version_tuple() 
        self.is_rocm_env = hasattr(torch.version, 'hip') and torch.version.hip is not None
        self.backend_name = "ROCm" if self.is_rocm_env else "CUDA"

        self.available_versions = get_available_cuda_binary_versions() 

        self.override_env_var = "BNB_ROCM_VERSION" if self.is_rocm_env else "BNB_CUDA_VERSION"
        self.override_value = os.environ.get(self.override_env_var)
        if not self.override_value and self.is_rocm_env: 
            self.override_value = os.environ.get("BNB_CUDA_VERSION")

        self.requested_version_str = (
            self.override_value
            if self.override_value
            else (self.user_gpu_version_tuple[0] * 10 + self.user_gpu_version_tuple[1] if self.user_gpu_version_tuple and not self.is_rocm_env else \
                  str(self.user_gpu_version_tuple[0]) + str(self.user_gpu_version_tuple[1]) if self.user_gpu_version_tuple and self.is_rocm_env else "unknown_raw")
        )
        self.requested_version_formatted = parse_cuda_version(self.requested_version_str, self.is_rocm_env)


        self.user_gpu_version_formatted = (
            f"{self.user_gpu_version_tuple[0]}.{self.user_gpu_version_tuple[1]}"
            if self.user_gpu_version_tuple
            else "unknown"
        )

        if "cannot open shared object file" in error_msg or "No such file or directory" in error_msg:
            self.formatted_error = self._format_dependency_error()
        else:
            self.formatted_error = self._format_lib_error_message(
                available_versions=self.available_versions,
                user_gpu_version=self.user_gpu_version_formatted,
                original_error=f"Original error: {self.error_msg}\n" if self.error_msg else "",
                requested_version=self.requested_version_formatted,
            )

    def _format_lib_error_message(
        self,
        available_versions: list[str],
        user_gpu_version: str,
        original_error: str = "",
        requested_version: Optional[str] = None,
    ) -> str:
        analysis = ""
        no_cpu_lib_found = "libbitsandbytes_cpu.so: cannot open" in original_error or \
                           "libbitsandbytes_cpu.dylib: cannot open" in original_error or \
                           "libbitsandbytes_cpu.dll: cannot open" in original_error
        # Generic "binary not found" check
        no_gpu_lib_found = f"{self.backend_name} binary not found" in original_error or \
                           (not any(self.backend_name in v for v in available_versions) and self.user_gpu_version_tuple is not None)


        if no_cpu_lib_found:
            analysis = f"\nFailed to load CPU-only bitsandbytes library ({PACKAGE_DIR / f'libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}'})\n\n"
        elif no_gpu_lib_found or (requested_version not in [v.split(' ')[1] for v in available_versions if self.backend_name in v]):
            version_list_str = "\n  " + "\n  ".join(available_versions) if available_versions else "  NONE"
            analysis = (
                f"\n{self.backend_name} VERSION MISMATCH\n"
                f"  Requested {self.backend_name} version: {requested_version}\n"
                f"  Detected PyTorch {self.backend_name} version: {user_gpu_version}\n"
                f"  Available pre-compiled versions for {self.backend_name}: {version_list_str}\n\n"
                "This means:\n"
            )
            if available_versions:
                 analysis += "  The version you're trying to use is NOT distributed with this package of bitsandbytes.\n\n"
            else:
                 analysis += (
                    f"  No pre-compiled {self.backend_name} binaries were found in the package.\n"
                    "  Forgot to compile the bitsandbytes library from source?\n"
                    "    1. You're not using the bitsandbytes package but a checked-out source code.\n"
                    "    2. You MUST compile from source.\n\n"
                 )

        base_msg = "Attempted to use bitsandbytes native library functionality but it's not available.\n\n"
        troubleshooting = (
            "This typically happens when:\n"
            f"  1. bitsandbytes doesn't ship with a pre-compiled binary for your {self.backend_name} version ({requested_version} for PyTorch {user_gpu_version}).\n"
            "  2. The library wasn't compiled properly during installation from source.\n\n"
        ) if no_gpu_lib_found or (requested_version not in [v.split(' ')[1] for v in available_versions if self.backend_name in v]) else \
        "This typically happens when you checked the code out from source and your torch installation doesn't detect a GPU on your machine.\n\n"

        note = (
            f"To make bitsandbytes work, the compiled library version MUST exactly match the linked {self.backend_name} version.\n"
            f"If your {self.backend_name} version ({user_gpu_version}) doesn't have a pre-compiled binary, you MUST compile from source.\n\n"
        ) if no_gpu_lib_found or (requested_version not in [v.split(' ')[1] for v in available_versions if self.backend_name in v]) else ""

        compile_instructions = (
            "You have two options:\n"
            "  1. COMPILE FROM SOURCE (required if no binary exists for your version):\n"
            f"     Follow instructions at {PACKAGE_DIR.parent / 'docs' / 'compile.md'} (adjust path if needed) or the main README.\n" 
            f"     Ensure you use the correct COMPUTE_BACKEND ({'cuda' if not self.is_rocm_env else 'hip'}).\n"
            f"  2. Use {self.override_env_var} to specify a DIFFERENT {self.backend_name} version from the detected one ({user_gpu_version}),\n"
            f"     which is installed on your machine AND matches an available pre-compiled version listed above.\n\n"
        ) if no_gpu_lib_found or (requested_version not in [v.split(' ')[1] for v in available_versions if self.backend_name in v]) else \
        "COMPILE FROM SOURCE for CPU-only:\n  cmake -DCOMPUTE_BACKEND=cpu -S. && make \n\n"

        diagnostics = (
            "Run this command for detailed diagnostics:\n"
            "  python -m bitsandbytes\n\n"
            "If you've tried everything and still have issues:\n"
            "  1. Include ALL version info (operating system, bitsandbytes, PyTorch, CUDA/ROCm, Python).\n"
            "  2. Describe what you've tried in detail.\n"
            f"  3. Open an issue with this information at {PACKAGE_DIR.parent / 'issues'}\n\n"
        )
        return f"{analysis}{base_msg}{troubleshooting}{note}{compile_instructions}{original_error}\n{diagnostics}"

    def _format_dependency_error(self) -> str:
        error_parts = self.error_msg.split(":")
        missing_lib_candidate = error_parts[0].strip() if len(error_parts) > 0 else "unknown library"
        missing_lib_match = re.search(r"(lib\w+\.(?:so|dylib|dll)(?:\.\d+)*)", self.error_msg)
        missing_lib = missing_lib_match.group(1) if missing_lib_match else missing_lib_candidate


        gpu_version_major_str = (
            self.requested_version_formatted.split(".")[0]
            if "." in self.requested_version_formatted and self.requested_version_formatted != "unknown"
            else self.requested_version_str # Fallback to raw if formatted is unknown
        )
        if gpu_version_major_str == "unknown_raw" and self.user_gpu_version_tuple:
            gpu_version_major_str = str(self.user_gpu_version_tuple[0])


        return (
            f"\n{self.backend_name} SETUP ERROR: Missing dependency: {missing_lib}\n\n"
            f"{self.backend_name} {gpu_version_major_str}.x runtime libraries were not found in your LD_LIBRARY_PATH.\n\n"
            f"To fix this, make sure that:\n"
            f"  1. You have installed the {self.backend_name} toolkit version {gpu_version_major_str}.x on your system.\n"
            f"  2. The {self.backend_name} runtime libraries are in your LD_LIBRARY_PATH.\n\n"
            f"You can add them by (example for {self.backend_name} {gpu_version_major_str}.x, adjust path as needed):\n"
            f"  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/{self.backend_name.lower()}-{gpu_version_major_str}.x/lib64\n"
            f"  (For ROCm, this might be /opt/rocm-{gpu_version_major_str}.x/lib or similar)\n"
            "  Persist this change by adding the line to your .bashrc or .zshrc file.\n\n"
            f"Original error: {self.error_msg}\n\n"
            "Run this command for detailed diagnostics:\n"
            "  python -m bitsandbytes\n\n"
            "If you've tried everything and still have issues:\n"
            "  1. Include ALL version info (operating system, bitsandbytes, PyTorch, CUDA/ROCm, Python).\n"
            "  2. Describe what you've tried in detail.\n"
            f"  3. Open an issue with this information at {PACKAGE_DIR.parent / 'issues'}\n\n" 
        )

    def __getattr__(self, name):
        def throw_on_call(*args, **kwargs):
            raise RuntimeError(f"{self.formatted_error}Native code method attempted to call: lib.{name}()")
        return throw_on_call

    def __getitem__(self, name):
        return self.__getattr__(name)


def get_native_library() -> BNBNativeLibrary:
    """
    Loads the appropriate native library (CPU, CUDA, or ROCm).
    Uses ErrorHandlerMockBNBNativeLibrary as a fallback if loading fails.
    """
    global HIP_ENVIRONMENT, BNB_BACKEND # Access globals

    binary_path = PACKAGE_DIR / f"libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}"
    is_gpu_attempt = False

    cuda_specs = get_cuda_specs() 

    if cuda_specs: 
        is_gpu_attempt = True
        gpu_binary_path = get_cuda_bnb_library_path(cuda_specs) 
        if gpu_binary_path.exists():
            binary_path = gpu_binary_path
        else:
            specific_error = f"{BNB_BACKEND} binary not found at {gpu_binary_path}. " \
                             f"PyTorch detected {BNB_BACKEND} version {cuda_specs.cuda_version_tuple[0]}.{cuda_specs.cuda_version_tuple[1]}, " \
                             f"and bitsandbytes sought a library compatible with this version."
            logger.error(specific_error) # Log the specific error
            return ErrorHandlerMockBNBNativeLibrary(specific_error)

    logger.debug(f"Loading bitsandbytes native library from: {binary_path}")

    try:
        dll = ct.cdll.LoadLibrary(str(binary_path))
        if hasattr(dll, "get_context"):
            return CudaBNBNativeLibrary(dll) # For both CUDA and ROCm GPU builds
        elif is_gpu_attempt and not binary_path.name.startswith("libbitsandbytes_cpu"):
         
             logger.warning(
                f"Attempted to load GPU library {binary_path} but it does not seem to be a GPU build (missing 'get_context'). "
                f"Falling back to CPU library if available, or erroring out."
            )

             cpu_path = PACKAGE_DIR / f"libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}"
             if cpu_path.exists() and binary_path != cpu_path :
                 logger.info(f"Trying to load CPU library from: {cpu_path}")
                 dll = ct.cdll.LoadLibrary(str(cpu_path))
                 if hasattr(dll, "get_context"):
                     logger.error("CPU library unexpectedly has 'get_context'. This is a bug.")
                     return ErrorHandlerMockBNBNativeLibrary("CPU library has GPU symbols.")
                 logger.warning("Successfully loaded CPU library as a fallback.")
                 return BNBNativeLibrary(dll) # Loaded CPU library
             else: # CPU library also not found or was the one that failed
                 return ErrorHandlerMockBNBNativeLibrary(f"Failed to load both target GPU library and CPU library. Last attempt: {binary_path}")

        logger.info(
            "Successfully loaded CPU-only bitsandbytes library. "
            "8-bit optimizers and GPU quantization are unavailable."
        )
        return BNBNativeLibrary(dll)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"bitsandbytes library load error: {error_msg}", exc_info=True)
        return ErrorHandlerMockBNBNativeLibrary(error_msg)

try:
    if hasattr(torch.version, 'hip') and torch.version.hip:
        hip_major, hip_minor = map(int, torch.version.hip.split(".")[:2])
        HIP_ENVIRONMENT, BNB_HIP_VERSION = True, hip_major * 100 + hip_minor
        BNB_HIP_VERSION_SHORT = f"{hip_major}{hip_minor}"
        BNB_BACKEND = "ROCm"
    elif hasattr(torch.version, 'cuda') and torch.version.cuda:
        HIP_ENVIRONMENT, BNB_HIP_VERSION = False, 0
        BNB_HIP_VERSION_SHORT = ""
        BNB_BACKEND = "CUDA"
    else: # Neither CUDA nor HIP in torch.version, assume CPU or other non-GPU torch
        HIP_ENVIRONMENT, BNB_HIP_VERSION = False, 0
        BNB_HIP_VERSION_SHORT = ""
        BNB_BACKEND = "CPU" # Or some other indicator

    ROCM_GPU_ARCH = get_rocm_gpu_arch() if HIP_ENVIRONMENT else "unknown"

    lib = get_native_library()

except Exception as e: # Catch any error during the initial setup
    lib = None # Ensure lib is None if setup fails
    logger.error(f"bitsandbytes C extension setup failed: {e}", exc_info=True)
    if torch.cuda.is_available(): # Generic check for any GPU registered with PyTorch as "cuda"
        detected_backend = "ROCm" if hasattr(torch.version, 'hip') and torch.version.hip else "CUDA"
        logger.warning(
            f"""
{detected_backend} Setup failed despite {detected_backend} being available in PyTorch.
Refer to the error message above for details.
To get more diagnostic information, run:
    python -m bitsandbytes
If you suspect a bug, please include the output of the diagnostic command
and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
""",
        )
