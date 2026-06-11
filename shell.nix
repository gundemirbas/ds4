let
  pkgs = import <nixpkgs> {
    config = {
      rocmSupport = true;
      cudaSupport = true;
      allowUnfree = true;
    };
  };

  rocmPkgs = pkgs.rocmPackages;
  cudaPkgs = pkgs.cudaPackages;

in pkgs.mkShell rec {
  name = "ds4-dev-env";

  packages = with pkgs; [
    # Core build tools
    gcc
    gnumake
    coreutils
    bison
    which
    bash

    # ROCm / HIP packages
    rocmPkgs.clr                    # HIP compiler, ROCm device libs, etc.
    rocmPkgs.hipblas                # HIP accelerated BLAS
    rocmPkgs.hipblas-common         # HIPBLAS common headers
    rocmPkgs.rocwmma                # ROCm Warp Matrix Multiply-Accumulate
    rocmPkgs.hipcub                 # HIP CUB wrappers
    rocmPkgs.rocprim                # ROCm C++ Parallel Primitives
    rocmPkgs.rocm-runtime           # ROCm Runtime (HSA)
    rocmPkgs.rocm-core              # ROCm Core (rocm-core)
    rocmPkgs.rocminfo               # rocminfo tool
    rocmPkgs.hipblaslt              # HIP BLAS LT (required for ROCM_LDLIBS)
    rocmPkgs.rocm-comgr             # ROCm Code Object Manager
    rocmPkgs.rocm-device-libs       # ROCm device libraries (bitcode)

    # CUDA packages
    cudaPkgs.cudatoolkit            # CUDA toolkit (nvcc, cudart, cublas, etc.)
    cudaPkgs.cuda_nvcc              # nvcc compiler (standalone)
    cudaPkgs.cuda_cudart            # CUDA runtime
    cudaPkgs.cuda_cccl              # CUDA C++ Core Compute Libraries (cub, etc.)
  ];

  # ROCm environment
  ROCM_PATH = "${rocmPkgs.clr}";
  ROCM_ARCH = "gfx1151";
  ROCM_VERSION = "${rocmPkgs.rocm-core.version}";
  HIP_PATH = "${rocmPkgs.clr}";
  HIP_PLATFORM = "amd";
  HSA_PATH = "${rocmPkgs.rocm-runtime}";
  DEVICE_LIB_PATH = "${rocmPkgs.rocm-device-libs}/amdgcn/bitcode";
  GPU_BACKEND = "rocm";

  ROCM_INCLUDE_PATH = "${rocmPkgs.clr}/include:"
    + "${rocmPkgs.hipblas}/include:"
    + "${rocmPkgs.hipblas-common}/include:"
    + "${rocmPkgs.rocwmma}/include:"
    + "${rocmPkgs.hipcub}/include:"
    + "${rocmPkgs.rocprim}/include";

  ROCM_CFLAGS = "-O3 -ffast-math -g -fno-finite-math-only -pthread"
    + " -D__HIP_PLATFORM_AMD__"
    + " -Wno-unused-command-line-argument"
    + " --offload-arch=${ROCM_ARCH}"
    + " -I${rocmPkgs.clr}/include"
    + " -I${rocmPkgs.rocwmma}/include"
    + " -I${rocmPkgs.hipcub}/include"
    + " -I${rocmPkgs.hipblas}/include"
    + " -I${rocmPkgs.hipblas-common}/include"
    + " -I${rocmPkgs.rocprim}/include"
    + " -I${rocmPkgs.hipblaslt}/include";

  ROCM_LDLIBS = "-lm -pthread"
    + " -L${rocmPkgs.hipblas}/lib"
    + " -L${rocmPkgs.hipblaslt}/lib"
    + " -lhipblas -lhipblaslt";

  # CUDA environment
  CUDA_HOME = "${cudaPkgs.cudatoolkit}";
  CUDA_PATH = "${cudaPkgs.cudatoolkit}";

  shellHook = ''
    echo "=== DS4 Development Environment ==="
    echo "ROCm target: make strix-halo (ROCM_ARCH=${ROCM_ARCH})"
    echo "CUDA target: make cuda-spark"
    echo "CPU target:  make cpu"
    echo ""
    echo "ROCm:"
    echo "  ROCM_PATH=$ROCM_PATH"
    echo "  ROCM_ARCH=$ROCM_ARCH"
    echo ""
    echo "CUDA:"
    echo "  CUDA_HOME=$CUDA_HOME"
    echo "  nvcc: $(which nvcc 2>/dev/null || echo 'not found')"
    echo ""
    echo "To build ROCm:  make strix-halo"
    echo "To build CUDA:  make cuda-spark"
    echo "To build CPU:   make cpu"
  '';
}
