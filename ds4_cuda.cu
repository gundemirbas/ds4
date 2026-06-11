// DS4 CUDA backend -- organized into modular .cuh files under /cuda/
//
// This file is the single translation unit that includes all CUDA kernels
// and host dispatch functions.  The split mirrors the ROCm backend layout
// under /rocm/ for consistency.
//
// Include order matters: kernels must be defined before any host function
// that launches them.  Files are grouped accordingly.

#include "cuda/ds4_cuda_common.cuh"
#include "ds4_iq2_tables_cuda.inc"
#include "cuda/ds4_cuda_runtime.cuh"

// --- Device kernels (and their immediate launch helpers) ---
// These come first so their symbols are visible to later host-dispatch files.
#include "cuda/ds4_cuda_embedding.cuh"
#include "cuda/ds4_cuda_matmul.cuh"
#include "cuda/ds4_cuda_norm_rope.cuh"
#include "cuda/ds4_cuda_fp8_kv.cuh"
#include "cuda/ds4_cuda_attention.cuh"
#include "cuda/ds4_cuda_hc.cuh"
#include "cuda/ds4_cuda_compressor.cuh"
#include "cuda/ds4_cuda_router.cuh"
#include "cuda/ds4_cuda_misc.cuh"
#include "cuda/ds4_cuda_indexer.cuh"
#include "cuda/ds4_cuda_q8_K.cuh"
#include "cuda/ds4_cuda_moe.cuh"

// --- Host dispatch functions (may reference kernels from above) ---
#include "cuda/ds4_cuda_attention_launch.cuh"
#include "cuda/ds4_cuda_embedding_launch.cuh"
#include "cuda/ds4_cuda_moe_launch.cuh"
#include "cuda/ds4_cuda_hc_output_launch.cuh"
