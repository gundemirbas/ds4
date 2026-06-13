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

// Vendored llama.cpp fused-dequant-matmul kernels (MMQ/MMVQ)
// Provides ~2.8x prefill speedup on Blackwell via mul_mat_q templates.
#include "cuda/mmq/ds4_mmq.h"

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

/* ------------------------------------------------------------------
 * Opp C Phase 1A.3: FP8 KV decode table initialization.
 * Populates the __constant__ dsv4_e4m3fn_decode_table array on device.
 * Called once from ds4_gpu_init when DS4_CUDA_FP8_KV is enabled.
 * ------------------------------------------------------------------ */
extern "C" int ds4_cuda_fp8_kv_decode_table_init(void) {
    float table[128];
    for (int i = 0; i < 128; i++) {
        const int exp  = (i >> 3) & 15;
        const int mant = i & 7;
        table[i] = (exp == 0)
            ? (float)mant * 0.001953125f
            : (1.0f + (float)mant * 0.125f) * ldexpf(1.0f, exp - 7);
    }
    return cuda_ok(cudaMemcpyToSymbol(dsv4_e4m3fn_decode_table, table,
                                      sizeof(table), 0, cudaMemcpyHostToDevice),
                   "fp8 decode table init");
}

/* ------------------------------------------------------------------
 * Opp C Phase 1A.3: FP8 KV enabled check.  Reads DS4_CUDA_FP8_KV env var.
 * ------------------------------------------------------------------ */
static int ds4_cuda_fp8_kv_enabled_init = 0;
static int ds4_cuda_fp8_kv_enabled_val = 0;

extern "C" int ds4_cuda_fp8_kv_enabled(void) {
    if (!ds4_cuda_fp8_kv_enabled_init) {
        ds4_cuda_fp8_kv_enabled_init = 1;
        const char *s = getenv("DS4_CUDA_FP8_KV");
        if (s && *s &&
            (strcmp(s, "1") == 0 ||
             strcmp(s, "on") == 0 || strcmp(s, "ON") == 0 ||
             strcmp(s, "yes") == 0 || strcmp(s, "YES") == 0 ||
             strcmp(s, "true") == 0 || strcmp(s, "TRUE") == 0)) {
            ds4_cuda_fp8_kv_enabled_val = 1;
        }
    }
    return ds4_cuda_fp8_kv_enabled_val;
}

extern "C" int ds4_cuda_fp8_kv_debug_enabled(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        init = 1;
        const char *s = getenv("DS4_CUDA_FP8_KV_DEBUG");
        if (s && *s &&
            (strcmp(s, "1") == 0 ||
             strcmp(s, "on") == 0 || strcmp(s, "ON") == 0 ||
             strcmp(s, "yes") == 0 || strcmp(s, "YES") == 0 ||
             strcmp(s, "true") == 0 || strcmp(s, "TRUE") == 0)) {
            enabled = 1;
        }
    }
    return enabled;
}

extern "C" unsigned long long ds4_cuda_fp8_kv_read_path_blocks(void) {
    unsigned long long v = 0ull;
    if (cudaMemcpyFromSymbol(&v, g_fp8_kv_read_path_blocks,
                             sizeof(v), 0, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0ull;
    }
    return v;
}

extern "C" unsigned long long ds4_cuda_fp8_kv_indexed_read_path_blocks(void) {
    unsigned long long v = 0ull;
    if (cudaMemcpyFromSymbol(&v, g_fp8_kv_indexed_read_path_blocks,
                             sizeof(v), 0, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0ull;
    }
    return v;
}
