/* ------------------------------------------------------------------
 * FP8 KV read-path block counters (device symbols).
 * Incremented by attention decode kernels when reading FP8-quantized KV.
 * Host reads them via ds4_cuda_fp8_kv_read_path_blocks() and
 * ds4_cuda_fp8_kv_indexed_read_path_blocks().
 * ------------------------------------------------------------------ */
__device__ unsigned long long g_fp8_kv_read_path_blocks = 0ull;
__device__ unsigned long long g_fp8_kv_indexed_read_path_blocks = 0ull;

/* __constant__ decode table for E4M3FN FP8 -> float.
 * Populated by ds4_cuda_fp8_kv_decode_table_init() at init time.
 * Used by attention decode kernels to dequantize FP8 KV on the fly. */
__constant__ float dsv4_e4m3fn_decode_table[128];

/* Host-side stubs for cleanup functions referenced in ds4_cuda_runtime.cuh.
 * These will be fleshed out when the FP8 KV decode path is fully active. */
extern "C" void ds4_gpu_decode_scalars_cleanup(void) { }
extern "C" void ds4_gpu_decode_layer_scalars_cleanup(void) { }

/* ------------------------------------------------------------------
 * Opp C Phase 1A.3: packed FP8 KV-mirror row geometry.
 * Matches the DS4_OPP_C_FP8_* macros in ds4.c (no shared header today).
 *
 * Packed format (per compressed row):
 *   codes_base row: 448 uint8_t E4M3 codes + 64 float32 rotary tail = 704 B
 *   scale_base row: 7 float32 scales (one per 64-lane block)         = 28 B
 *
 * The rotary tail (last 64 dims) is stored as float32 verbatim.
 * ------------------------------------------------------------------ */
#define DS4_OPP_C_FP8_NOPE_DEV       448u    /* 512 - 64 */
#define DS4_OPP_C_FP8_BLOCKS_DEV     7u      /* 448 / 64  */
#define DS4_OPP_C_FP8_ROW_BYTES_DEV  704u    /* 448 codes + 64 FP32 tail */

/* Forward declaration for encode helper used by fp8_kv_quantize_kernel. */
__device__ static unsigned char dsv4_e4m3fn_encode_dev(float v);

/* Read one (row c, dim d) lane out of the packed FP8 mirror.
 * For d < 448: code is 1 byte (sign bit 7, magnitude index 0..126 in bits 0..6).
 *   value = sign * decode_table[idx] * scale[block]
 * For d >= 448: FP32 rotary tail copied verbatim at emit. */
__device__ static float fp8_kv_read(
        const unsigned char * __restrict__ codes_base,
        const float         * __restrict__ scale_base,
        uint32_t                          c,
        uint32_t                          d) {
    const uint64_t row_off = (uint64_t)c * DS4_OPP_C_FP8_ROW_BYTES_DEV;
    if (d < DS4_OPP_C_FP8_NOPE_DEV) {
        const unsigned char code = codes_base[row_off + d];
        const float scale = scale_base[(uint64_t)c * DS4_OPP_C_FP8_BLOCKS_DEV + (d >> 6)];
        const float mag   = dsv4_e4m3fn_decode_table[code & 0x7fu];
        return ((code & 0x80u) ? -mag : mag) * scale;
    }
    const float *tail = (const float *)(codes_base + row_off + DS4_OPP_C_FP8_NOPE_DEV);
    return tail[d - DS4_OPP_C_FP8_NOPE_DEV];
}

/* ------------------------------------------------------------------
 * FP8 KV quantize kernel: writes packed FP8 codes + scales + float tail.
 * Input:  float buffer x (n_tok rows, head_dim elements each)
 * Output: codes_base (uint8_t), scale_base (float)
 *
 * When both bases are NULL, falls back to in-place float quantization
 * (E4M3 precision rounding, no packing).
 * ------------------------------------------------------------------ */
__global__ static void fp8_kv_quantize_kernel(
        float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot,
        unsigned char * __restrict__ codes_base,
        float * __restrict__ scale_base) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    const uint64_t row_stride = (uint64_t)n_nope + (uint64_t)n_rot * sizeof(float);
    const uint64_t scale_stride = (uint64_t)(n_nope / 64u);
    float *xr = x + (uint64_t)row * head_dim;
    unsigned char *codes_row = codes_base
        ? codes_base + (uint64_t)row * row_stride
        : (unsigned char *)0;
    float *scale_row = scale_base
        ? scale_base + (uint64_t)row * scale_stride
        : (float *)0;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float clamp = fminf(448.0f, fmaxf(-448.0f, v / scale));
            xr[off + tid] = dsv4_e4m3fn_dequant_dev(clamp) * scale;
            if (codes_row) codes_row[off + tid] = dsv4_e4m3fn_encode_dev(clamp);
        }
        if (scale_row && tid == 0) scale_row[off / 64u] = scale;
        __syncthreads();
    }
    /* Copy FP32 rotary tail into the mirror row. */
    if (codes_row && tid < n_rot) {
        float *tail = (float *)(codes_row + (uint64_t)n_nope);
        tail[tid] = xr[(uint64_t)n_nope + tid];
    }
}

/* ------------------------------------------------------------------
 * R1 / Step-4c variant: writes exactly one row at base[row].
 * Used by per-layer decode body to emit a single compressed row.
 * ------------------------------------------------------------------ */
__global__ static void fp8_kv_quantize_row_kernel(
        float *base, uint32_t head_dim, uint32_t n_rot, uint32_t row,
        unsigned char * __restrict__ codes_base,
        float * __restrict__ scale_base) {
    /* Single row: treat base as the row pointer directly. */
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    float *xr = base;
    unsigned char *codes_row = codes_base
        ? codes_base + (uint64_t)row * ((uint64_t)n_nope + (uint64_t)n_rot * sizeof(float))
        : (unsigned char *)0;
    float *scale_row = scale_base
        ? scale_base + (uint64_t)row * ((uint64_t)n_nope / 64u)
        : (float *)0;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float clamp = fminf(448.0f, fmaxf(-448.0f, v / scale));
            xr[off + tid] = dsv4_e4m3fn_dequant_dev(clamp) * scale;
            if (codes_row) codes_row[off + tid] = dsv4_e4m3fn_encode_dev(clamp);
        }
        if (scale_row && tid == 0) scale_row[off / 64u] = scale;
        __syncthreads();
    }
    if (codes_row && tid < n_rot) {
        float *tail = (float *)(codes_row + (uint64_t)n_nope);
        tail[tid] = xr[(uint64_t)n_nope + tid];
    }
}

/* ------------------------------------------------------------------
 * Helper: encode a float32 value (clamped to E4M3 range) to its
 * 8-bit E4M3 code (sign in bit 7, magnitude index in bits 0..6).
 * ------------------------------------------------------------------ */
__device__ static unsigned char dsv4_e4m3fn_encode_dev(float v) {
    float ax = fabsf(v);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_decode_table[mid] <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_decode_table[best]);
        float nd = fabsf(ax - dsv4_e4m3fn_decode_table[best + 1]);
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return (unsigned char)(v < 0.0f ? ((uint32_t)best | 0x80u) : (uint32_t)best);
}

/* ------------------------------------------------------------------
 * Legacy in-place float quantize kernel (E4M3 precision rounding,
 * no packing). Kept for backward compat.
 * ------------------------------------------------------------------ */
__global__ static void fp8_kv_quantize_float_kernel(float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            xr[off + tid] = q;
        }
        __syncthreads();
    }
}

/* ------------------------------------------------------------------
 * Indexer FP4 Hadamard kernel (unchanged).
 * ------------------------------------------------------------------ */
__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
}

/* ------------------------------------------------------------------
 * Host-callable entry points.
 * ------------------------------------------------------------------ */

/* Legacy: in-place float quantization (E4M3 precision rounding). */
extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim || x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    fp8_kv_quantize_float_kernel<<<n_tok, 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}

/* Packed FP8 quantization: writes codes_base and scale_base from float x.
 * x is also modified in-place to the dequantized values. */
extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_pack_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot,
        ds4_gpu_tensor *codes_base, ds4_gpu_tensor *scale_base) {
    if (!x || !codes_base || !scale_base || n_rot > head_dim ||
        x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    /* Validate buffer sizes for packed format. */
    uint32_t n_nope = head_dim - n_rot;
    uint64_t codes_bytes = (uint64_t)n_tok * ((uint64_t)n_nope + (uint64_t)n_rot * sizeof(float));
    uint64_t scale_bytes = (uint64_t)n_tok * ((uint64_t)n_nope / 64u) * sizeof(float);
    if (codes_base->bytes < codes_bytes || scale_base->bytes < scale_bytes) return 0;
    dim3 grid(n_tok, 1, 1);
    fp8_kv_quantize_kernel<<<grid, 64>>>((float *)x->ptr, n_tok, head_dim, n_rot,
                                          (unsigned char *)codes_base->ptr,
                                          (float *)scale_base->ptr);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize_pack launch");
}

/* Single-row packed FP8 quantization variant. */
extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_row_tensor(
        ds4_gpu_tensor *x, uint32_t head_dim, uint32_t n_rot, uint32_t row,
        ds4_gpu_tensor *codes_base, ds4_gpu_tensor *scale_base) {
    if (!x || !codes_base || !scale_base || n_rot > head_dim ||
        x->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    fp8_kv_quantize_row_kernel<<<1, 64>>>((float *)x->ptr, head_dim, n_rot, row,
                                           (unsigned char *)codes_base->ptr,
                                           (float *)scale_base->ptr);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize_row launch");
}

extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<<<n_rows, 128>>>((float *)x->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4 launch");
}
