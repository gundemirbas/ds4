__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}


extern "C" int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    (void)n_vocab;
    if (!out_hc || !model_map || weight_offset >= model_size) return 0;
    uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    uint32_t n = n_embd * n_hc;
    embed_token_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr, (const unsigned short *)wptr, token, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed token launch");
}

