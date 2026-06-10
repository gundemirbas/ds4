

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
    const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
    acc = __dp4a(g0, *(const int32_t *)(q8 + 0), acc);
    acc = __dp4a(g1, *(const int32_t *)(q8 + 4), acc);
    return acc;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8) {
    int32_t sum = 0;
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid0], cuda_ksigns_iq2xs[sign0], q8, sum);
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid1], cuda_ksigns_iq2xs[sign1], q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = dev_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

__device__ static float dev_dot_iq2_xxs_q8_K_block(const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y) {
    const float d = dev_f16_to_f32(x->d) * y->d;
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = 0;
        sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8);
        q8 += 16;
        sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * d * (float)bsum;
}

__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8],
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q = q8[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
            sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
            sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
            sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
            sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
            sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
            sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
            sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
            bsum[p] += sumi * ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_dot_iq2_xxs_q8_K_block4(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[4] = {0, 0, 0, 0};
    const int8_t *q8[4] = {
        y0 ? y0->qs : NULL,
        y1 ? y1->qs : NULL,
        y2 ? y2->qs : NULL,
        y3 ? y3->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static DS4_CUDA_UNUSED void dev_dot_iq2_xxs_q8_K_block8(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

__device__ __forceinline__ static void dev_dot_q4_32_q8_K_block8(
        const uint8_t *qs,
        const cuda_block_q8_K *const ys[8],
        uint32_t n,
        uint32_t y_off,
        int shift,
        int32_t sums[8]) {
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < n) sums[p] = __dp4a(v, *(const int32_t *)(ys[p]->qs + y_off + i), sums[p]);
        }
    }
}

__device__ __forceinline__ static void dev_dot_q4_32_q8_K_block8_full(
        const uint8_t *qs,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t y_off,
        int shift,
        int32_t sums[8]) {
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sums[0] = __dp4a(v, *(const int32_t *)(y0->qs + y_off + i), sums[0]);
        sums[1] = __dp4a(v, *(const int32_t *)(y1->qs + y_off + i), sums[1]);
        sums[2] = __dp4a(v, *(const int32_t *)(y2->qs + y_off + i), sums[2]);
        sums[3] = __dp4a(v, *(const int32_t *)(y3->qs + y_off + i), sums[3]);
        sums[4] = __dp4a(v, *(const int32_t *)(y4->qs + y_off + i), sums[4]);
        sums[5] = __dp4a(v, *(const int32_t *)(y5->qs + y_off + i), sums[5]);
        sums[6] = __dp4a(v, *(const int32_t *)(y6->qs + y_off + i), sums[6]);
        sums[7] = __dp4a(v, *(const int32_t *)(y7->qs + y_off + i), sums[7]);
    }
}

__device__ static void dev_dot_q4_K_q8_K_block8(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t y_off = j * 32u;
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        int32_t dots[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        dev_dot_q4_32_q8_K_block8(x->qs + byte_off, ys, n, y_off, shift, dots);
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < n) {
                summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
                isum[p] += (int)sc * dots[p];
            }
        }
    }
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p < n) {
            const float yd = ys[p]->d;
            acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
        }
    }
}

__device__ static void dev_dot_q4_K_q8_K_block8_full(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t y_off = j * 32u;
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        int32_t dots[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        dev_dot_q4_32_q8_K_block8_full(
            x->qs + byte_off,
            y0, y1, y2, y3, y4, y5, y6, y7,
            y_off,
            shift,
            dots);
        const int ms0 = (int)m * (int)(y0->bsums[2u * j] + y0->bsums[2u * j + 1u]);
        const int ms1 = (int)m * (int)(y1->bsums[2u * j] + y1->bsums[2u * j + 1u]);
        const int ms2 = (int)m * (int)(y2->bsums[2u * j] + y2->bsums[2u * j + 1u]);
        const int ms3 = (int)m * (int)(y3->bsums[2u * j] + y3->bsums[2u * j + 1u]);
        const int ms4 = (int)m * (int)(y4->bsums[2u * j] + y4->bsums[2u * j + 1u]);
        const int ms5 = (int)m * (int)(y5->bsums[2u * j] + y5->bsums[2u * j + 1u]);
        const int ms6 = (int)m * (int)(y6->bsums[2u * j] + y6->bsums[2u * j + 1u]);
        const int ms7 = (int)m * (int)(y7->bsums[2u * j] + y7->bsums[2u * j + 1u]);
        summs[0] += ms0;
        summs[1] += ms1;
        summs[2] += ms2;
        summs[3] += ms3;
        summs[4] += ms4;
        summs[5] += ms5;
        summs[6] += ms6;
        summs[7] += ms7;
        isum[0] += (int)sc * dots[0];
        isum[1] += (int)sc * dots[1];
        isum[2] += (int)sc * dots[2];
        isum[3] += (int)sc * dots[3];
        isum[4] += (int)sc * dots[4];
        isum[5] += (int)sc * dots[5];
        isum[6] += (int)sc * dots[6];
        isum[7] += (int)sc * dots[7];
    }
    acc[0] += y0->d * xd * (float)isum[0] - y0->d * xmin * (float)summs[0];
    acc[1] += y1->d * xd * (float)isum[1] - y1->d * xmin * (float)summs[1];
    acc[2] += y2->d * xd * (float)isum[2] - y2->d * xmin * (float)summs[2];
    acc[3] += y3->d * xd * (float)isum[3] - y3->d * xmin * (float)summs[3];
    acc[4] += y4->d * xd * (float)isum[4] - y4->d * xmin * (float)summs[4];
    acc[5] += y5->d * xd * (float)isum[5] - y5->d * xmin * (float)summs[5];
    acc[6] += y6->d * xd * (float)isum[6] - y6->d * xmin * (float)summs[6];
    acc[7] += y7->d * xd * (float)isum[7] - y7->d * xmin * (float)summs[7];
}

__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    (void)lane16;
    return v;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    (void)lane8;
    return v;
}

