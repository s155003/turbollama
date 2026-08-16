// turbollama: Armv8.2 SDOT (vdotq_s32) kernels.
// Compiled with -march=armv8.2-a+dotprod. Only entered after a runtime
// HWCAP_ASIMDDP check, so the binary still loads on Armv8.0 parts.
// SPDX-License-Identifier: MIT
#include "kernels.h"

#if defined(__aarch64__)
#include <arm_neon.h>

// Accumulate one group of `gs` int8 products into an int32 sum.
// gs is a multiple of 16 in every llama2.c checkpoint (typically 64), which
// lets us keep four independent accumulators and hide SDOT's ~4-cycle latency.
static inline int32_t tl_group_dot(const int8_t *a, const int8_t *b, int gs) {
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int k = 0;
    for (; k + 64 <= gs; k += 64) {
        a0 = vdotq_s32(a0, vld1q_s8(a + k),      vld1q_s8(b + k));
        a1 = vdotq_s32(a1, vld1q_s8(a + k + 16), vld1q_s8(b + k + 16));
        a2 = vdotq_s32(a2, vld1q_s8(a + k + 32), vld1q_s8(b + k + 32));
        a3 = vdotq_s32(a3, vld1q_s8(a + k + 48), vld1q_s8(b + k + 48));
    }
    for (; k + 16 <= gs; k += 16) {
        a0 = vdotq_s32(a0, vld1q_s8(a + k), vld1q_s8(b + k));
    }
    int32x4_t sum = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
    int32_t acc = vaddvq_s32(sum);
    // tail for any gs not a multiple of 16 (not hit by stock checkpoints)
    for (; k < gs; k++) acc += (int32_t) a[k] * (int32_t) b[k];
    return acc;
}

void tl_matmul_dot(float *xout, const TLQTensor *x, const TLQTensor *w,
                   int n, int d, int gs) {
    const int ng = n / gs;
    int i;
#if defined(_OPENMP)
    #pragma omp parallel for private(i)
#endif
    for (i = 0; i < d; i++) {
        const int8_t *wr = w->q + (size_t) i * n;
        const float  *ws = w->s + (size_t) i * ng;
        float val = 0.0f;
        for (int g = 0; g < ng; g++) {
            int32_t ival = tl_group_dot(x->q + g * gs, wr + g * gs, gs);
            val += (float) ival * ws[g] * x->s[g];
        }
        xout[i] = val;
    }
}

void tl_gemm_dot(float *out, const TLQTensor *x, const TLQTensor *w,
                 int m, int n, int d, int gs) {
    const int ng = n / gs;
    int i;
#if defined(_OPENMP)
    #pragma omp parallel for private(i)
#endif
    for (i = 0; i < d; i++) {
        const int8_t *wr = w->q + (size_t) i * n;
        const float  *ws = w->s + (size_t) i * ng;
        for (int r = 0; r < m; r++) {
            const int8_t *xr = x->q + (size_t) r * n;
            const float  *xs = x->s + (size_t) r * ng;
            float val = 0.0f;
            for (int g = 0; g < ng; g++) {
                int32_t ival = tl_group_dot(xr + g * gs, wr + g * gs, gs);
                val += (float) ival * ws[g] * xs[g];
            }
            out[(size_t) r * d + i] = val;
        }
    }
}

#else  // not aarch64: never selected by dispatch, but keep the symbols linkable

void tl_matmul_dot(float *xout, const TLQTensor *x, const TLQTensor *w,
                   int n, int d, int gs) {
    tl_matmul_scalar(xout, x, w, n, d, gs);
}
void tl_gemm_dot(float *out, const TLQTensor *x, const TLQTensor *w,
                 int m, int n, int d, int gs) {
    tl_gemm_scalar(out, x, w, m, n, d, gs);
}

#endif
