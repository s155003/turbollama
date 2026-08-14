// NeonForge: Armv8.6 I8MM (vmmlaq_s32) kernel.
//
// SMMLA is a matrix-matrix instruction: one issue computes a full 2x2 int32
// tile from two 2x8 int8 operands, i.e. 4 dot products of length 8 = 64 MACs,
// versus 16 MACs for SDOT. It therefore only pays off when there are at least
// two activation rows to feed it -- prefill / prompt processing, not
// single-token decode. That is exactly why NeonForge routes prefill here and
// decode to SDOT, rather than pretending one kernel wins everywhere.
//
// Compiled with -march=armv8.2-a+i8mm, entered only after a runtime HWCAP2_I8MM
// check.
// SPDX-License-Identifier: MIT
#include "kernels.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
#include <arm_neon.h>

void nf_gemm_i8mm(float *out, const NFQTensor *x, const NFQTensor *w,
                  int m, int n, int d, int gs) {
    const int ng = n / gs;
    const int mpair = m & ~1;   // largest even count of activation rows
    const int dpair = d & ~1;   // largest even count of weight rows

    int i;
#if defined(_OPENMP)
    #pragma omp parallel for private(i)
#endif
    for (i = 0; i < dpair; i += 2) {
        const int8_t *w0 = w->q + (size_t)  i      * n;
        const int8_t *w1 = w->q + (size_t) (i + 1) * n;
        const float  *s0 = w->s + (size_t)  i      * ng;
        const float  *s1 = w->s + (size_t) (i + 1) * ng;

        for (int r = 0; r < mpair; r += 2) {
            const int8_t *x0 = x->q + (size_t)  r      * n;
            const int8_t *x1 = x->q + (size_t) (r + 1) * n;
            const float  *t0 = x->s + (size_t)  r      * ng;
            const float  *t1 = x->s + (size_t) (r + 1) * ng;

            float v00 = 0.f, v01 = 0.f, v10 = 0.f, v11 = 0.f;

            for (int g = 0; g < ng; g++) {
                const int8_t *xa = x0 + g * gs, *xb = x1 + g * gs;
                const int8_t *wa = w0 + g * gs, *wb = w1 + g * gs;

                int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
                int k = 0;
                for (; k + 16 <= gs; k += 16) {
                    acc0 = vmmlaq_s32(acc0,
                        vcombine_s8(vld1_s8(xa + k), vld1_s8(xb + k)),
                        vcombine_s8(vld1_s8(wa + k), vld1_s8(wb + k)));
                    acc1 = vmmlaq_s32(acc1,
                        vcombine_s8(vld1_s8(xa + k + 8), vld1_s8(xb + k + 8)),
                        vcombine_s8(vld1_s8(wa + k + 8), vld1_s8(wb + k + 8)));
                }
                for (; k + 8 <= gs; k += 8) {
                    acc0 = vmmlaq_s32(acc0,
                        vcombine_s8(vld1_s8(xa + k), vld1_s8(xb + k)),
                        vcombine_s8(vld1_s8(wa + k), vld1_s8(wb + k)));
                }
                int32x4_t acc = vaddq_s32(acc0, acc1);

                // acc = [ x0.w0, x0.w1, x1.w0, x1.w1 ]
                v00 += (float) vgetq_lane_s32(acc, 0) * t0[g] * s0[g];
                v01 += (float) vgetq_lane_s32(acc, 1) * t0[g] * s1[g];
                v10 += (float) vgetq_lane_s32(acc, 2) * t1[g] * s0[g];
                v11 += (float) vgetq_lane_s32(acc, 3) * t1[g] * s1[g];
            }

            out[(size_t)  r      * d +  i     ] = v00;
            out[(size_t)  r      * d + (i + 1)] = v01;
            out[(size_t) (r + 1) * d +  i     ] = v10;
            out[(size_t) (r + 1) * d + (i + 1)] = v11;
        }
    }

    // Odd leftovers (odd m and/or odd d) fall back to the SDOT path.
    if (mpair != m) {
        NFQTensor xt = { x->q + (size_t) mpair * n, x->s + (size_t) mpair * ng };
        float *tail = out + (size_t) mpair * d;
        nf_gemm_dot(tail, &xt, w, m - mpair, n, d, gs);
    }
    if (dpair != d) {
        for (int r = 0; r < mpair; r++) {
            NFQTensor xr = { x->q + (size_t) r * n, x->s + (size_t) r * ng };
            NFQTensor wt = { w->q + (size_t) dpair * n, w->s + (size_t) dpair * ng };
            nf_matmul_dot(out + (size_t) r * d + dpair, &xr, &wt, n, d - dpair, gs);
        }
    }
}

#else  // no i8mm at compile time: never selected by dispatch

void nf_gemm_i8mm(float *out, const NFQTensor *x, const NFQTensor *w,
                  int m, int n, int d, int gs) {
    nf_gemm_dot(out, x, w, m, n, d, gs);
}

#endif
