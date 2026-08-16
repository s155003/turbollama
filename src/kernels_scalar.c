// turbollama: portable scalar baseline. Compiled with plain -O3 (no -march),
// so this is exactly what a stock build of llama2.c/runq.c gives you.
// SPDX-License-Identifier: MIT
#include "kernels.h"

void tl_matmul_scalar(float *xout, const TLQTensor *x, const TLQTensor *w,
                      int n, int d, int gs) {
    int i;
#if defined(_OPENMP)
    #pragma omp parallel for private(i)
#endif
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int in = i * n;
        for (int j = 0; j <= n - gs; j += gs) {
            int32_t ival = 0;
            for (int k = 0; k < gs; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / gs] * x->s[j / gs];
        }
        xout[i] = val;
    }
}

void tl_gemm_scalar(float *out, const TLQTensor *x, const TLQTensor *w,
                    int m, int n, int d, int gs) {
    int i;
#if defined(_OPENMP)
    #pragma omp parallel for private(i)
#endif
    for (i = 0; i < d; i++) {
        int in = i * n;
        for (int r = 0; r < m; r++) {
            const int8_t *xr = x->q + (size_t) r * n;
            const float  *xs = x->s + (size_t) r * (n / gs);
            float val = 0.0f;
            for (int j = 0; j <= n - gs; j += gs) {
                int32_t ival = 0;
                for (int k = 0; k < gs; k++) {
                    ival += ((int32_t) xr[j + k]) * ((int32_t) w->q[in + j + k]);
                }
                val += ((float) ival) * w->s[(in + j) / gs] * xs[j / gs];
            }
            out[(size_t) r * d + i] = val;
        }
    }
}
