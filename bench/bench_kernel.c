// NeonForge kernel benchmark + correctness harness.
//
// Correctness note: every kernel accumulates each quantization group into an
// int32 before touching float, and integer addition is associative, so the
// SIMD paths are expected to be *bit-exact* against the scalar baseline --
// not merely within tolerance. The harness asserts that.
// SPDX-License-Identifier: MIT
#include "../src/kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint32_t rng = 12345;
static int8_t rand_i8(void) {
    rng = rng * 1664525u + 1013904223u;
    return (int8_t) ((rng >> 16) & 0xFF);
}
static float rand_scale(void) {
    rng = rng * 1664525u + 1013904223u;
    return 1e-3f + (float) ((rng >> 16) & 0x3FF) * 1e-6f;
}

static NFQTensor alloc_qt(size_t elems, int gs) {
    NFQTensor t;
    t.q = (int8_t *) malloc(elems);
    t.s = (float *) malloc((elems / gs) * sizeof(float));
    for (size_t i = 0; i < elems; i++) t.q[i] = rand_i8();
    for (size_t i = 0; i < elems / gs; i++) t.s[i] = rand_scale();
    return t;
}
static void free_qt(NFQTensor *t) { free(t->q); free(t->s); }

// Median of a small sample, plus min and relative spread.
static int cmpd(const void *a, const void *b) {
    double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

typedef struct { double med, min, spread; } stat_t;
static stat_t summarize(double *v, int n) {
    qsort(v, n, sizeof(double), cmpd);
    stat_t s;
    s.med = (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    s.min = v[0];
    s.spread = (v[n - 1] - v[0]) / s.med * 100.0;
    return s;
}

#define TRIALS 7

typedef void (*mv_fn)(float *, const NFQTensor *, const NFQTensor *, int, int, int);
typedef void (*mm_fn)(float *, const NFQTensor *, const NFQTensor *, int, int, int, int);

static int failures = 0;

static void check_exact(const char *label, const float *ref, const float *got, size_t n) {
    size_t bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (ref[i] != got[i]) {
            bad++;
            double d = fabs((double) ref[i] - (double) got[i]);
            double rel = d / (fabs((double) ref[i]) + 1e-30);
            if (rel > worst) worst = rel;
        }
    }
    if (bad == 0) {
        printf("    correctness %-8s : BIT-EXACT vs scalar (%zu values)\n", label, n);
    } else {
        printf("    correctness %-8s : MISMATCH %zu/%zu, worst rel err %.3e\n",
               label, bad, n, worst);
        failures++;
    }
}

static void bench_matvec(int n, int d, int gs) {
    NFQTensor w = alloc_qt((size_t) n * d, gs);
    NFQTensor x = alloc_qt((size_t) n, gs);
    float *ref = malloc(d * sizeof(float));
    float *got = malloc(d * sizeof(float));
    double t[TRIALS];

    printf("  matvec  W(%d x %d) @ x(%d)   [decode / tokens-per-second path]\n", d, n, n);

    for (int r = 0; r < TRIALS; r++) {
        double t0 = now_sec();
        nf_matmul_scalar(ref, &x, &w, n, d, gs);
        t[r] = now_sec() - t0;
    }
    stat_t sc = summarize(t, TRIALS);
    double ops = 2.0 * n * d;
    printf("    %-10s %8.3f ms   %6.2f GOP/s   (spread %.1f%%)\n",
           "scalar", sc.med * 1e3, ops / sc.min * 1e-9, sc.spread);

    if (nf_cpu_has_dotprod()) {
        for (int r = 0; r < TRIALS; r++) {
            double t0 = now_sec();
            nf_matmul_dot(got, &x, &w, n, d, gs);
            t[r] = now_sec() - t0;
        }
        stat_t dp = summarize(t, TRIALS);
        printf("    %-10s %8.3f ms   %6.2f GOP/s   (spread %.1f%%)   %.2fx vs scalar\n",
               "dotprod", dp.med * 1e3, ops / dp.min * 1e-9, dp.spread, sc.med / dp.med);
        check_exact("dotprod", ref, got, d);
    } else {
        printf("    %-10s unavailable on this CPU\n", "dotprod");
    }

    free_qt(&w); free_qt(&x); free(ref); free(got);
}

static void bench_gemm(int m, int n, int d, int gs) {
    NFQTensor w = alloc_qt((size_t) n * d, gs);
    NFQTensor x = alloc_qt((size_t) m * n, gs);
    float *ref = malloc((size_t) m * d * sizeof(float));
    float *got = malloc((size_t) m * d * sizeof(float));
    double t[TRIALS];

    printf("  gemm    W(%d x %d) @ X(%d x %d)^T   [prefill / time-to-first-token path]\n",
           d, n, m, n);

    for (int r = 0; r < TRIALS; r++) {
        double t0 = now_sec();
        nf_gemm_scalar(ref, &x, &w, m, n, d, gs);
        t[r] = now_sec() - t0;
    }
    stat_t sc = summarize(t, TRIALS);
    double ops = 2.0 * m * n * d;
    printf("    %-10s %8.3f ms   %6.2f GOP/s   (spread %.1f%%)\n",
           "scalar", sc.med * 1e3, ops / sc.min * 1e-9, sc.spread);

    if (nf_cpu_has_dotprod()) {
        for (int r = 0; r < TRIALS; r++) {
            double t0 = now_sec();
            nf_gemm_dot(got, &x, &w, m, n, d, gs);
            t[r] = now_sec() - t0;
        }
        stat_t dp = summarize(t, TRIALS);
        printf("    %-10s %8.3f ms   %6.2f GOP/s   (spread %.1f%%)   %.2fx vs scalar\n",
               "dotprod", dp.med * 1e3, ops / dp.min * 1e-9, dp.spread, sc.med / dp.med);
        check_exact("dotprod", ref, got, (size_t) m * d);

        if (nf_cpu_has_i8mm()) {
            memset(got, 0, (size_t) m * d * sizeof(float));
            for (int r = 0; r < TRIALS; r++) {
                double t0 = now_sec();
                nf_gemm_i8mm(got, &x, &w, m, n, d, gs);
                t[r] = now_sec() - t0;
            }
            stat_t mm = summarize(t, TRIALS);
            printf("    %-10s %8.3f ms   %6.2f GOP/s   (spread %.1f%%)   %.2fx vs scalar, %.2fx vs dotprod\n",
                   "i8mm", mm.med * 1e3, ops / mm.min * 1e-9, mm.spread,
                   sc.med / mm.med, dp.med / mm.med);
            check_exact("i8mm", ref, got, (size_t) m * d);
        } else {
            printf("    %-10s unavailable on this CPU\n", "i8mm");
        }
    }

    free_qt(&w); free_qt(&x); free(ref); free(got);
}

int main(int argc, char **argv) {
    int gs = 64;
    if (argc > 1) gs = atoi(argv[1]);

    printf("NeonForge kernel benchmark\n");
    printf("  group size : %d\n", gs);
    printf("  dotprod    : %s\n", nf_cpu_has_dotprod() ? "yes" : "no");
    printf("  i8mm       : %s\n", nf_cpu_has_i8mm() ? "yes" : "no");
    printf("  auto-ISA   : %s\n\n", nf_isa_name(nf_select_isa()));

    // Shapes taken straight from the TinyLlama checkpoints we run end-to-end:
    //   stories15M : dim=288  hidden=768
    //   stories110M: dim=768  hidden=2048
    struct { int n, d; const char *tag; } shapes[] = {
        { 288,  288,  "15M  attn (dim x dim)" },
        { 288,  768,  "15M  ffn  up" },
        { 768,  288,  "15M  ffn  down" },
        { 768,  768,  "110M attn (dim x dim)" },
        { 768,  2048, "110M ffn  up" },
        { 2048, 768,  "110M ffn  down" },
    };

    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        printf("== %s ==\n", shapes[i].tag);
        bench_matvec(shapes[i].n, shapes[i].d, gs);
        printf("\n");
    }

    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        printf("== %s (batched) ==\n", shapes[i].tag);
        bench_gemm(16, shapes[i].n, shapes[i].d, gs);
        printf("\n");
    }

    if (failures) {
        printf("FAILED: %d correctness check(s) did not match scalar\n", failures);
        return 1;
    }
    printf("All kernels bit-exact against the scalar baseline.\n");
    return 0;
}
