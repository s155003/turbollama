// turbollama kernel benchmark + correctness harness.
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

static TLQTensor alloc_qt(size_t elems, int gs) {
    TLQTensor t;
    t.q = (int8_t *) malloc(elems);
    t.s = (float *) malloc((elems / gs) * sizeof(float));
    for (size_t i = 0; i < elems; i++) t.q[i] = rand_i8();
    for (size_t i = 0; i < elems / gs; i++) t.s[i] = rand_scale();
    return t;
}
static void free_qt(TLQTensor *t) { free(t->q); free(t->s); }

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

// Emulated runs (QEMU, no Arm hardware) only need to prove correctness, and a
// full timing sweep under emulation takes minutes. TURBOLLAMA_QUICK=1 trims the
// trial count and the shape list; timings from such a run are meaningless and
// the harness says so.
#define MAX_TRIALS 16
static int TRIALS = 7;
static int quick = 0;

typedef void (*mv_fn)(float *, const TLQTensor *, const TLQTensor *, int, int, int);
typedef void (*mm_fn)(float *, const TLQTensor *, const TLQTensor *, int, int, int, int);

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
    // Precondition shared with llama2.c: the group size must divide the row
    // length, or the trailing n % gs values of each row are never visited.
    if (n % gs != 0) {
        printf("  matvec  n=%d skipped: group size %d does not divide n\n", n, gs);
        return;
    }
    TLQTensor w = alloc_qt((size_t) n * d, gs);
    TLQTensor x = alloc_qt((size_t) n, gs);
    float *ref = malloc(d * sizeof(float));
    float *got = malloc(d * sizeof(float));
    double t[MAX_TRIALS];

    printf("  matvec  W(%d x %d) @ x(%d)   [decode / tokens-per-second path]\n", d, n, n);

    for (int r = 0; r < TRIALS; r++) {
        double t0 = now_sec();
        tl_matmul_scalar(ref, &x, &w, n, d, gs);
        t[r] = now_sec() - t0;
    }
    stat_t sc = summarize(t, TRIALS);
    double ops = 2.0 * n * d;
    printf("    %-10s %8.3f ms   %6.2f GOP/s   (spread %.1f%%)\n",
           "scalar", sc.med * 1e3, ops / sc.min * 1e-9, sc.spread);

    if (tl_cpu_has_dotprod()) {
        for (int r = 0; r < TRIALS; r++) {
            double t0 = now_sec();
            tl_matmul_dot(got, &x, &w, n, d, gs);
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
    if (n % gs != 0) {
        printf("  gemm    n=%d skipped: group size %d does not divide n\n", n, gs);
        return;
    }
    TLQTensor w = alloc_qt((size_t) n * d, gs);
    TLQTensor x = alloc_qt((size_t) m * n, gs);
    float *ref = malloc((size_t) m * d * sizeof(float));
    float *got = malloc((size_t) m * d * sizeof(float));
    double t[MAX_TRIALS];

    printf("  gemm    W(%d x %d) @ X(%d x %d)^T   [prefill / time-to-first-token path]\n",
           d, n, m, n);

    for (int r = 0; r < TRIALS; r++) {
        double t0 = now_sec();
        tl_gemm_scalar(ref, &x, &w, m, n, d, gs);
        t[r] = now_sec() - t0;
    }
    stat_t sc = summarize(t, TRIALS);
    double ops = 2.0 * m * n * d;
    printf("    %-10s %8.3f ms   %6.2f GOP/s   (spread %.1f%%)\n",
           "scalar", sc.med * 1e3, ops / sc.min * 1e-9, sc.spread);

    if (tl_cpu_has_dotprod()) {
        for (int r = 0; r < TRIALS; r++) {
            double t0 = now_sec();
            tl_gemm_dot(got, &x, &w, m, n, d, gs);
            t[r] = now_sec() - t0;
        }
        stat_t dp = summarize(t, TRIALS);
        printf("    %-10s %8.3f ms   %6.2f GOP/s   (spread %.1f%%)   %.2fx vs scalar\n",
               "dotprod", dp.med * 1e3, ops / dp.min * 1e-9, dp.spread, sc.med / dp.med);
        check_exact("dotprod", ref, got, (size_t) m * d);

        if (tl_cpu_has_i8mm()) {
            memset(got, 0, (size_t) m * d * sizeof(float));
            for (int r = 0; r < TRIALS; r++) {
                double t0 = now_sec();
                tl_gemm_i8mm(got, &x, &w, m, n, d, gs);
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
    // 32 is the largest power of two dividing stories15M's dim of 288, which is
    // what tools/quantize.py settles on for that checkpoint.
    int gs = 32;
    if (argc > 1) gs = atoi(argv[1]);

    const char *q = getenv("TURBOLLAMA_QUICK");
    if (q && *q && strcmp(q, "0") != 0) { quick = 1; TRIALS = 1; }

    printf("turbollama kernel benchmark\n");
    if (quick) {
        printf("  MODE       : QUICK -- correctness only.\n");
        printf("               Timings below are NOT valid performance data\n");
        printf("               (1 trial, and typically run under emulation).\n");
    }
    printf("  group size : %d\n", gs);
    printf("  dotprod    : %s\n", tl_cpu_has_dotprod() ? "yes" : "no");
    printf("  i8mm       : %s\n", tl_cpu_has_i8mm() ? "yes" : "no");
    printf("  auto-ISA   : %s\n\n", tl_isa_name(tl_select_isa()));

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

    size_t nshapes = sizeof(shapes) / sizeof(shapes[0]);
    if (quick) nshapes = 2;  // emulation is slow; two shapes prove correctness

    for (size_t i = 0; i < nshapes; i++) {
        printf("== %s ==\n", shapes[i].tag);
        bench_matvec(shapes[i].n, shapes[i].d, gs);
        printf("\n");
    }

    for (size_t i = 0; i < nshapes; i++) {
        printf("== %s (batched) ==\n", shapes[i].tag);
        bench_gemm(16, shapes[i].n, shapes[i].d, gs);
        printf("\n");
    }

    // The i8mm kernel tiles 2x2 and hands odd leftovers to the SDOT path.
    // Those tail branches are the easiest place to get an off-by-one, so
    // exercise odd row counts on both sides explicitly.
    printf("== edge cases: odd batch / odd output rows ==\n");
    bench_gemm(5, 256, 288, gs);   // odd m
    bench_gemm(4, 256, 289, gs);   // odd d
    bench_gemm(3, 256, 287, gs);   // both odd
    bench_gemm(1, 256, 288, gs);   // degenerate batch of one
    printf("\n");

    if (failures) {
        printf("FAILED: %d correctness check(s) did not match scalar\n", failures);
        return 1;
    }
    printf("All kernels bit-exact against the scalar baseline.\n");
    return 0;
}
