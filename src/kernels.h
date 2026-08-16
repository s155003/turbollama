// turbollama: int8 quantized matmul kernels for Arm64
// SPDX-License-Identifier: MIT
#ifndef TURBOLLAMA_KERNELS_H
#define TURBOLLAMA_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A group-quantized tensor, layout-compatible with llama2.c's QuantizedTensor.
//   q: int8 values, row-major
//   s: one fp32 scale per `gs` consecutive values
typedef struct {
    int8_t *q;
    float  *s;
} TLQTensor;

// ---------------------------------------------------------------------------
// Matrix-vector: W(d,n) @ x(n,) -> xout(d,)   [the decode / token-generation path]
// ---------------------------------------------------------------------------
void tl_matmul_scalar(float *xout, const TLQTensor *x, const TLQTensor *w,
                      int n, int d, int gs);
void tl_matmul_dot(float *xout, const TLQTensor *x, const TLQTensor *w,
                   int n, int d, int gs);

// ---------------------------------------------------------------------------
// Matrix-matrix: W(d,n) @ X(m,n)^T -> out(m,d)  [the prefill / TTFT path]
//   X->q is m*n int8 row-major, X->s is m*(n/gs) fp32 row-major.
//   out is m*d fp32 row-major.
// ---------------------------------------------------------------------------
void tl_gemm_scalar(float *out, const TLQTensor *x, const TLQTensor *w,
                    int m, int n, int d, int gs);
void tl_gemm_dot(float *out, const TLQTensor *x, const TLQTensor *w,
                 int m, int n, int d, int gs);
void tl_gemm_i8mm(float *out, const TLQTensor *x, const TLQTensor *w,
                  int m, int n, int d, int gs);

// ---------------------------------------------------------------------------
// Runtime dispatch (AT_HWCAP based). Safe to call on any architecture.
// ---------------------------------------------------------------------------
typedef enum {
    TL_ISA_SCALAR = 0,
    TL_ISA_DOTPROD = 1,
    TL_ISA_I8MM = 2,
} tl_isa_t;

// Bitmask of what this CPU actually supports.
int tl_cpu_has_dotprod(void);
int tl_cpu_has_i8mm(void);

// Best available kernel for each shape. Honours the TURBOLLAMA_ISA env var
// ("scalar", "dotprod", "i8mm") so benchmarks can force a specific path.
tl_isa_t tl_select_isa(void);
const char *tl_isa_name(tl_isa_t isa);

// Dispatching entry points used by the model runner.
void tl_matmul(float *xout, const TLQTensor *x, const TLQTensor *w,
               int n, int d, int gs);
void tl_gemm(float *out, const TLQTensor *x, const TLQTensor *w,
             int m, int n, int d, int gs);

#ifdef __cplusplus
}
#endif
#endif // TURBOLLAMA_KERNELS_H
