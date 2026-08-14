// NeonForge: int8 quantized matmul kernels for Arm64
// SPDX-License-Identifier: MIT
#ifndef NEONFORGE_KERNELS_H
#define NEONFORGE_KERNELS_H

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
} NFQTensor;

// ---------------------------------------------------------------------------
// Matrix-vector: W(d,n) @ x(n,) -> xout(d,)   [the decode / token-generation path]
// ---------------------------------------------------------------------------
void nf_matmul_scalar(float *xout, const NFQTensor *x, const NFQTensor *w,
                      int n, int d, int gs);
void nf_matmul_dot(float *xout, const NFQTensor *x, const NFQTensor *w,
                   int n, int d, int gs);

// ---------------------------------------------------------------------------
// Matrix-matrix: W(d,n) @ X(m,n)^T -> out(m,d)  [the prefill / TTFT path]
//   X->q is m*n int8 row-major, X->s is m*(n/gs) fp32 row-major.
//   out is m*d fp32 row-major.
// ---------------------------------------------------------------------------
void nf_gemm_scalar(float *out, const NFQTensor *x, const NFQTensor *w,
                    int m, int n, int d, int gs);
void nf_gemm_dot(float *out, const NFQTensor *x, const NFQTensor *w,
                 int m, int n, int d, int gs);
void nf_gemm_i8mm(float *out, const NFQTensor *x, const NFQTensor *w,
                  int m, int n, int d, int gs);

// ---------------------------------------------------------------------------
// Runtime dispatch (AT_HWCAP based). Safe to call on any architecture.
// ---------------------------------------------------------------------------
typedef enum {
    NF_ISA_SCALAR = 0,
    NF_ISA_DOTPROD = 1,
    NF_ISA_I8MM = 2,
} nf_isa_t;

// Bitmask of what this CPU actually supports.
int nf_cpu_has_dotprod(void);
int nf_cpu_has_i8mm(void);

// Best available kernel for each shape. Honours the NEONFORGE_ISA env var
// ("scalar", "dotprod", "i8mm") so benchmarks can force a specific path.
nf_isa_t nf_select_isa(void);
const char *nf_isa_name(nf_isa_t isa);

// Dispatching entry points used by the model runner.
void nf_matmul(float *xout, const NFQTensor *x, const NFQTensor *w,
               int n, int d, int gs);
void nf_gemm(float *out, const NFQTensor *x, const NFQTensor *w,
             int m, int n, int d, int gs);

#ifdef __cplusplus
}
#endif
#endif // NEONFORGE_KERNELS_H
