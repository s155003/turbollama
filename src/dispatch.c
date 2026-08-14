// NeonForge: runtime ISA detection + kernel dispatch.
// One binary, built once, picks the best path for whatever Arm core it lands
// on. Set NEONFORGE_ISA=scalar|dotprod|i8mm to pin a path (used by benchmarks).
// SPDX-License-Identifier: MIT
#include "kernels.h"
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1u << 20)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1u << 13)
#endif
#endif

int nf_cpu_has_dotprod(void) {
#if defined(__aarch64__) && defined(__linux__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(__aarch64__) && defined(__APPLE__)
    return 1;  // every Apple Silicon part has dotprod
#else
    return 0;
#endif
}

int nf_cpu_has_i8mm(void) {
#if defined(__aarch64__) && defined(__linux__)
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
    return 0;
#endif
}

nf_isa_t nf_select_isa(void) {
    const char *force = getenv("NEONFORGE_ISA");
    if (force && *force) {
        if (!strcmp(force, "scalar"))  return NF_ISA_SCALAR;
        if (!strcmp(force, "dotprod")) return NF_ISA_DOTPROD;
        if (!strcmp(force, "i8mm"))    return NF_ISA_I8MM;
    }
    if (nf_cpu_has_i8mm())    return NF_ISA_I8MM;
    if (nf_cpu_has_dotprod()) return NF_ISA_DOTPROD;
    return NF_ISA_SCALAR;
}

const char *nf_isa_name(nf_isa_t isa) {
    switch (isa) {
        case NF_ISA_I8MM:    return "i8mm";
        case NF_ISA_DOTPROD: return "dotprod";
        default:             return "scalar";
    }
}

void nf_matmul(float *xout, const NFQTensor *x, const NFQTensor *w,
               int n, int d, int gs) {
    // Decode is a matrix-vector product; SMMLA has no second row to fill, so
    // SDOT is the best available kernel even on an i8mm-capable core.
    switch (nf_select_isa()) {
        case NF_ISA_I8MM:
        case NF_ISA_DOTPROD: nf_matmul_dot(xout, x, w, n, d, gs); break;
        default:             nf_matmul_scalar(xout, x, w, n, d, gs); break;
    }
}

void nf_gemm(float *out, const NFQTensor *x, const NFQTensor *w,
             int m, int n, int d, int gs) {
    nf_isa_t isa = nf_select_isa();
    if (m == 1) {  // degenerate batch: matvec rules apply
        if (isa == NF_ISA_SCALAR) nf_gemm_scalar(out, x, w, 1, n, d, gs);
        else                      nf_gemm_dot(out, x, w, 1, n, d, gs);
        return;
    }
    switch (isa) {
        case NF_ISA_I8MM:    nf_gemm_i8mm(out, x, w, m, n, d, gs); break;
        case NF_ISA_DOTPROD: nf_gemm_dot(out, x, w, m, n, d, gs); break;
        default:             nf_gemm_scalar(out, x, w, m, n, d, gs); break;
    }
}
