#!/usr/bin/env bash
# Verify the Arm kernels WITHOUT owning an Arm machine.
#
# Cross-compiles with aarch64-linux-gnu-gcc and executes under QEMU user-mode
# emulation with -cpu max, which exposes both dotprod and i8mm. This proves
# correctness only -- emulated timings are meaningless, so the harness runs in
# TURBOLLAMA_QUICK mode and labels them as such.
#
#   sudo apt install gcc-aarch64-linux-gnu qemu-user
#   ./scripts/verify_qemu.sh
#
# SPDX-License-Identifier: MIT
set -euo pipefail

CROSS=${CROSS:-aarch64-linux-gnu-gcc}
QEMU=${QEMU:-qemu-aarch64}
SYSROOT=${SYSROOT:-/usr/aarch64-linux-gnu}
GS=${GS:-32}

command -v "$CROSS" >/dev/null || { echo "missing $CROSS"; exit 1; }
command -v "$QEMU"  >/dev/null || { echo "missing $QEMU";  exit 1; }

echo "== cross-compiling for aarch64 =="
make clean >/dev/null 2>&1 || true
make ARCH=aarch64 CC="$CROSS" bench

echo
echo "== confirming the SIMD instructions are actually present =="
OBJDUMP=${OBJDUMP:-aarch64-linux-gnu-objdump}
if command -v "$OBJDUMP" >/dev/null; then
  echo "  sdot  in kernels_dot.o  : $($OBJDUMP -d build/src/kernels_dot.o  | grep -c '\bsdot\b'  || true)"
  echo "  smmla in kernels_i8mm.o : $($OBJDUMP -d build/src/kernels_i8mm.o | grep -c '\bsmmla\b' || true)"
  echo "  sdot/smmla in scalar    : $($OBJDUMP -d build/src/kernels_scalar.o | grep -cE '\b(sdot|smmla)\b' || true)  (expected 0)"
fi

echo
echo "== running under QEMU (-cpu max: dotprod + i8mm available) =="
TURBOLLAMA_QUICK=1 "$QEMU" -L "$SYSROOT" -cpu max ./build/bench_kernel "$GS"
