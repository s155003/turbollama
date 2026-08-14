#!/usr/bin/env bash
# End-to-end model benchmark: same binary, same weights, same seed, same
# temperature -- only the kernel changes. Because temperature is 0 the sampler
# is deterministic, so a correct SIMD kernel must reproduce the baseline's
# output text byte for byte. We assert that, then report the speedup.
#
# SPDX-License-Identifier: MIT
set -euo pipefail

BIN=${BIN:-./build/runq_nf}
MODEL=${MODEL:-stories15M_q80.bin}
TOKENIZER=${TOKENIZER:-tokenizer.bin}
STEPS=${STEPS:-256}
PROMPT=${PROMPT:-"One day, Lily met a robot"}
REPS=${REPS:-5}
OUT=${OUT:-results}

mkdir -p "$OUT"

have_isa() {
  # the binary prints the ISA it selected; ask it what it can actually do
  NEONFORGE_ISA=$1 "$BIN" "$MODEL" -z "$TOKENIZER" -t 0 -n 2 -i "hi" 2>&1 >/dev/null \
    | grep -q "^isa: $1$"
}

run_one() {  # $1=isa -> prints "ttft_ms tok_s"
  NEONFORGE_ISA=$1 "$BIN" "$MODEL" -z "$TOKENIZER" -t 0 -s 42 \
      -n "$STEPS" -i "$PROMPT" \
      >"$OUT/text_$1.txt" 2>"$OUT/stderr_$1.txt"
  local ttft toks
  ttft=$(grep -oP 'time to first token \(ms\): \K[0-9.]+' "$OUT/stderr_$1.txt")
  toks=$(grep -oP 'decode tok/s: \K[0-9.]+' "$OUT/stderr_$1.txt")
  echo "$ttft $toks"
}

best_of() {  # $1=isa -> prints "best_ttft best_toks"
  local bt="" bs=""
  for _ in $(seq 1 "$REPS"); do
    read -r t s < <(run_one "$1")
    # keep the fastest observed: least noise-contaminated on a shared vCPU
    if [ -z "$bs" ] || awk "BEGIN{exit !($s > $bs)}"; then bs=$s; fi
    if [ -z "$bt" ] || awk "BEGIN{exit !($t < $bt)}"; then bt=$t; fi
  done
  echo "$bt $bs"
}

ISAS=(scalar)
if have_isa dotprod; then ISAS+=(dotprod); fi

echo "NeonForge end-to-end: $MODEL, $STEPS steps, best of $REPS"
echo

declare -A TTFT TOKS
for isa in "${ISAS[@]}"; do
  read -r t s < <(best_of "$isa")
  TTFT[$isa]=$t
  TOKS[$isa]=$s
  echo "  $isa: ttft ${t} ms, decode ${s} tok/s"
done
echo

# --- correctness: identical text, or the speedup means nothing ---
STATUS=0
for isa in "${ISAS[@]}"; do
  [ "$isa" = scalar ] && continue
  if cmp -s "$OUT/text_scalar.txt" "$OUT/text_$isa.txt"; then
    echo "  output check $isa: IDENTICAL to scalar baseline"
  else
    echo "  output check $isa: DIFFERS from scalar baseline"
    diff "$OUT/text_scalar.txt" "$OUT/text_$isa.txt" | head -20 || true
    STATUS=1
  fi
done
echo

# --- markdown table for the job summary / README ---
{
  echo "| kernel | time to first token | decode tok/s | speedup |"
  echo "|---|---|---|---|"
  for isa in "${ISAS[@]}"; do
    sp=$(awk "BEGIN{printf \"%.2fx\", ${TOKS[$isa]}/${TOKS[scalar]}}")
    echo "| \`$isa\` | ${TTFT[$isa]} ms | ${TOKS[$isa]} | $sp |"
  done
} | tee "$OUT/e2e.md"

exit $STATUS
