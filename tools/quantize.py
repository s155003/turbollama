#!/usr/bin/env python3
"""
Convert a legacy fp32 llama2.c checkpoint (.bin) into the int8 "version 2"
format that runq.c / runq_nf.c consume.

Upstream's export.py does this too, but only from a PyTorch .pt checkpoint --
which drags a ~2.5 GB torch install into CI just to divide some floats. This
reads the published fp32 .bin directly and needs nothing but numpy, so the
benchmark job installs in seconds instead of minutes.

Usage:
    python3 tools/quantize.py stories15M.bin stories15M_q80.bin [--group-size 64]

SPDX-License-Identifier: MIT
"""
import argparse
import struct
import sys

import numpy as np

HEADER_BYTES = 256
MAGIC = 0x616B3432  # "ak42"
VERSION = 2


def read_legacy(path):
    """Parse the legacy (version 0) llama2.c export."""
    with open(path, "rb") as f:
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = \
            struct.unpack("7i", f.read(28))

        # legacy quirk: a negative vocab_size flags an unshared classifier
        shared_classifier = vocab_size > 0
        vocab_size = abs(vocab_size)
        head_size = dim // n_heads

        def rd(count):
            buf = f.read(count * 4)
            if len(buf) != count * 4:
                sys.exit(f"error: {path} truncated (wanted {count} floats)")
            return np.frombuffer(buf, dtype=np.float32)

        # order is fixed by legacy_export() upstream
        tok = rd(vocab_size * dim)
        rms_att = rd(n_layers * dim)
        wq = rd(n_layers * dim * n_heads * head_size)
        wk = rd(n_layers * dim * n_kv_heads * head_size)
        wv = rd(n_layers * dim * n_kv_heads * head_size)
        wo = rd(n_layers * n_heads * head_size * dim)
        rms_ffn = rd(n_layers * dim)
        w1 = rd(n_layers * dim * hidden_dim)
        w2 = rd(n_layers * hidden_dim * dim)
        w3 = rd(n_layers * dim * hidden_dim)
        rms_final = rd(dim)
        rd(seq_len * head_size // 2)  # freq_cis_real, recomputed at runtime
        rd(seq_len * head_size // 2)  # freq_cis_imag, recomputed at runtime
        wcls = tok if shared_classifier else rd(vocab_size * dim)

    cfg = dict(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers,
               n_heads=n_heads, n_kv_heads=n_kv_heads, vocab_size=vocab_size,
               seq_len=seq_len, shared_classifier=shared_classifier)
    w = dict(tok=tok, rms_att=rms_att, wq=wq, wk=wk, wv=wv, wo=wo,
             rms_ffn=rms_ffn, w1=w1, w2=w2, w3=w3, rms_final=rms_final,
             wcls=wcls)
    return cfg, w


def quantize_q80(x, gs):
    """Symmetric int8, one fp32 scale per `gs` values. Mirrors runq.c exactly."""
    assert x.size % gs == 0, f"tensor of {x.size} not divisible by group size {gs}"
    g = x.reshape(-1, gs).astype(np.float32)
    wmax = np.abs(g).max(axis=1)
    scale = wmax / 127.0
    # an all-zero group would divide by zero; any nonzero scale reproduces it
    safe = np.where(scale == 0.0, 1.0, scale)
    q = np.rint(g / safe[:, None]).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int8)
    err = float(np.abs(q.astype(np.float32) * safe[:, None] - g).max())
    return q.reshape(-1), scale.astype(np.float32), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    cfg, w = read_legacy(args.src)
    gs = args.group_size
    n_layers = cfg["n_layers"]

    # Upstream export.py only checks that each tensor's *total* element count
    # divides by the group size. That is not sufficient. The matmul walks each
    # row with `for (j = 0; j <= n - GS; j += GS)`, so if the group size does
    # not divide the row length n, the trailing n % GS values of every row are
    # silently dropped -- and the scale index (i*n + j)/GS drifts out of
    # alignment with the row it belongs to.
    #
    # stories15M trips exactly this: dim=288 with the default group size 64
    # leaves 288 % 64 = 32 values per row unread. So we additionally require
    # the group size to divide every `n` that a matmul is actually called with:
    # dim (wq/wk/wv/wo/w1/w3/wcls) and hidden_dim (w2).
    per_layer = [w["wq"].size // n_layers, w["wk"].size // n_layers,
                 w["wv"].size // n_layers, w["wo"].size // n_layers,
                 w["w1"].size // n_layers, w["w2"].size // n_layers,
                 w["w3"].size // n_layers, w["tok"].size, w["wcls"].size]
    row_lengths = [cfg["dim"], cfg["hidden_dim"]]
    while any(s % gs != 0 for s in per_layer) or any(r % gs != 0 for r in row_lengths):
        gs //= 2
        if gs < 1:
            sys.exit("error: no usable group size for this checkpoint")
    if gs != args.group_size:
        print(f"note: group size reduced {args.group_size} -> {gs} for divisibility")

    out = open(args.dst, "wb")

    # --- 256-byte header ---
    header = struct.pack("<II", MAGIC, VERSION)
    header += struct.pack("<7i", cfg["dim"], cfg["hidden_dim"], n_layers,
                          cfg["n_heads"], cfg["n_kv_heads"], cfg["vocab_size"],
                          cfg["seq_len"])
    header += struct.pack("<B", 1 if cfg["shared_classifier"] else 0)
    header += struct.pack("<i", gs)
    out.write(header.ljust(HEADER_BYTES, b"\x00"))

    # --- fp32 norm weights, in runq.c's memory_map_weights order ---
    for key in ("rms_att", "rms_ffn", "rms_final"):
        out.write(w[key].astype(np.float32).tobytes())

    worst = 0.0

    def emit(x):
        """One quantized tensor: all int8 values, then all fp32 scales."""
        nonlocal worst
        q, s, err = quantize_q80(x, gs)
        worst = max(worst, err)
        out.write(q.tobytes())
        out.write(s.tobytes())

    # runq.c reads per-layer tensors as [values|scales] repeated per layer,
    # not as one big block followed by one big block of scales.
    def emit_layered(x):
        chunk = x.size // n_layers
        for l in range(n_layers):
            emit(x[l * chunk:(l + 1) * chunk])

    emit(w["tok"])
    for key in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        emit_layered(w[key])
    if not cfg["shared_classifier"]:
        emit(w["wcls"])

    out.close()

    import os
    src_mb = os.path.getsize(args.src) / 1e6
    dst_mb = os.path.getsize(args.dst) / 1e6
    print(f"group size            : {gs}")
    print(f"shared classifier     : {cfg['shared_classifier']}")
    print(f"fp32  {args.src:<24} {src_mb:8.2f} MB")
    print(f"int8  {args.dst:<24} {dst_mb:8.2f} MB   ({src_mb / dst_mb:.2f}x smaller)")
    print(f"max abs quantization error: {worst:.6f}")


if __name__ == "__main__":
    main()
