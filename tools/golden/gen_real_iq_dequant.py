"""Regenerate tests/golden/real_iq_dequant.tsv: for every IQ4_NL / IQ4_XS / IQ3_S / IQ3_XXS / IQ2_XXS tensor in the real GGUF
files, the SHA-256 of ggml's own dequantize_row_* output (little-endian f32), from the pinned llama.cpp.

usage: gen_real_iq_dequant.py <models_dir> <reference_binary_cmd> <workdir>
  models_dir  holds gguf-corpus/quants/SmolLM2-135M-Instruct-{IQ3_XS,IQ4_XS}.gguf and
              gguf-corpus/iq-extra/smollm2-135m-instruct-iq3_xxs-imat.gguf
  reference_binary_cmd  command prefix that runs tools/golden/gguf_dequant_reference, e.g.
                        "wsl.exe -d Ubuntu-24.04 -- /root/out/gguf_dequant_reference"
  workdir     a scratch dir visible to both (the binary is given the same path)
"""
import hashlib, os, struct, subprocess, sys

TYPES = {16: ("IQ2_XXS", 256, 66), 18: ("IQ3_XXS", 256, 98), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 23: ("IQ4_XS", 256, 136),
         19: ("IQ1_S", 256, 50), 29: ("IQ1_M", 256, 56)}
# paths relative to <models_dir>/gguf-corpus; this is also the TSV file column
FILES = ["quants/SmolLM2-135M-Instruct-IQ3_XS.gguf", "quants/SmolLM2-135M-Instruct-IQ4_XS.gguf",
         "iq-extra/smollm2-135m-instruct-iq3_xxs-imat.gguf",
         "iq-extra/phi-3-mini-128k-instruct-iq2_xxs-imat.gguf",
         "iq-extra/tinyllama-1.1b-chat-v1.0-iq1_s-imat.gguf"]


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def rstr(f):
    (n,) = rd(f, "<Q")
    return f.read(n).decode()


def skipval(f, t):
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sz:
        f.read(sz[t])
    elif t == 8:
        rstr(f)
    elif t == 9:
        (et,) = rd(f, "<I")
        (n,) = rd(f, "<Q")
        for _ in range(n):
            skipval(f, et)
    else:
        raise SystemExit("bad kv type %d" % t)


def parse(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    (ver,) = rd(f, "<I")
    assert ver == 3
    nt, nkv = rd(f, "<QQ")
    align = 32
    for _ in range(nkv):
        k = rstr(f)
        (t,) = rd(f, "<I")
        if k == "general.alignment":
            assert t == 4
            (align,) = rd(f, "<I")
        else:
            skipval(f, t)
    ts = []
    for _ in range(nt):
        n = rstr(f)
        (nd,) = rd(f, "<I")
        dims = rd(f, "<%dQ" % nd)
        (ty,) = rd(f, "<I")
        (off,) = rd(f, "<Q")
        ts.append((n, dims, ty, off))
    pos = f.tell()
    base = (pos + align - 1) // align * align
    return base, ts


def wp(p, cmd):
    """Path as the reference binary sees it: /mnt/c/... when it runs under wsl.exe."""
    p = p.replace("\\", "/")
    if cmd[0].lower().startswith("wsl") and len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    models, cmd, work = sys.argv[1], sys.argv[2].split(), sys.argv[3]
    rows = []
    for name in FILES:
        path = os.path.join(models, "gguf-corpus", name)
        whole = hashlib.sha256(open(path, "rb").read()).hexdigest()
        base, ts = parse(path)
        spec, sel = [], []
        for i, (n, dims, ty, off) in enumerate(ts):
            if ty not in TYPES:
                continue
            tn, blk, bytes_ = TYPES[ty]
            nel = 1
            for d in dims:
                nel *= d
            assert nel % blk == 0
            nb = nel // blk * bytes_
            spec.append("%d %d %d %d %d" % (len(sel), ty, base + off, nb, nel))
            sel.append((n, tn, nel))
        sp = os.path.join(work, "spec.txt")
        open(sp, "w").write("\n".join(spec) + "\n")
        out = os.path.join(work, "out")
        os.makedirs(out, exist_ok=True)
        subprocess.run(cmd + [wp(path, cmd), wp(sp, cmd), wp(out, cmd)], check=True)
        for i, (n, tn, nel) in enumerate(sel):
            d = open(os.path.join(out, "%d.f32" % i), "rb").read()
            assert len(d) == nel * 4
            rows.append("\t".join([name, whole, n, tn, str(nel), sha(d)]))
    open(sys.argv[4] if len(sys.argv) > 4 else "tests/golden/real_iq_dequant.tsv", "w", newline="\n").write("\n".join(rows) + "\n")
    print(len(rows), "rows")


main()
