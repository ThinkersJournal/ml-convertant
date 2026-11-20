"""Regenerate tests/golden/real_iq_encode.tsv: ggml's own quantize_iq4_nl / quantize_iq4_xs / quantize_iq3_s (no importance
matrix) on rows of REAL dense weights (Qwen3-0.6B, bf16), as the SHA-256 of the bytes ggml writes.

usage: gen_real_iq_encode.py <models_dir> <ref_dir_cmd_prefix> <workdir> [out.tsv]
  ref command prefix runs tools/golden/{iq4_nl,iq4_xs,iq3_s}_reference built from the pinned llama.cpp; the binary for
  type T is "<prefix>/ref_T", e.g. "wsl.exe -d Ubuntu-24.04 -- /root/o3"
The row selection (tensor, first row, row count) is written into the TSV, which the Rust test reads back.
"""
import hashlib, json, os, struct, subprocess, sys

DIR = "Qwen3-0.6B-dense"
TYPES = ["iq4_nl", "iq4_xs", "iq3_s"]
NAMES = ["model.layers.%d.%s.weight" % (l, p) for l in (0, 13, 27) for p in
         ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj")]
NAMES += ["model.embed_tokens.weight"]
ROWS = 16


def wp(p, cmd):
    p = p.replace("\\", "/")
    if cmd[0].lower().startswith("wsl") and len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def main():
    models, cmd, work = sys.argv[1], sys.argv[2].split(), sys.argv[3]
    path = os.path.join(models, DIR, "model.safetensors")
    f = open(path, "rb")
    (n,) = struct.unpack("<Q", f.read(8))
    hdr = json.loads(f.read(n))
    base = 8 + n
    rows = []
    for name in NAMES:
        t = hdr[name]
        assert t["dtype"] == "BF16", name
        r, c = t["shape"]
        assert c % 256 == 0
        a, _ = t["data_offsets"]
        first = (r // 2) // ROWS * ROWS if name.endswith("embed_tokens.weight") else 0
        f.seek(base + a + first * c * 2)
        raw = f.read(ROWS * c * 2)
        h = struct.unpack("<%dH" % (ROWS * c), raw)
        x = struct.pack("<%dI" % (ROWS * c), *[v << 16 for v in h])
        inp = os.path.join(work, "in.f32")
        open(inp, "wb").write(x)
        for ty in TYPES:
            ob, od = os.path.join(work, "o.bin"), os.path.join(work, "o.deq")
            subprocess.run(cmd[:-1] + [cmd[-1] + "/ref_" + ty, wp(inp, cmd), wp(ob, cmd), wp(od, cmd)], check=True,
                           stdout=subprocess.DEVNULL)
            rows.append("\t".join([name, str(first), str(ROWS), ty.upper(), hashlib.sha256(open(ob, "rb").read()).hexdigest()]))
    out = sys.argv[4] if len(sys.argv) > 4 else "tests/golden/real_iq_encode.tsv"
    open(out, "w", newline="\n").write("\n".join(rows) + "\n")
    print(len(rows), "rows")


main()
