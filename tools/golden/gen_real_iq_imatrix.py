"""Regenerate tests/golden/real_iq_imatrix.tsv: ggml's own quantize_iq4_nl / quantize_iq4_xs / quantize_iq3_s WITH a real
importance matrix, on rows of REAL dense weights (SmolLM2-135M-Instruct, bf16), as the SHA-256 of the bytes ggml writes.

usage: gen_real_iq_imatrix.py <models_dir> <ref_cmd_prefix> <workdir> [out.tsv]
  ref command prefix as in gen_real_iq_encode.py ("wsl.exe -d Ubuntu-24.04 -- /root/o3" runs /root/o3/ref_<type>).
The imatrix is bartowski's legacy-format file (C:/Models/imatrix/smollm2-bartowski/SmolLM2-135M-Instruct.imatrix); its
per-column weight vector is sum_of_squares / ncall, in f32, exactly as the Rust test computes it.
Row length must be a multiple of 256 for IQ4_XS and IQ3_S, so those two cover only ffn_down (1536 columns); IQ4_NL
(multiple of 32) also covers the 576-column projections.
"""
import hashlib, json, os, struct, subprocess, sys

DIR = "SmolLM2-135M-Instruct"
IMAT = "imatrix/smollm2-bartowski/SmolLM2-135M-Instruct.imatrix"
LAYERS = (0, 15, 29)
PARTS = [("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
         ("self_attn.o_proj", "attn_output"), ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"),
         ("mlp.down_proj", "ffn_down")]
ROWS = 16


def wp(p, cmd):
    p = p.replace("\\", "/")
    if cmd[0].lower().startswith("wsl") and len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def read_imatrix(path):
    b = open(path, "rb").read()
    (n,) = struct.unpack_from("<i", b, 0)
    o, out = 4, {}
    for _ in range(n):
        (ln,) = struct.unpack_from("<i", b, o); o += 4
        name = b[o:o + ln].decode(); o += ln
        ncall, nval = struct.unpack_from("<ii", b, o); o += 8
        vals = struct.unpack_from("<%df" % nval, b, o); o += 4 * nval
        out[name] = (ncall, vals)
    return out


def main():
    models, cmd, work = sys.argv[1], sys.argv[2].split(), sys.argv[3]
    im = read_imatrix(os.path.join(models, IMAT))
    f = open(os.path.join(models, DIR, "model.safetensors"), "rb")
    (n,) = struct.unpack("<Q", f.read(8))
    hdr = json.loads(f.read(n))
    base = 8 + n
    rows = []
    for l in LAYERS:
        for hf, gg in PARTS:
            name = "model.layers.%d.%s.weight" % (l, hf)
            iname = "blk.%d.%s.weight" % (l, gg)
            t = hdr[name]
            assert t["dtype"] == "BF16", name
            r, c = t["shape"]
            ncall, vals = im[iname]
            assert len(vals) == c, (iname, len(vals), c)
            w = [struct.unpack("<f", struct.pack("<f", v / ncall))[0] for v in vals]
            a, _ = t["data_offsets"]
            f.seek(base + a)
            h = struct.unpack("<%dH" % (ROWS * c), f.read(ROWS * c * 2))
            open(os.path.join(work, "in.f32"), "wb").write(struct.pack("<%dI" % (ROWS * c), *[v << 16 for v in h]))
            open(os.path.join(work, "w.f32"), "wb").write(struct.pack("<%df" % c, *w))
            types = ["iq4_nl"] + (["iq4_xs", "iq3_s"] if c % 256 == 0 else [])
            for ty in types:
                ob, od = os.path.join(work, "o.bin"), os.path.join(work, "o.deq")
                subprocess.run(cmd[:-1] + [cmd[-1] + "/ref_" + ty, wp(os.path.join(work, "in.f32"), cmd), wp(ob, cmd),
                                           wp(od, cmd), wp(os.path.join(work, "w.f32"), cmd)], check=True,
                               stdout=subprocess.DEVNULL)
                rows.append("\t".join([name, iname, "0", str(ROWS), ty.upper(), hashlib.sha256(open(ob, "rb").read()).hexdigest()]))
    out = sys.argv[4] if len(sys.argv) > 4 else "tests/golden/real_iq_imatrix.tsv"
    open(out, "w", newline="\n").write("\n".join(rows) + "\n")
    print(len(rows), "rows")


main()
