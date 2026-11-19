"""Golden generator: LoRA merge as produced by PEFT itself, on a real Qwen3-0.6B projection.

DEV-ONLY (the test needs none of this). Venv outside the repo:
    python -m venv <venv>; <venv>/pip install --index-url https://download.pytorch.org/whl/cpu torch
    <venv>/pip install peft transformers safetensors numpy
Run:  <venv>/python tools/golden/gen_peft_merge.py [models_dir]

Base: Qwen/Qwen3-0.6B (Apache-2.0), layer 0 `self_attn.v_proj` (1024x1024), bf16 on disk, loaded as float32
(exact widening). Adapter: PEFT `LoraConfig(init_lora_weights=False)` gives random A and B; we then overwrite
both with a seeded generator so the adapter is reproducible. `merge_and_unload()` is PEFT's own merge
(`peft/tuners/lora/layer.py`, `get_delta_weight` + `merge`). Output, per case, in tests/golden/peft_merge/:
  <case>_ab.bin      f32 LE: A [rank, in] then B [out, rank]
  <case>_merged.bin  f32 LE: first ROWS rows of PEFT's merged weight [ROWS, in]
and the manifest tests/golden/peft_merge/manifest.tsv (case rank alpha rslora out in rows peft transformers torch).
"""

import os
import sys

import numpy as np
import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

ROWS = 32
CASES = [("lora_r4", 4, 8.0, False), ("rslora_r8", 8, 16.0, True)]

models = sys.argv[1] if len(sys.argv) > 1 else "C:/Models"
base = os.path.join(models, "Qwen3-0.6B-dense")
out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "golden", "peft_merge")
os.makedirs(out_dir, exist_ok=True)

rows = []
for case, rank, alpha, rslora in CASES:
    cfg = AutoConfig.from_pretrained(base)
    cfg.num_hidden_layers = 1
    cfg.layer_types = cfg.layer_types[:1] if getattr(cfg, "layer_types", None) else None
    model = AutoModelForCausalLM.from_pretrained(base, config=cfg, torch_dtype=torch.float32)
    lc = LoraConfig(r=rank, lora_alpha=alpha, target_modules=["v_proj"], use_rslora=rslora,
                    init_lora_weights=False, lora_dropout=0.0)
    pm = get_peft_model(model, lc)
    lin = pm.base_model.model.model.layers[0].self_attn.v_proj
    g = np.random.default_rng(20261119 + rank)
    a = (g.standard_normal((rank, lin.in_features)) * 0.05).astype(np.float32)
    b = (g.standard_normal((lin.out_features, rank)) * 0.05).astype(np.float32)
    with torch.no_grad():
        lin.lora_A["default"].weight.copy_(torch.from_numpy(a))
        lin.lora_B["default"].weight.copy_(torch.from_numpy(b))
    merged = pm.merge_and_unload()
    w = merged.model.layers[0].self_attn.v_proj.weight.detach().numpy().astype(np.float32)
    assert w.shape == (lin.out_features, lin.in_features)
    with open(os.path.join(out_dir, f"{case}_ab.bin"), "wb") as f:
        f.write(a.astype("<f4").tobytes())
        f.write(b.astype("<f4").tobytes())
    with open(os.path.join(out_dir, f"{case}_merged.bin"), "wb") as f:
        f.write(w[:ROWS].astype("<f4").tobytes())
    rows.append([case, rank, alpha, int(rslora), w.shape[0], w.shape[1], ROWS,
                 peft.__version__, transformers.__version__, torch.__version__])
    print(case, "scale", alpha / (rank ** 0.5 if rslora else rank))

with open(os.path.join(out_dir, "manifest.tsv"), "w", newline="\n") as f:
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
