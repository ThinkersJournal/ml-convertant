# Draft upstream issue: `quantize_iq3_s` corrupts a block that has an exactly-zero 32-element sub-block

Status: DRAFT, not filed (2026-11-16). Target: https://github.com/ggml-org/llama.cpp. Filing goes through the PM (outbound-issue ledger).
Search before filing (2026-11-16): `gh search issues --repo ggml-org/llama.cpp` for "iq3_s zero", "quantize_row_iq3_s_impl", "iq3_s sub-block all zero" returned nothing;
the control query "IQ3_S" returns IQ3_S issues, so the search reaches the topic. Not searched: closed PRs by file path, other forks.

---

**Title:** quantize_iq3_s: an exactly-zero 32-element sub-block shifts the grid indices and signs of the following sub-blocks

**Summary**

In `quantize_row_iq3_s_impl` (`ggml/src/ggml-quants.c`), a 32-element sub-block whose values are all exactly zero takes the `if (!max) { scales[ib] = 0; continue; }` branch. That `continue` skips `qs += bs4;` and `signs += bs8;`, which every other sub-block executes at the end of its iteration. The next non-zero sub-block therefore writes its grid indices into the zero sub-block's slot in `qs` and its signs into the zero sub-block's slot in `signs`, and so on for the rest of the block: they are all one slot early. `qh` is indexed by `ib` and the scales by `ib`, so those stay in place. `dequantize_row_iq3_s` reads every field by `ib`, so every sub-block after a zero one decodes against the wrong grid indices and signs, and the last sub-block of the block decodes from unwritten (zeroed) bytes.

Seen on `master` at 5cb95167983f7484febb0e97c7b00887e265ebcf (2026-11-16) and on `9d57ce456c94d241dde672b2db9cf18879766568`.

**Repro** (self-contained; same build stubs as the ggml test harnesses; `docs/upstream/iq3s_zero_subblock_repro.c` in ThinkersJournal/ml-convertant)

One 256-element row: a deterministic non-zero pattern with elements 32..63 set to exactly 0.0, `quantize_iq3_s` then `dequantize_row_iq3_s`. Mean |x| and mean |decoded - x| per sub-block, built with `gcc -O2 -DNDEBUG` on Ubuntu 24.04, gcc 13.3:

| sub-block | mean abs x | error, ggml as is | error, with the two-line fix |
|---|---|---|---|
| 0 | 1.032 | 0.112 | 0.112 |
| 1 (zero) | 0.000 | 0.021 | 0.004 |
| 2 | 1.037 | 2.019 | 0.124 |
| 3 | 0.968 | 1.876 | 0.138 |
| 4 | 1.022 | 1.939 | 0.133 |
| 5 | 0.972 | 1.925 | 0.155 |
| 6 | 1.032 | 1.940 | 0.129 |
| 7 | 0.965 | 1.202 | 0.123 |

(The zero sub-block itself decodes to `d`, small but not zero, because IQ3_S has no zero level; that is separate and expected.)

**Fix** (two lines, inside the `if (!max)` branch):

```c
            if (!max) {
                scales[ib] = 0;
                qs += bs4;
                signs += bs8;
                continue;
            }
```

An all-zero sub-block's grid index, `qh` bits and signs are all zero, so writing nothing at its own slot (the buffer is zeroed by `memset(&y[ibl], 0, ...)`) is correct once the pointers advance.

**Scope**

The result is only wrong for a block that has an exactly-zero sub-block and a later non-zero one. Real weights rarely have 32 exact zeros in a row, but pruned or masked tensors do, as can padding. I have only checked `quantize_iq3_s` (`quantize_row_iq3_s_impl`); `quantize_row_iq3_xxs_impl` writes into a `q3` scratch buffer indexed by `ib` and has no such pointer, and I have not read the IQ2 quantizers for the same pattern.

**Cross-check**

An independent implementation (`ml-convertant-ggml`, Rust, written from ggml's algorithm) that places each sub-block at its own slot is byte-identical to ggml with the two-line fix on 400 random 256-element blocks in which each 32-element sub-block was zeroed with probability 0.3 (44000 bytes), and byte-identical to unpatched ggml on every block without an exactly-zero sub-block next to a non-zero one (five random seeds of 640000 values, and the committed vectors). So the fix does not change any output that was already correct.

Existing quantized files made by an affected version have the bad blocks baked in; the fix only affects new quantizations.
