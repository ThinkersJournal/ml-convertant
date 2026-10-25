# Patent sweep: ml-convertant 0.1.x (DRAFT, INCOMPLETE: NOT CLEARED FOR PUBLICATION; renamed to PATENT-SWEEP.md only when complete)

Date started: 2026-10-25; Run by: ml-convertant lane, Claude Sonnet 5.5; Method: AI-assisted search and read; NOT legal advice; NOT a freedom-to-operate opinion.

Process: `C:\Projects\PATENT-SWEEP-PROCESS.md` (CireSnave, board 286 item 1).

**Status: the search was blocked and the hits are UNREAD.** Google Patents rate-limited this machine's IP
(HTTP 503) after the first two queries, and every other source tried refused or could not be driven
without a browser (see "Source status"). Nothing below is a clearance. Per the PM (2026-10-25): *"If all stay
blocked, the sweep stays UNREAD for those hits and nothing publishes."* This file exists so the work done so
far, the ideas to be swept and the leads are recorded; it is completed (queries, claims read, verdicts)
before the first publish, and the PM checks for it then.

Commit swept (once complete): the workspace at the time of reading, version line 0.1.x.

## Ideas swept (the mechanisms, as claim-like sentences)

Shipped, or in open PRs at the time of writing:

1. Block-wise symmetric 8-bit quantization of a float tensor: one f16 scale per block of 32 elements,
   scale = block absolute maximum / 127, each element rounded half away from zero (ggml Q8_0), and the inverse.
2. Block-wise 4-bit quantization where the scale is the signed value of the block's largest-magnitude element
   divided by -8, with elements 0-15 in the low nibbles and 16-31 in the high nibbles (ggml Q4_0).
3. 4-bit NormalFloat: a fixed 16-entry codebook at quantiles of a normal distribution, per-block (64) absolute
   maximum normalization, nearest codebook entry chosen by midpoints, two codes per byte (bitsandbytes NF4).
4. Double quantization: the per-block absolute maxima are themselves quantized to 8 bits, after subtracting their
   mean, against a "dynamic" map with an exponent/fraction structure, in blocks of 256.
5. Generation of the signed 8-bit dynamic quantization map from a linear-spaced fraction table per exponent.
6. Super-block ("K-quant") formats: 256 elements in sub-blocks of 16 or 32, each with a quantized scale (and
   minimum), the scales themselves stored in 4, 6 or 8 bits against one or two f16 super-block factors; the
   scale search minimizing a weighted squared or absolute error over trial scales (ggml Q2_K to Q6_K).
7. Repacking a 4-bit weight layer from the AutoAWQ interleaved word order to an output-channel-major layout
   with zero-point and scale rows padded for a GEMM kernel, and the inverse.
8. Offline merge of a low-rank adapter into base weights, `W + scale * B @ A`, in f32 with a fixed
   accumulation order.
9. A conversion that returns, with its output, a measured error (maximum and mean absolute error, RMSE, counts
   of non-finite values and of values whose finiteness changed) and an analytic worst-case bound computed for
   that input from the format's scale rounding and clamping.
10. Verifying a conversion byte for byte against a reference implementation compiled from its source at a pinned
    commit, on seeded random input, in continuous integration.

Planned, not shipped (swept again when built; the inherited leads are recorded below):

11. Post-training quantization with calibration (min/max, percentile, entropy-histogram scale and zero point;
    INT8/INT4; dynamic and static) (the code removed from MLMF; ROADMAP "candidates").
12. Merging a LoRA adapter into a quantized base (dequantize, merge, requantize) (ROADMAP row 11).

## Source status (what was tried on 2026-10-25, and what happened)

| source | access | result |
|---|---|---|
| Google Patents (`patents.google.com`, XHR query and patent pages via `tools/patent-claims.py`) | HTTP, stdlib | **Worked for 2 queries, then HTTP 503 for every later request** from this IP (many probes over more than an hour, retried every 5 minutes). Rate limit, not a block of the tool. |
| Web search tool, restricted to `patents.google.com` / `patents.justia.com` | search API | Returned papers and a few patent URLs; **not a patent database**: no claim text and no way to enumerate. Useful only for leads. |
| Justia Patents (`patents.justia.com`) | HTTP | HTTP 403 (page and search) with a browser User-Agent. The fetch tool also got 403. |
| FreePatentsOnline | HTTP | HTTP 403. |
| Espacenet (`worldwide.espacenet.com`) | HTTP | HTTP 403 (as other lanes reported). |
| WIPO Patentscope | HTTP | Landing page loads; the search is a JavaServer Faces session application that cannot be driven without a browser. Not usable here. |
| USPTO public search (`ppubs.uspto.gov`) | HTTP | Direct document URL returned 404; the search needs a browser session. Other lanes found scanned PDFs readable only as page images. Not attempted further. |
| PatentsView (`api.patentsview.org`) | HTTP | Returns an AWS WAF JavaScript challenge page, not data. The newer `search.patentsview.org` did not resolve. |
| EPO publication server (`data.epo.org`) | HTTP | Landing page only; no full-text search. |
| A real browser (Claude in Chrome, built-in browser) | tools | Not available in this session. A session that has one could finish the reading. |

## Queries run (source, query, date)

| # | idea | query | source | result |
|---|---|---|---|---|
| Q1 | 1, 2 | block-wise quantization neural network weights per-block scale 4-bit | Google Patents XHR, 2026-10-25 | 10 results; candidates listed under Leads |
| Q2 | 3 | quantization normal distribution quantiles codebook 4-bit data type neural network weights | Google Patents XHR, 2026-10-25 | 18 results, none relevant (wireless, video) |
| Q3 to Q16 | 4-12 | (queued: double quantization; dynamic exponent 8-bit map; LoRA merge; activation-aware quantization; repack layouts; conversion error reporting; dequant-merge-requant; block floating point; super-block scales; nibble packing; zero-point packing; error bounds) | Google Patents XHR | **503, not run**; a retry loop continues with 5-minute backoff and 150 s between queries |
| W1 to W5 | 1, 3, 5, 7, 8 | five phrasings of the above through the web search tool (two restricted to Google Patents and Justia) | web search | no patent claims; leads only (EP4341862A1) |

## Leads (NOT read; no verdict)

From Q1, the nearest by title (claims not yet read):

| patent | title | assignee | why it is a lead |
|---|---|---|---|
| US12045307B2, US12412082B2 | Fine-grained per-vector scaling for neural network quantization | Nvidia | per-vector (16 to 64 elements) scale factors with a two-level scale scheme: near ideas 1, 2, 6 |
| US20250117639A1 | Loss-error-aware quantization of a low-bit neural network | Intel | quantization minimizing a loss error: near idea 6's scale search |
| EP4341862A1 | Low-rank adaptation of neural network models | (not yet identified) | LoRA; claims not seen; near idea 8 |
| US12099930B2 | (TI) histogram tail-discard, moving-average maximum, power-of-two clip | Texas Instruments | inherited from the MLMF lane's sweep; they read its claims and found it clear against MLMF's old code, **not against us if we add calibration** (idea 11) |
| US20230342613A1 | integer-only QAT | Alphaics | same: inherited from the MLMF lane; relevant only if idea 11 is built |

Background that is prior art, not patents (recorded so a later reader does not re-derive it): the ggml block
formats are open source (MIT) and predate this work; NF4 and double quantization are published in the QLoRA
paper (Dettmers et al.) and implemented in bitsandbytes (MIT); AWQ is published (Lin et al.) and AutoAWQ is MIT;
LoRA is published (Hu et al.) and PEFT is Apache-2.0. A published method can still be claimed by a patent filed
earlier or by a different party; none of this clears anything.

## Hits

None read yet. To complete the sweep, for each idea: run the documented queries, read the independent claims of
the closest five hits in full with `tools/patent-claims.py`, quote the overlapping and the differing limitations,
and give `clear`, `design-around` or `needs-review`. Any `needs-review` goes to the PM.

## Limitations

- Incomplete: see Status. The only patents examined are titles in two result lists; none was read.
- AI-assisted, machine-converted patent text (when it can be read); search coverage is whatever the sources
  return; the general web search tool is not a patent database.
- Not legal advice; not a freedom-to-operate opinion. A lawyer's review is a separate, later purchase.
- A hit being absent here does not mean the idea is free of patents.
