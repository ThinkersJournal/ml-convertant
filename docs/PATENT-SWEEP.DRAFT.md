# Patent sweep: ml-convertant 0.1.x (DRAFT, INCOMPLETE: NOT CLEARED FOR PUBLICATION; renamed to PATENT-SWEEP.md only when complete)

Date started: 2026-10-25; Run by: ml-convertant lane, Claude Sonnet 5.5; Method: AI-assisted search and read; NOT legal advice; NOT a freedom-to-operate opinion.

Process: `C:\Projects\PATENT-SWEEP-PROCESS.md` (CireSnave, board 286 item 1).

**Status: the search was blocked and most hits are UNREAD (three read, see Hits).** Google Patents rate-limited this machine's IP
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
| USPTO image server (`image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/<number>`) | web fetch tool (saves the PDF) + PyMuPDF | **Worked** for US grants and applications (PM's route): scanned pages, claims read as images. US documents only. |
| USPTO public search (`ppubs.uspto.gov`) | HTTP | Direct document URL returned 404; the search needs a browser session. |
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

Q1 produced three US leads; they are now READ and moved to Hits (H1 to H3). Still unread:

| patent | title | assignee | why it is a lead |
|---|---|---|---|
| EP4341862A1 | Low-rank adaptation of neural network models | (not yet identified) | LoRA; claims not seen; near idea 8 |
| US12099930B2 | (TI) histogram tail-discard, moving-average maximum, power-of-two clip | Texas Instruments | inherited from the MLMF lane's sweep; they read its claims and found it clear against MLMF's old code, **not against us if we add calibration** (idea 11) |
| US20230342613A1 | integer-only QAT | Alphaics | same: inherited from the MLMF lane; relevant only if idea 11 is built |

Background that is prior art, not patents (recorded so a later reader does not re-derive it): the ggml block
formats are open source (MIT) and predate this work; NF4 and double quantization are published in the QLoRA
paper (Dettmers et al.) and implemented in bitsandbytes (MIT); AWQ is published (Lin et al.) and AutoAWQ is MIT;
LoRA is published (Hu et al.) and PEFT is Apache-2.0. A published method can still be claimed by a patent filed
earlier or by a different party; none of this clears anything.

## Hits (read so far)

Read on 2026-10-25 from the USPTO's scanned grant/publication PDFs (`image-ppubs.uspto.gov`, fetched with the
web fetch tool, rendered with PyMuPDF, claim pages read as images). The independent claims were read in full.
Google Patents was throttled, so these are not machine-converted text.

| # | patent | title | assignee | claims read | overlap | our difference | verdict |
|---|---|---|---|---|---|---|---|
| H1 | US12045307B2 (priority 2020-08-28) | Fine-grained per-vector scaling for neural network quantization | Nvidia | indep. 1, 14, 17, 20 (all read in full) | Per-vector scale factors (a scale per vector of quantized parameters within one dimension), a coarse scale in dependents 5-7: the same two-level shape as our K-quants (ideas 4, 6) | Every independent claim requires **computing, with "a set of multipliers" operating "at the bitwidth", a vector dot-product value for each vector of quantized parameters, scaling that dot-product by the per-vector scale factor, and accumulating the scaled dot-products to produce each element of an output tensor** (claim 20: processing by a neural network model). We convert data; we compute no dot products and run no layer | clear |
| H2 | US12412082B2 (continuation of H1) | Fine-grained per-vector scaling for neural network quantization | Nvidia | indep. 1, 16, 19 (all read in full) | Closest in structure: "computing, for each vector of parameters ..., a per-vector scale factor", "computing a coarse scale factor ... shared by at least two of the vectors", "reducing ... a bitwidth of each parameter ... according to a corresponding per-vector scale factor" with the bitwidth "also reduced according to the coarse scale factor": our K-quants store an integer sub-block scale and one f16 super-block factor | Every independent claim ends with "**processing the reduced bitwidth vectors of parameters using a set of multiplier circuits that operate at the reduced bitwidth in a layer of the neural network model to produce each element of a multi-dimensional output tensor**". We run no layer and have no multiplier circuits | clear for conversions |
| H3 | US20250117639A1 (application, published 2025-04-10) | Loss-error-aware quantization of a low-bit neural network | Intel | indep. 1, 8, 15 and the means claim (read in full) | Quantizes network weights to low-bit weights | Requires partitioning the weights into a group to be quantized and a group "to be retrained", calculating a first loss and a second loss "with respect to a loss function", updating the retrained group from their difference, repeating the partition/quantize/retrain, and deploying; we quantize in one pass with no loss function and no retraining. An application; its claims may change before grant | clear |

**Flag for the PM, outside this repository:** H1 and H2 are about the arithmetic of running a quantized layer
(multiplier circuits at the reduced bitwidth, per-vector scaling of dot-products). That is the territory of
Fuel's quantized matrix-multiplication kernels (`fuel-quantized` `vec_dot`, the W4A16 AWQ and NF4 kernels,
the CUDA/Metal/Vulkan quantized paths), not of this project. Fuel's own sweep should read H1 and H2 against
those kernels: their independent claims plausibly read on a per-block-scaled integer dot product in a
neural-network layer. This record does not assess Fuel.

If ML-ConvertAnt ever adds quantized matrix multiplication or inference, H1 and H2 must be re-read against that
code: the limitation that makes them clear for conversions (a multiplier or dot-product step in a layer) would
then be present.

### Still unread

US12099930B2 and US20230342613A1 (inherited from the MLMF lane; relevant only if idea 11 is built); EP4341862A1
(LoRA; not a US document, needs Google Patents or Espacenet); and every hit that queries Q3 to Q16 would return.

## Limitations

- Incomplete: see Status. Three patents were read (H1 to H3); the other queries never ran.
- AI-assisted, machine-converted patent text (when it can be read); search coverage is whatever the sources
  return; the general web search tool is not a patent database.
- Not legal advice; not a freedom-to-operate opinion. A lawyer's review is a separate, later purchase.
- A hit being absent here does not mean the idea is free of patents.
