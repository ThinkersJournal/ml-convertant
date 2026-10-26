# Patent sweep: ml-convertant 0.1.x (DRAFT, INCOMPLETE: NOT CLEARED FOR PUBLICATION; renamed to PATENT-SWEEP.md only when complete)

Date started: 2026-10-25; Run by: ml-convertant lane, Claude Sonnet 5.5; Method: AI-assisted search and read; NOT legal advice; NOT a freedom-to-operate opinion.

Process: `C:\Projects\PATENT-SWEEP-PROCESS.md` (CireSnave, board 286 item 1).

**Status: the search was blocked and most hits are UNREAD (six read, see Hits).** Google Patents rate-limited this machine's IP
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
| Q3 | 4 | double quantization quantization constants quantized again block-wise neural network | Google Patents XHR | never ran (503 on 8 attempts) |
| Q4 | 5 | dynamic exponent data type 8-bit quantization map linear fraction region | Google Patents XHR, 2026-10-25/26 | 10 results: image/video processing; none relevant |
| Q5 | 8 | low-rank adaptation merge adapter matrices into pretrained weight matrices inference | same | 10 results: unrelated AI/medical/security patents; none relevant (search is semantic and noisy) |
| Q6 | 7 | activation-aware weight quantization scaling salient weight channels low-bit | same | none relevant |
| Q7 | 7 | repack quantized weights layout matrix multiplication kernel interleave 4-bit packed integer words | same | 4 results, none relevant |
| Q8 | 9 | report quantization error each tensor conversion maximum absolute error mean error model conversion | same | none relevant |
| Q9 | 12 | dequantize quantized weights apply low-rank update requantize merged weights | same | none relevant (video coding) |
| Q10 | 1, 2, 6 | block floating point shared scale per block of weights 32 elements neural network | same | **relevant**: Nvidia H1/H2, Microsoft H4 to H6 (and unread Microsoft CN113348474B, EP3895075B1, US12585926B2, CN113273082B, US20250384257A1 Intel) |
| Q11 | 6 | super-block quantization scales quantized minimums hierarchical scale weights | same | none relevant (video/entropy coding) |
| Q12 | 1, 2 | packing 4-bit weights two per byte nibble block scale half precision large language model | same | none relevant (ray tracing hardware) |
| Q13 | 7 | asymmetric group-wise quantization zero point packed with weights scales per group | same | none relevant except CN111937010B (Amazon, "Accelerated quantized multiplication and addition operations": compute-side; unread) |
| Q14 | 9 | worst-case quantization error bound per block scale rounding neural network weights | same | none relevant (Cerebras compute fabric, video coding) |
| Q15, Q16 | 10, 9 | (queued) | same | never ran |
| A1 to A12 | 3-12 | second batch of sharper keyword queries (double quantization, normal float, LoRA merge, AWQ, repack, super-block, 32-weight block file format, error reporting, format conversion, bit-exact verification, calibration, dequantize-merge-quantize) | Google Patents XHR, started 2026-10-26 | see "Second batch" below |
| W1 to W5 | 1, 3, 5, 7, 8 | five phrasings of the above through the web search tool (two restricted to Google Patents and Justia) | web search | no patent claims; leads only (EP4341862A1) |

## Leads (NOT read; no verdict)

Q1 and Q10 produced six US leads; they are now READ and moved to Hits (H1 to H6). Still unread:

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
| H1 | US12045307B2 (priority 2020-08-28) | Fine-grained per-vector scaling for neural network quantization | Nvidia | indep. 1, 14, 17, 20 (all read in full) | Per-vector scale factors (a scale per vector of quantized parameters within one dimension), a coarse scale in dependents 5-7: the same two-level shape as our K-quants (ideas 4, 6) | Independent claims 1, 14 and 17 require **receiving per-vector scale factors for vectors of quantized parameters, then computing, with "a set of multipliers" operating "at the bitwidth", a vector dot-product value for each vector, scaling each dot-product by its per-vector scale factor, and accumulating the scaled dot-products to produce each element of an output tensor**; a neural-network layer is NOT required by these three (it appears in dependent claim 8; independent claim 20 is the one that has "a neural network model" process the tensors, with the same multiply/scale steps and a coarse scale). Quantizing alone (our encode, decode, repack and merge) lacks the multiply, per-vector-scale and accumulate steps: we compute no vector dot products at a reduced bitwidth | clear |
| H2 | US12412082B2 (continuation of H1) | Fine-grained per-vector scaling for neural network quantization | Nvidia | indep. 1, 16, 19 (all read in full) | Closest in structure: "computing, for each vector of parameters ..., a per-vector scale factor", "computing a coarse scale factor ... shared by at least two of the vectors", "reducing ... a bitwidth of each parameter ... according to a corresponding per-vector scale factor" with the bitwidth "also reduced according to the coarse scale factor": our K-quants store an integer sub-block scale and one f16 super-block factor | Every independent claim ends with "**processing the reduced bitwidth vectors of parameters using a set of multiplier circuits that operate at the reduced bitwidth in a layer of the neural network model to produce each element of a multi-dimensional output tensor**". We run no layer and have no multiplier circuits | clear for conversions: the same dot-product ending (in a layer of a neural network model) as H1; '082 adds the coarse scale (the K-quant shape) |
| H3 | US20250117639A1 (application, published 2025-04-10) | Loss-error-aware quantization of a low-bit neural network | Intel | indep. 1, 8, 15 and the means claim (read in full) | Quantizes network weights to low-bit weights | Requires partitioning the weights into a group to be quantized and a group "to be retrained", calculating a first loss and a second loss "with respect to a loss function", updating the retrained group from their difference, repeating the partition/quantize/retrain, and deploying; we quantize in one pass with no loss function and no retraining. An application; its claims may change before grant | clear |
| H4 | US11645493B2 (priority 2018-05-04) | Flow for quantized neural networks | Microsoft | indep. 1, 11, 17 (read in full) | Quantizing a model's tensors with exponents shared over granules finer than a whole tensor (block floating point), per-row or per-column | Every independent claim requires **applying input tensors to an input layer of the quantized model, comparing the quantized output with the output of the normal-precision model, and, based on that comparison, selecting a new quantized-precision format** with a different parameter (per-row/per-column shared exponent). Our error report compares tensors element by element; we run no model, compare no model outputs, and select no format from them. If ML-ConvertAnt ever picks a format automatically from measured error, re-read this | clear |
| H5 | US12443848B2 | Neural network activation compression with narrow block floating-point | Microsoft | indep. 1, 10, 17 (read; claim 17 to its visible end) | Converts values between block floating-point formats of different precision | All three are about **training**: forward propagation per layer producing activations in a first BFP format, converting them to a lower-precision second format, storing them, calculating a loss, retrieving and decompressing them, and backpropagating. We handle weights offline, with no layers, activations, loss or backpropagation | clear |
| H6 | US10167800B1 (priority 2017-08-18) | Hardware node having a matrix vector unit with block-floating point processing | Microsoft | indep. 1, 7, 14 (read in full) | Closest on the block side: "first process a first subset of the [vector data] to determine a first shared exponent ... second subset ... a second shared exponent ... in a manner that no vector data from the second subset influences the first shared exponent", i.e. independently determined per-block shared factors, as in every block format here | Each independent claim is tied to a **matrix vector unit** processing "vector data corresponding to at least one layer of a neural network model" (claim 1 also a message processor, scalar processor and a pipeline of multifunction units), and the shared quantity is an **exponent** of floating-point data. Our scales are real-valued multipliers (f16/f32), not exponents, and we run no matrix vector unit. The nearest claim on the format side; a power-of-two-scale block format (e.g. MX-style E8M0) or any hardware-oriented code would need a fresh read | clear |

**Flag for the PM, outside this repository (board 307):** H1 and H2 are about the arithmetic of running a quantized computation
(multipliers at the reduced bitwidth, per-vector scaling of dot-products, accumulation; for H1 claims 1, 14 and 17 no layer is needed). That is the territory of
Fuel's quantized matrix-multiplication kernels (`fuel-quantized` `vec_dot`, the W4A16 AWQ and NF4 kernels,
the CUDA/Metal/Vulkan quantized paths), not of this project. Fuel's own sweep should read H1 and H2 against
those kernels: H1's independent claims 1, 14 and 17 plausibly read on a per-block-scaled integer dot product with accumulation, and H2's on the same inside a layer. This record does not assess Fuel.

If ML-ConvertAnt ever adds quantized matrix multiplication or inference, H1 and H2 must be re-read against that
code: the limitation that makes them clear for conversions (a multiply/scale/accumulate step on quantized vectors) would
then be present.

### Still unread

US12099930B2 and US20230342613A1 (inherited from the MLMF lane; relevant only if idea 11 is built); EP4341862A1
(LoRA; not a US document, needs Google Patents or Espacenet); the Microsoft block-floating-point family members listed
under Q10; CN111937010B; and whatever the second batch returns.

## Limitations

- Incomplete: see Status. Six patents were read (H1 to H6); queries Q3, Q15 and Q16 never ran and Q4 to Q14 returned mostly unrelated patents (see Queries).
- AI-assisted, machine-converted patent text (when it can be read); search coverage is whatever the sources
  return; the general web search tool is not a patent database.
- Not legal advice; not a freedom-to-operate opinion. A lawyer's review is a separate, later purchase.
- A hit being absent here does not mean the idea is free of patents.
