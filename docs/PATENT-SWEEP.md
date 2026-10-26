# Patent sweep: ml-convertant 0.1.x @ main 042764f plus open PRs #10-#15 (first record, 2026-10-26)

Date: 2026-10-25 to 2026-10-26; Run by: ml-convertant lane, Claude Sonnet 5.5; Method: AI-assisted search and read; NOT legal advice; NOT a freedom-to-operate opinion.

Process: `C:\Projects\PATENT-SWEEP-PROCESS.md` (CireSnave, board 286 item 1).

**Status: complete as an AI-assisted first pass, with the limits listed below. No `needs-review` hit.** Twenty-two
patents were read (H1 to H22) and every verdict is `clear` for ML-ConvertAnt as it is today: a library that converts
model data offline (quantize, dequantize, repack, merge) and runs no network, no training and no dot-product
arithmetic. Google Patents rate-limited this machine for most of a day (HTTP 503); the other sources were refused
or unusable (see "Source status"); the claims were read from Google Patents' text where it answered and from the
USPTO's scanned PDFs (read as images) otherwise. Several hits are clear only because of a limitation we lack
(a multiplier or dot-product step, a layer, training, a loss function, a matrix vector unit); the record says which,
and "Re-read triggers" below lists what would change the verdicts.

Commit swept: `origin/main` `042764f` (version 0.1.6) with the open PRs #10 (differential workflow), #11 (LoRA merge) and #12 to #15 (K-quants) read as source; version line **0.1.x** of the manifest (`Cargo.toml` `[workspace.package] version`).

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

## Source status (what was tried on 2026-10-25 and 26, and what happened)

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
| Q10 | 1, 2, 6 | block floating point shared scale per block of weights 32 elements neural network | same | **relevant**: Nvidia H1/H2, Microsoft H4 to H6 (also read: Microsoft H13 to H20 and Intel H21) |
| Q11 | 6 | super-block quantization scales quantized minimums hierarchical scale weights | same | none relevant (video/entropy coding) |
| Q12 | 1, 2 | packing 4-bit weights two per byte nibble block scale half precision large language model | same | none relevant (ray tracing hardware) |
| Q13 | 7 | asymmetric group-wise quantization zero point packed with weights scales per group | same | none relevant except CN111937010B (Amazon, compute-side; read as H22) |
| Q14 | 9 | worst-case quantization error bound per block scale rounding neural network weights | same | none relevant (Cerebras compute fabric, video coding) |
| Q15, Q16 | 10, 9 | (queued) | same | never ran |
| A1 to A12 | 3-12 | second batch of sharper keyword queries (double quantization, normal float, LoRA merge, AWQ, repack, super-block, 32-weight block file format, error reporting, format conversion, bit-exact verification, calibration, dequantize-merge-quantize), 2026-10-26 | Google Patents XHR | mostly unrelated (video coding, wireless sensing, hardware safety); A7 repeated the Nvidia/Microsoft cluster; A8 found MathWorks EP3843013A1 (H7); A10 found EP3788557B1 (H13); A5, A6, A9, A11, A12 found nothing relevant |
| W1 to W5 | 1, 3, 5, 7, 8 | five phrasings of the above through the web search tool (two restricted to Google Patents and Justia) | web search | no patent claims; leads only (EP4341862A1) |

## Leads

Every lead the searches produced was read and is in Hits (H1 to H22). Background that is prior art, not patents
(recorded so a later reader does not re-derive it): the ggml block formats are open source (MIT) and predate this
work; NF4 and double quantization are published in the QLoRA paper (Dettmers et al.) and implemented in
bitsandbytes (MIT); AWQ is published (Lin et al.) and AutoAWQ is MIT; LoRA is published (Hu et al.) and PEFT is
Apache-2.0. A published method can still be claimed by a patent filed earlier or by a different party; none of this
clears anything.

## Hits

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

| H7 | EP3843013A1 (MathWorks, pending) | Systems and methods for quantizing a neural network | MathWorks | claim 1 (the only independent; read in full, Google Patents text) | Quantizing floating-point values to integer or fixed point | Requires determining points within a network that generate numeric values, **presenting visualizations of statistics** and of performance information, quantizing "based on ... characteristics of the one or more target hardware devices", and **generating code** for the quantized network. We do none of these | clear |
| H8 | EP4341862A1 (Microsoft, pending) | Low-rank adaptation of neural network models | Microsoft | claim 1 (the only independent method; claims 11-12 incorporate 1-10) | Base weight matrices plus low-rank factorization matrices; dependent claims 8-9 combine them for inference and remove them by subtracting | Claim 1 requires adding low-rank matrices "treated as trainable parameters" and **training the domain model with domain-specific data without modifying the base weights**. Our merge performs no training: it adds an already-trained adapter to the weights offline | clear (the training step is absent). Priority 2021-05-19; the merge arithmetic itself is described in dependent claims 8-9, so any future feature that trains adapters must be re-read |
| H9 | US12099930B2 (TI, active, expiry est. 2043) | Parametric power-of-2 clipping activations for quantization for convolutional neural networks | Texas Instruments | indep. 1, 10, 17 (claim 1 read in full; 10 and 17 read to the end of the quantizing steps: same limitations in system and medium form) | Quantization with a clipping value | Requires receiving image data through a capture subsystem and quantizing with a power-of-2 parametric activation function: **histogram of the data, discard the histogram tail, estimate a maximum, determine a new maximum with a moving average over historical values, clip to the nearest power of two**. We do none of these on weights. **Relevant if calibration (idea 11) is built: do not combine tail-discard, moving-average maximum and power-of-two clipping** | clear |
| H10 | US20230342613A1 (Alphaics, abandoned) | System and method for integer only quantization aware training on edge devices | Alphaics | indep. 1, 8 (read) | Integer quantization | Training: pseudo cross-entropy, gradient stabilization, back propagation, residual weight error. We train nothing. Status abandoned (Google Patents) | clear |
| H11 | US12417389B2 (Adobe) | Efficient mixed-precision search for quantizers in artificial neural networks | Adobe | indep. 1, 12, 16 (claim 1 read in full; 12 read to the first elements) | Choosing a bit-width per layer | **Training** an ANN with a "folding quantizer" of alternatively sampled candidate quantizers and a parallel full-precision edge, weighted sums, converging to a fixed quantizer. We choose no bit-widths by training or search | clear |
| H12 | US20220044114A1 (Nvidia, application) | Hybrid quantization of neural networks for edge computing applications | Nvidia | indep. 1, 10, 16 (read in full) | Changes precision of weights | "Changing precision ... of a portion of a neural network to generate a first trained model" then more layers to a second trained model, during **training**, by quantization-aware training. We train nothing | clear |
| H13 | EP3788557B1 (Microsoft; same family as H4) | Design flow for quantized neural networks | Microsoft | claim 1 (the only independent; read in full) | As H4 | Adds to H4's limitations "configuring a hardware accelerator ... implemented on an FPGA" with bit-widths from the FPGA's lookup-table size | clear |
| H14 | US12585926B2 (Microsoft) | Adjusting precision and topology parameters for neural network training based on a performance metric | Microsoft | indep. 1, 8, 15 (claims 1 and 8 read; 15 by its opening) | Changes precision parameters | **Training**: forward and backward propagation per iteration and dynamic adjustment of a precision or topology parameter from a performance metric. We train nothing | clear |

| H15 | EP3895075B1 (Microsoft) | Residual quantization for neural networks | Microsoft | claim 1 (read in full; the only independent shown) | Converts floating-point values to a common exponent and mantissas | Executing network: decomposing mantissas into quantized and residual values, **dot products** with weights, selectively performing a second dot product on the residual depending on needed precision, combining outputs. We run no network and compute no dot products | clear |
| H16 | US12067495B2 (Microsoft) | Neural network activation compression with non-uniform mantissas | Microsoft | claim 1 (read in full) | Converts values to fewer mantissa bits | **Forward propagation** for a layer producing activations in a first floating-point format, converting them to a second format with fewer mantissa bits, storing them. Activation compression during execution; we convert stored weights offline | clear |
| H17 | EP3906616B1 (Microsoft) | Neural network activation compression with outlier block floating-point | Microsoft | claim 1 (read to the shifter limitation) | Block format with a shared exponent | Activations of a neural network converted by an "outlier quantizer" to a block format plus outlier values for some activations, with a shifter that shifts mantissas to the shared exponent; activations, not weights; outliers stored separately | clear |
| H18 | US20250061320A1 (Microsoft application) | Adjusting activation compression for neural network training | Microsoft | claim 1 (read in full) | Selecting a lower-precision format | Producing a network with activations, **selecting a second floating-point format "based on a value of a performance metric for the neural network"**, converting activations, storing them in bulk memory (training-time activation compression). We select no format from a network performance metric | clear |
| H19 | CN113348474B (Microsoft; Chinese family member of US12067495B2) | Neural network activation compression with non-uniform mantissas | Microsoft | claim 1 (machine translation read to the compressor step) | as H16 | forward propagation on a layer, activation compression with shared exponent and non-uniform mantissas by a block floating-point compressor; activations during execution | clear |
| H20 | CN113273082B (Microsoft; family of H17) | Neural network activation compression with outlier block floating point | Microsoft | claim 1 (machine translation read to the second conversion) | as H17 | generating activations for training data, determining outlier activation values with extra bits, converting to a block format with shared exponent; activations in training | clear |
| H21 | US20250384257A1 (Intel application) | Machine learning sparse computation mechanism for arbitrary neural networks | Intel | claim 1 (read in full) | A matrix accelerator | A **graphics processor** with a hardware scheduler, multiprocessors, a matrix accelerator and offload circuitry. Hardware; we are software conversions | clear |
| H22 | CN111937010B (Amazon) | Accelerated quantized multiplication and addition operations | Amazon | claim 1 (machine translation, read to the subtraction-engine step) | Linear quantization of floating-point values to n-bit integers with the minimum mapped to zero, the maximum to 2^n - 1 and floating-point zero represented exactly (the shape of an affine min/scale format such as Q4_K) | Concerns the **inputs to a convolutional layer at inference**: store the integers, "read ... by the convolutional layer", a **subtraction engine** subtracting the zero-point integer, and the following accelerated multiply-add. We convert stored weights offline; no convolution, no subtraction engine. Near on the quantization shape, so any future runtime kernel must be re-read against it | clear |

**Flag for the PM, outside this repository (board 307):** H1 and H2 are about the arithmetic of running a quantized computation
(multipliers at the reduced bitwidth, per-vector scaling of dot-products, accumulation; for H1 claims 1, 14 and 17 no layer is needed). That is the territory of
Fuel's quantized matrix-multiplication kernels (`fuel-quantized` `vec_dot`, the W4A16 AWQ and NF4 kernels,
the CUDA/Metal/Vulkan quantized paths), not of this project. Fuel's own sweep should read H1 and H2 against
those kernels: H1's independent claims 1, 14 and 17 plausibly read on a per-block-scaled integer dot product with accumulation, and H2's on the same inside a layer. This record does not assess Fuel.

If ML-ConvertAnt ever adds quantized matrix multiplication or inference, H1 and H2 must be re-read against that
code: the limitation that makes them clear for conversions (a multiply/scale/accumulate step on quantized vectors) would
then be present.

### Reading limits

No lead is left unread. The ideas with no relevant patent in any query (3 NormalFloat, 5 dynamic map, 7 repack, 9 error
reporting, 10 bit-exact verification) are findings about these searches, not proof that none exists; Q3, Q15 and Q16
never ran. Claims of H19, H20 and H22 were read in Google's machine translation. For H9 to H14, H16 to H18 and H21 the
independent claims after claim 1 were identified by their opening words (and, for H9, H12 and H14, partly read) rather
than read to the end; they are the system or medium counterparts of claim 1 in every case.

## Re-read triggers

Sweep again (new mechanism, per `PATENT-SWEEP-PROCESS.md`) before shipping any of:

- quantized matrix multiplication, a dot-product kernel, or inference of any kind (H1, H2, H6, H15, H22 become relevant);
- calibration: histogram, percentile or entropy statistics, a moving-average maximum, power-of-two clipping (H9);
- choosing a format automatically from measured error or from a model's output (H4, H18);
- training adapters or quantization-aware training (H8, H10, H11, H12, H14);
- a power-of-two (exponent-only) block scale, MX-style formats, or hardware-targeted code (H6, H13);
- merging into a quantized base (ROADMAP row 11) re-reads H8 and the quantization hits.

## Limitations

- Coverage: twenty-two patents were read (H1 to H22); queries Q3, Q15 and Q16 never ran and most other queries returned unrelated patents (see Queries). For H9 to H14, H16 to H18 and H21 the independent claims after claim 1 were not read to the end (see Hits, last note).
- AI-assisted, machine-converted patent text (when it can be read); search coverage is whatever the sources
  return; the general web search tool is not a patent database.
- Not legal advice; not a freedom-to-operate opinion. A lawyer's review is a separate, later purchase.
- A hit being absent here does not mean the idea is free of patents.
