<!-- Copied 2026-11-16 by the ml-convertant lane from the coderipper-b lane's record
     C:\Projects\coderipper-wt-pkg-b\IQ-PATENT-SEARCH-2026-11-16.md (untracked there; routed by the PM, message pm@ciresnave/tof1tt-1g64-o3).
     The text below is that record unchanged except the final section, which is ml-convertant's own reading and is labelled. -->

# IQ* patent search for ml-convertant (database/assignee pass), 2026-11-16

Run by: coderipper-b lane, Claude Sonnet 5.5. AI-assisted search and read; NOT legal advice; NOT a freedom-to-operate opinion.
Scope: the PM's [TASK] patent-search-IQ, on top of ml-convertant `docs/PATENT-SWEEP.md` at branch `iq-sweep` (PR #37, fa3201a), whose three web queries W8 to W10 were not repeated.
Format: the hits table of `PATENT-SWEEP-PROCESS.md` (record template). I could not find a record named WSP-013 in any `docs/` I could read, so "same format as WSP-013" is not verified.

## What I could and could not access (stated first)
- **Searched:** Google Patents through a web search restricted to `patents.google.com` (and, in one query, `worldwide.espacenet.com`), 9 queries below. Claims read with `tools/patent-claims.py` (Google Patents' public page, machine text).
- **NOT accessible:** Google Patents' own search page with a CPC filter (the page loads but is JavaScript-rendered: nothing to read); Espacenet (HTTP 403 to a plain request); USPTO, WIPO and CPC-browser full text. **So no CPC-class search (G06N 3/0495, H03M 7/30) was actually run**: those codes were written into one query's words only, and a web search does not filter by class. **No assignee search was run in a patent database**: the assignee queries below are web searches that name the assignee or inventors. Treat both as not done.
- A web search returns what a search engine ranks, not a complete result set; a miss is a statement about that query.

## Control (the absence claims below are only as good as this)
Q1 (control) was written from US11348009B2's own language ("non-uniform quantization of pre-trained deep neural network range of weights master bit-width squarer circuit"). It returned US 11348009 and its continuation US 11710043, so the engine reaches known patents in this neighbourhood. Run the same day, same engine as Q2 to Q9. The control proves the query path works, not that Q2 to Q9 ask the right question.

## Queries (all 2026-11-16, domain filter patents.google.com unless noted)
| # | Topic | Result |
|---|---|---|
| Q1 | control, US11348009B2 | found US 11348009, US 11710043 (family), US 11775611 |
| Q2 | lattice / E8 codebook vector quantization of LLM weights | papers (QuIP#, Leech-lattice VQ, grouped lattice VQ); patents US5883981A, US6728413B2 (image/video lattice VQ, not weights); **US 12524655 (neural-network weights, lattice VQ)** |
| Q3 | 4-bit non-uniform table of sixteen values with block scale | any4 paper; WO2024152124A1, WO2019080483A1 (read below); no patent on a 16-entry weight table |
| Q4 | importance-matrix / activation-weighted quantization error | papers only (llama.cpp imatrix users); no patent |
| Q5 | learned per-row 4-bit lookup table by clustering (any4 and relatives) | papers (any4, AAAC, SqueezeLLM); no patent |
| Q6 | GANQ-style GPU-adaptive non-uniform LUT quantization | GANQ paper; patents returned were unrelated (LLR quantization, video) |
| Q7 | Cornell / QuIP# incoherence + lattice codebook (assignee by name) | papers (QuIP#, QTIP); **no patent**; the engine does not show inventor or assignee for the papers |
| Q8 | Meta / any4 per-row codebook (assignee by name) | no Meta patent; EP3612989B1 (flexible hardware for vector dequantization) listed, not read |
| Q9 | CPC wording G06N 3/0495, H03M 7/30 + codebook indices, with espacenet.com allowed | US 12524655 again; US 12149263 (soft vector quantization) listed, not read |

## Hits read to claim level (independent claims read in full from `patent-claims.py`)
| patent | title | assignee | status (Google Patents, not the official record) | claims read | overlap with IQ* | our difference | verdict |
|---|---|---|---|---|---|---|---|
| US11348009B2 (control) | Non-uniform quantization of pre-trained DNN | Samsung | active, est. expiry 2041-03-01 | 1 (all) read earlier in ml-convertant H23 | none new | no bit-width chosen per layer or channel by a squared-range noise measure | unchanged: clear as read in H23 |
| US11710043B2 | same title (continuation, filed 2022-05-13) | Samsung | active | independent 1, 7, 13 in full | claim 13: for a layer or channel pick, from a set of master bit-widths, the one with minimum quantization noise (circuit steps) | ml-convertant fixes the format per type and never chooses a bit-width per layer or channel from measured noise; the IQ* types are fixed tables | clear as read; **same re-read trigger as H23: choosing a format or bit-width per tensor from measured error** |
| US11775611B2 | Piecewise quantization for neural networks | Samsung | active | independent 1 and 11 in full | weights stored with an index saying the type of quantizing applied | claim needs an activation buffer, selector, per-column **multiplier, shift register and adder tree** (a hardware dot-product); we convert stored weights offline and run no multiply | clear as read (outside on the multiplier/adder limitation, as for H1 and H2) |
| WO2024152124A1 | Lookup tables for ultra low-bit operations | Deeplite Inc | ceased | claim 1 (method) read | quantized neural network plus a lookup table | claim 1 is **convolution via a lookup table with packed weights and activations unpacked by bitwise ops**; an inference method, not a weight format | clear as read; ceased by the engine's status |
| WO2019080483A1 | NN computation acceleration with non-uniform quantization and lookup table | Beijing Deephi | ceased | claim 1 (machine translation) read | non-uniform quantization of each layer's parameters | claim 1 also quantizes each layer's **inputs**, builds a table of all weight-by-input products and looks the **multiplication** up in the forward pass; we do no inference | clear as read; ceased |
| **US12524655B2** | Artificial neural network processing methods and system | STMicroelectronics | **active** | independent claim 1 read (to the "provided ... to a neural processing" limitation) | **closest hit.** Weights of a layer are vectorized into weight vectors, normalized, put through **lattice vector quantization producing a codebook of codewords**, and the codebook is indexed "as a function of the lattice" into tuples of indices | claim 1 requires an ANN with an activation-function parameter, a lattice of chosen dimension from basis vectors, normalization of the vector matrix, LVQ producing a codebook and the three-index encoding for a neural processing unit. IQ3_S and IQ2_* are fixed published grid tables (ggml, MIT) with signs and a block scale, not a lattice built from basis vectors, no normalization, no index tuple for an NPU | **needs-review as to IQ3_S / IQ2_* / IQ3_XXS only**, not cleared: a grid-point codebook of weight groups is the neighbourhood of this claim; I read claim 1 only and not its dependents or its other independents |

## Named, unread
- US12524655B2: the other independent claims and all dependents.
- EP3612989B1 (flexible hardware, vector dequantization with codebook size), US12149263 (soft vector quantization): listed by the engine, claims not read.
- Any patent on QuIP#, QTIP, GLVQ, any4 or GANQ held by their authors' institutions: **none found by name in web search; not searched in a patent database** (see access limits). This is not evidence that none exist.
- Patents on importance-matrix error weighting for scale choice: none found in Q4; the engine returned only papers.

## Result
No patent was found on the fixed 4-bit tables (`IQ4_NL`, `IQ4_XS`) or on importance-weighted scale choice by these queries. **One hit needs review for the grid-codebook types (`IQ3_S`, later `IQ2_*`, `IQ3_XXS`): US12524655B2 (STMicroelectronics, active).** Two Samsung families keep the existing re-read trigger. The CPC and assignee parts of the task were NOT done as stated (no database access); to do them, someone needs a Google Patents or Espacenet session that can run a CPC-filtered search.

## Limits
Web search only; machine-converted claim text (Chinese claim read in Google's translation); one claim read for US12524655B2; no citation chasing; status is Google Patents' estimate; AI-assisted; not legal advice.

## ml-convertant's reading of US12524655B2's independent claims (2026-11-16, lane ml-convertant; AI-assisted; NOT legal advice)

Read with `tools/patent-claims.py` (Google Patents machine text), independent claims 1, 8, 16 and 18 in full, and the opening of the dependents. Against our dequantize and
convert-only use of `IQ3_S` (ggml's fixed `iq3s_grid` plus signs and a block scale):

- **All four independent claims require the same optimization.** Claims 1 and 8 (method, device) produce the weights by "performing the normalization processing ... comprises
  solving an optimization problem having: a first term to provide normalized weight values which approximate the at least one output activation parameter (b_j) of the ANN as a
  function of the at least one input activation parameter (a_j) ... and a regularization term to amplify normalized weight values based on distances of weight values from the lattice
  points of the selected lattice". Claims 16 and 18 (system) recite stored compressed weights that "correspond to a solution to" that same problem and then decompress them to perform an ANN operation.
- Claims 1 and 8 also require a lattice set by a dimensional parameter and basis vectors, vectorization into a matrix, normalization, a codebook of codewords produced by lattice vector quantization, and tuples of indices (i_s, i_a, i_a LUT).
- **Our difference:** ggml's `IQ3_S` encoder (which we would reproduce) picks, for groups of weights, the nearest entry of a fixed published grid table with signs and a block scale, using only the
  weights (optionally weighted by an importance vector). There is no objective that approximates output activations from input activations, no activation-function parameter, no
  regularizer on distance to lattice points, no lattice defined by basis vectors, no normalization optimization, and no neural processing circuit or accelerator: this repository dequantizes and converts stored bytes offline.
- **Reading:** outside as read on independent claims 1, 8, 16 and 18, each of which needs the activation-approximating optimization with the lattice-distance regularizer. Not read: dependents beyond their openings, the prosecution file, any
  continuation or family member, any foreign counterpart. **Re-read trigger:** an encoder that solves an activation-approximating objective with a lattice-distance regularizer, or a lattice built from basis vectors, or an accelerator path.
- This lifts the PM's hold on `IQ3_S` only as far as this reading goes; the PM decides whether it is enough.
