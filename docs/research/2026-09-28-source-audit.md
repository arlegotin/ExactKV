# ExactKV source notes — 28 September 2026

This is the planning provenance ledger. The research map below is supplied by the user's brief, not a claim that every paper was independently retrieved in this documentation task. G0 must read the priority mechanisms/evaluations and record exact versions, sections, access status, and overlap findings before accepting a novelty claim. Recheck mutable implementation sources against installed files. No source benchmark is a local M3 measurement.

## Checks made while writing the spec

| Source | Observed | Execution consequence |
|---|---|---|
| [Official Qwen config](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json) | BF16 configuration; 28 layers, 16 Q heads, 8 KV heads, head dimension 128; position maximum 40,960. | Verify snapshot geometry and actual runtime dtype. |
| [Official model card](https://huggingface.co/Qwen/Qwen3-0.6B) | Advertises 32,768 context and documents `enable_thinking=False`. | Keep conservative 32,768 processed-position limit; greedy is an experiment, not the recommended quality setting. |
| [MLX quantize API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) | Affine group quantization supports Q4/group64, metadata same type as input, uint32 packing from low to high bits. | Assert the native ABI with fixtures, including true BF16 metadata and allocation shapes. |
| [MLX-LM cache](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py) | `QuantizedKVCache` uses packed tuples and 256-token growth increments in inspected source. | Inspect installed implementation and reserve/count actual capacity. |
| [MLX-LM attention](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/base.py) | Quantized attention uses native quantized matmuls. | Measure actual local draft cost; do not infer it from bit depth. |
| [MLX-LM Qwen3](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/qwen3.py) | Existing module includes Q/K normalization, GQA projections, and RoPE setup. | Reuse installed modules and capture the actual post-RoPE cache input. |
| [Custom Metal interface](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) | Python-defined kernel documentation available. This check did not establish a supported fast-math-disable flag. | Verify deterministic arithmetic using the installed compiler/API; do not invent an option. |
| [Vayne publisher page](https://ieeexplore.ieee.org/document/11575358/) | No usable full paper text retrieved. | Full mechanism overlap unresolved. |
| [ProofKV OpenReview page](https://openreview.net/forum?id=3Zmk1MTztm) | Redirected to browser verification; paper text inaccessible. | Full mechanism overlap unresolved. |

These checks used mutable main/docs URLs; they are navigation evidence, not runtime pins. No model weights were downloaded, MLX installed, or inference run during planning.

## Required focused audit

Read PackKV, QuantSpec, Lynx and VeriCache mechanism/evaluation sections, inspect current FAFO, and make one focused retrieval attempt for full Vayne and ProofKV papers or official code. Search titles individually when results are noisy. Record exact version/revision and evidence location for each answer to:

> Does the prior system retain an unchanged native low-bit KV view, encode the missing original BF16 bits conditional on that view, and reconstruct only bounded exact working state for verification, with all side information counted?

Do not infer absence from inaccessible text or relabel overlapping work as novelty. Reclassify direct overlap as reproduction/platform adaptation, naming any specific remaining distinction. The rest of this map bounds claims; it does not require reproducing every competitor.

## Research map retained from the brief

| ID | Primary source | Claim boundary to verify or preserve |
|---|---|---|
| P01 | [PackKV v2](https://arxiv.org/html/2512.24449v2) | Lossy quantization, packing, block residency, compute-integrated decompression already exist. |
| P01C | [PackKV native API](https://github.com/BoJiang03/PackKV/blob/master/packkv_cuda_ext/packkv_cuda.pyi) | Brief inspected blob `588df06e347b420f729794d6d6078d825835a550`; not a current pin. |
| P02 | [SplitZip v3](https://arxiv.org/html/2605.01708v3) | Exact BF16 field/palette coding is established; local palette baseline is not an official reproduction. |
| P03 | [QuantSpec v1](https://arxiv.org/html/2502.10424v1) and [Apple publication](https://machinelearning.apple.com/research/quantspec) | Hierarchical INT4 draft/INT8 target plus full-precision residual buffer; check target precision in section 4.2. |
| P04 | [Lynx v1](https://arxiv.org/html/2607.01831v1) | Progressive anchor/residual streams and speculative transfer; brief describes effective INT8 reconstruction. |
| P05 | [ExANS author article](https://www.theopenlake.com/blog/exans-lossless-gpu-compression-for-bf16-kv-cache) | Exact BF16 field/entropy coding and transfer; transfer throughput does not prove shared-view utility. |
| P06 | [VeriCache v1](https://arxiv.org/html/2605.17613v1) | Approximate drafting/full exact verification with off-device state; verifier architecture is borrowed. |
| P07 | [FAFO implementation](https://github.com/Escanord/FAFO/blob/main/README.md) | Single-model lossy guesses/full verification; brief inspected blob `453ba371f8415374b2b16556fe49361341dcaca6`. |
| P08 | [Speculative KV coding](https://fergusfinn.com/blog/kv-entropy-coder/) | Predictor-conditioned exact coding is explored; identify source precision and predictor cost. |
| P09 | [Lossless Compression of Neural Network Components v1](https://arxiv.org/html/2508.19263v1) | Low-precision component/KV redundancy already studied. |
| P10 | [DFloat11](https://arxiv.org/abs/2504.11651) | Exact floating-point compression is prior art. |
| P11 | [NeuZip](https://arxiv.org/abs/2410.20650) | Distinguish exact and lossy configurations. |
| P12 | [ZipServ v1](https://arxiv.org/html/2603.17435v1) | Runtime weight reconstruction/compute integration is established. |
| P13 | [Unweight](https://blog.cloudflare.com/unweight-tensor-compression/) | Compute-local tensor reconstruction is a broader occupied direction. |
| P14 | [ECF8 / Exponent Concentration v1](https://arxiv.org/html/2510.02676v1) | Exactness relative to a low-precision source does not imply BF16 exactness. |
| P15 | [JoLT v4](https://arxiv.org/html/2607.12550v4) | Low-rank plus quantized residuals/near-lossless quality differs from source-word equality. |
| P16 | [KIVI](https://arxiv.org/abs/2402.02750) | Lossy asymmetric KV quantization competitor. |
| P17 | [KVQuant](https://arxiv.org/abs/2401.18079) | Low-bit KV, not original BF16 recovery. |
| P18 | [TurboQuant](https://arxiv.org/abs/2504.19874) | Lossy vector quantization; native Q4 does not reproduce it. |
| P19 | [CacheGen](https://arxiv.org/abs/2310.07240) | Storage/transfer compression is distinct from active physical-memory reduction. |
| P20 | [KVTC](https://arxiv.org/abs/2511.01815) | Transforms/quantization/coding; artifact size alone is insufficient. |
| P21 | [Resident KV Claims](https://arxiv.org/abs/2605.24259) | Reusable-state scheduling/admission is orthogonal to this numeric representation. |
| P22 | [Vayne / Near-Zero Cost KV Cache Compression](https://ieeexplore.ieee.org/document/11575358/) | Partial access; DOI `10.1109/IPDPS65963.2026.00058`; no negative overlap assertion. |
| P23 | [ProofKV](https://openreview.net/forum?id=3Zmk1MTztm) | Partial access; exact greedy/refinement relevance from the brief; representation overlap unresolved. |
| P24 | [ZipNN](https://arxiv.org/abs/2411.05239) | Float-aware lossless coding is prior art. |
| P25 | [KVComp v1](https://arxiv.org/html/2509.00579v1) | Quantize then losslessly code integers is not original BF16 preservation. |
| P26 | [KeepKV v2](https://arxiv.org/html/2504.09936v2) | Single-step/multi-step consistency terminology is distinct from arbitrary word recovery. |
| P27 | [CLLA / Lossless KV Cache Compression to 2% v1](https://arxiv.org/html/2410.15252v1) | Architecture/training changes and quality equivalence do not meet this contract. |

## Implementation and data references

- MLX: [unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html), [lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), [allocation telemetry](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html). Confirm adjacent synchronization/peak/cache APIs in the installed version.
- MLX-LM: [cache source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py), [attention source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/base.py). Brief's attention blob was `b6f58f5b4ae0fb01f7a3115882a45edbeae981fe`; record actual installed hashes.
- Data: [WikiText publisher dataset](https://huggingface.co/datasets/Salesforce/wikitext), [official CPython stdlib](https://github.com/python/cpython/tree/main/Lib). Pin revisions, retain source terms/license attribution, and preserve development/held-out source disjointness.
- Independent control: [official Zstandard](https://github.com/facebook/zstd). Use page-local frames and account for framing/indexing.
