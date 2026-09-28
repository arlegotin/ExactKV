# ExactKV findings — G1 size no-go

The CPU conditional interval-rank codec reconstructed every original BF16 word on the measured real-cache pages, but its fully counted shared prompt view saved too little memory to pay for exact layer staging. G1 stopped the runtime branch. This is an informative early no-go for the measured format and inputs, not a proof that all conditional KV codecs fail.

Evidence: [G0 decision](decisions/G0.json), [G1 stop decision](decisions/G1.json), [all-layer raw size record](evidence/G1/size-512.json), and [source audit](../docs/research/2026-09-28-source-audit.md). The G1 decision hashes the size record and its producing code; this report refuses changed evidence.

Machine and model: Apple M3 Max, 36 GiB physical RAM; Python 3.12.9; MLX 0.32.2; MLX-LM 0.31.3; original Qwen/Qwen3-0.6B BF16 snapshot `c1899de289a04d12100db370d81485cdf75e47ca`. The prompt length is the final chat-template token count, including non-thinking instructions.

## Exactness and measured storage

E1 passed in the CPU reference: 21,504 real 4,096-value pages across all 28 layers, 8 KV heads, both K and V, and the listed development prompts. The conditional codec, independent palette, field-split Zstd, and XOR control each decoded to the original uint16 BF16 words on every page. Synthetic tests also exercise every 16-bit BF16 pattern. E2 paired-verifier equality was not tested. E3 stock greedy equivalence was not tested.

All numbers in the next table are measured prompt MiB. Q4 includes both packed codes and native BF16 scales/biases. Conditional bytes are the complete version-one EKVT serialization: literal groups, EKVR page headers, modes, restart offsets, alignment, guards, and the tensor-container header, manifest, source-side hash, tensor descriptors, and page-offset index. Independent controls are measured page-local code streams; their tensor indexes are excluded, so their figures are lower bounds. Independent exact-only columns intentionally omit Q4.

| Development prompt | Raw BF16 | Q4 codes + S/B | Q4 + conditional | Q4 + palette | Q4 + field-Zstd | Palette only | Field-Zstd only |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev-code-512 | 56.00 | 15.75 | 55.18 | 59.80 | 55.94 | 44.05 | 40.19 |
| dev-prose-512 | 56.00 | 15.75 | 55.30 | 59.87 | 56.15 | 44.12 | 40.40 |
| dev-structured-512 | 56.00 | 15.75 | 55.22 | 59.79 | 55.94 | 44.04 | 40.19 |

The EKVT container contributes 0.09 MiB per prompt. The page-local field-Zstd control is a CPU size comparator; it has no GPU decoder or fair inference runtime result. The simple palette is likewise a local baseline, not a reproduction of SplitZip.

## Projected working bytes and decision

These figures add the measured serialized prompt to a 28.00 MiB authoritative BF16 tail, 0.98 MiB verifier-candidate workspace, 15.75 MiB additional native Q4 tail capacity, and one or two 2.00 MiB layer-sized BF16 stages at 512 tokens. The Q4 reservation covers 256 outputs plus 9 verifier positions and rounds to the installed 256-token allocation step; raw exact execution needs only its authoritative tail. These are **projected cache-related bytes**, not process high-water marks; other transient copies and allocator effects could increase them. No maximum fitting context was measured. The G1 capture deliberately retained a full raw cache while diagnosing each layer, so it cannot establish final-system physical peak savings.

| Development prompt | Conditional + one stage + tail | Conditional + two stages + tail | Raw BF16 + tail |
|---|---:|---:|---:|
| dev-code-512 | 101.92 | 103.92 | 84.00 |
| dev-prose-512 | 102.04 | 104.04 | 84.00 |
| dev-structured-512 | 101.96 | 103.96 | 84.00 |

The table below applies the same one/two-stage and tail schedules to independent controls. Its controls omit a tensor-level index, making their bytes optimistic lower bounds. The exact-only controls have no native Q4 tail, but would need to decode for ordinary exact attention; that runtime was not measured. Ratios use the raw exact-plus-tail bytes for the same prompt.

| Development prompt | Representation | One stage MiB | Two stages MiB | One-stage ratio to raw exact | Two-stage ratio to raw exact |
|---|---|---:|---:|---:|---:|
| dev-code-512 | Q4 + palette | 106.53 | 108.53 | 1.27× | 1.29× |
| dev-code-512 | Q4 + field-Zstd | 102.68 | 104.68 | 1.22× | 1.25× |
| dev-code-512 | Palette exact only | 74.05 | 76.05 | 0.88× | 0.91× |
| dev-code-512 | Field-Zstd exact only | 70.19 | 72.19 | 0.84× | 0.86× |
| dev-prose-512 | Q4 + palette | 106.60 | 108.60 | 1.27× | 1.29× |
| dev-prose-512 | Q4 + field-Zstd | 102.88 | 104.88 | 1.22× | 1.25× |
| dev-prose-512 | Palette exact only | 74.12 | 76.12 | 0.88× | 0.91× |
| dev-prose-512 | Field-Zstd exact only | 70.40 | 72.40 | 0.84× | 0.86× |
| dev-structured-512 | Q4 + palette | 106.52 | 108.52 | 1.27× | 1.29× |
| dev-structured-512 | Q4 + field-Zstd | 102.68 | 104.68 | 1.22× | 1.25× |
| dev-structured-512 | Palette exact only | 74.04 | 76.04 | 0.88× | 0.91× |
| dev-structured-512 | Field-Zstd exact only | 70.19 | 72.19 | 0.84× | 0.86× |

Across 3 development prompts, Q4 plus conditional refinement occupied 55.18–55.30 MiB of the 56.00 MiB raw prompt. The mandatory exact stage exceeded the 0.70–0.82 MiB prompt saving even in an optimistic projection that omitted native draft and candidate tails: 85.18–85.30 MiB against 84.00 MiB raw exact. Removing every refinement metadata byte (1.35 MiB per prompt) from that optimistic case would leave at most 0.16 MiB one-stage headroom and would still lose with two stages. The specified-buffer projection is worse still. The proposed conditional format therefore has no credible measured memory advantage worth building a Metal decoder for. A longer context or different model may have different statistics; none was measured after this stop.

Process peak and MLX allocator peak were not measured for G1; projected cache bytes are not physical high-water marks.

G2–G6: not run. Committed throughput was not measured. Whole-request time, acceptance, output-burst latency, verifier correctness, CPU-to-Metal arithmetic agreement, and physical process peak were also not measured. There is no end-to-end speedup or usable verified-decoding claim. The low-bit Q4-only path is a lossy lower-memory choice and has a different correctness contract.

## Prior art and scope

[PackKV](https://arxiv.org/html/2512.24449v2) already combines lossy quantization, packing, and compute-integrated decompression. [QuantSpec](https://arxiv.org/html/2502.10424v1) and [Lynx](https://arxiv.org/html/2607.01831v1) establish hierarchical low-bit views and speculative refinement to their integer verification targets. [VeriCache](https://arxiv.org/html/2605.17613v1) and [FAFO](https://github.com/Escanord/FAFO/blob/main/README.md) establish approximate-cache proposals with exact verification. [SplitZip](https://arxiv.org/html/2605.01708v3) establishes independent exact BF16 field/palette coding. The narrowed conditional original-BF16 representation remains only a provisional distinction: full [Vayne](https://ieeexplore.ieee.org/document/11575358/) and [ProofKV](https://openreview.net/forum?id=3Zmk1MTztm) mechanisms could not be retrieved, so their overlap is unresolved. The source audit records the specific reviewed sections and access limits. This report makes no global novelty or patent-clearance claim.

## Reproduction and limits

The [README](../README.md) pins the environment, checkpoint, source partitions, small exactness tests, and explicit three-prompt G1 command. Re-run `python -m kvrefine.cli report --results results --out results/report.md` to regenerate this report after validating the committed evidence. The committed G1 artifact contains per-layer/head, K/V, mode, rank-width, and full serialized-byte details. The diagnostic capture and CPU packing are not a bounded construction implementation; no physical memory saving should be inferred from their temporary allocations. A fresh independent person's reproduction has not yet been recorded.
