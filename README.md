# ExactKV

**Result: exact reconstruction worked, but this version did not save useful memory.** ExactKV tests a way to keep an LLM's attention history as an unchanged native 4-bit KV cache plus a compact stream that recovers the original BF16 values. The intended 4-bit view would support fast draft tokens; the exact view would support verification without a separate full-size BF16 prompt cache.

A KV cache holds information from previous tokens. Compressing it can make longer contexts affordable, but ordinary 4-bit quantization changes its values. ExactKV asks whether the already-stored 4-bit codes can help recover the *original bits* cheaply enough to support verified decoding with less memory.

## What the experiment found

On an M3 Max with 36 GiB RAM, we tested three 512-token Qwen3-0.6B prompts: prose, code, and structured text. The CPU codec recovered all **21,504** tested cache pages bit-for-bit. The space saving, however, was too small to pay for the memory needed to reconstruct a layer during verification.

| Cache option | Stored prompt, measured | Working cache, projected |
|---|---:|---:|
| Original BF16 | 56.00 MiB | 84.00 MiB |
| Native 4-bit only (lossy) | 15.75 MiB | — |
| 4-bit + ExactKV refinement | 55.18–55.30 MiB | 101.92–102.04 MiB |
| Independent exact field-Zstd | 40.19–40.40 MiB* | 70.19–70.40 MiB* |

The working-cache estimates include a 256-token output tail; ExactKV also needs verifier candidates, reserved 4-bit tail capacity, and one reconstructed BF16 layer. They are **projections from measured serialized bytes**, not physical process peaks. *The field-Zstd control is page-local, omits its tensor index, and has no measured GPU decode time.* See the [full findings and per-prompt results](results/report.md).

**What worked:** the native 4-bit view stayed usable, and the CPU refinement recovered the original BF16 words exactly, including unusual bit patterns through literal fallback.

**What did not:** the combined prompt saved only about 0.7–0.8 MiB against raw BF16, while its required working buffers cost more. A simple independent exact codec also stored the prompt in fewer bytes, though its runtime was not tested. We stopped at the size gate; there is no Metal decoder, integrated verifier, end-to-end speed benchmark, or measured physical-memory saving. This result applies to this format and these prompts, not to every possible exact-refinement design.

## Where to go next

- Try a substantially smaller refinement representation on real KV tensors before investing in GPU decoding.
- If memory alone is the goal, compare an independent exact cache with a practical decoder; the field-Zstd result here measures size only.
- Only after a promising size result, implement bounded layer reconstruction and measure full-request time, verified output behavior, and physical peak memory on longer prompts.

## Reproduce the small result

Use an ARM-native Python 3.12 environment. The lockfile pins the working packages; the model and input revisions are recorded in [data/model-lock.json](data/model-lock.json) and [data/sources.json](data/sources.json).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m kvrefine.cli doctor --out results/runs/doctor.json
```

Download the pinned original-BF16 checkpoint and verify its file hashes (allow at least 3 GiB free disk space):

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from kvrefine.model import SNAPSHOT_PATTERNS, load_model
lock = json.loads(Path('data/model-lock.json').read_text())
snapshot_download(lock['model_id'], revision=lock['revision'],
                  local_dir=lock['local_dir'], allow_patterns=list(SNAPSHOT_PATTERNS))
load_model(lock)
PY
```

Regenerate the pinned prompts, then run the **three-prompt CPU size probe** deliberately; it downloads public source text and runs the model. The output goes under ignored `results/runs/`, leaving committed evidence intact.

```bash
.venv/bin/python -m kvrefine.cli data --split dev --tokens 512 --out data/manifest.jsonl
.venv/bin/python -m kvrefine.cli data --split heldout --tokens 512 --out data/manifest.jsonl
.venv/bin/python -m kvrefine.cli probe --manifest data/manifest.jsonl --tokens 512 --out results/runs/g1-512
.venv/bin/python -m pytest --run-metal --run-model -m 'not slow' -q
```

The [design](docs/superpowers/specs/2026-09-28-exactkv-design.md), [implementation plan](docs/superpowers/plans/2026-09-28-exactkv.md), [source audit](docs/research/2026-09-28-source-audit.md), and [committed G1 evidence](results/evidence/G1/size-512.json) contain the technical details and limits of the conclusion.
