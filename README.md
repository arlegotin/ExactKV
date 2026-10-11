# ExactKV

An LLM keeps an attention cache of earlier tokens. Quantizing that cache saves memory, but changes its values. ExactKV tests whether a native 4-bit cache can also carry a small refinement stream that recovers **every original BF16 bit** when verification needs it.

**Exact recovery worked on all 21,504 tested cache pages. The memory budget didn't.** The combined representation saved less than 1 MiB on a 56 MiB prompt, then needed more than that saving just to reconstruct one layer. The experiment stopped before building a GPU decoder.

The useful result is a working exact codec and a measured account of where its apparent compression gain disappears during inference.

## The idea

Keep one prompt in two views: the unchanged native 4-bit cache for draft tokens, and an exact BF16 view reconstructed from those codes plus refinement data for verification. If the refinement is compact enough, this could avoid retaining a separate full-size BF16 prompt cache.

The CPU codec stores the original BF16 word's rank within an interval conditioned on its quantization code. Pages that don't compress use literal fallback. It preserves unusual bit patterns too; synthetic tests cover every 16-bit BF16 pattern.

## What the experiment found

The size probe used Qwen3-0.6B on an Apple M3 Max with 36 GiB RAM: three 512-token development prompts containing prose, code, and structured text. Exact reconstruction passed across all 28 layers, both keys and values, and all eight KV heads.

| Representation | Measured prompt storage | Projected working cache |
|---|---:|---:|
| Original BF16 | 56.00 MiB | 84.00 MiB |
| Native 4-bit, lossy | 15.75 MiB | — |
| 4-bit + ExactKV refinement | 55.18–55.30 MiB | 101.92–102.04 MiB |
| Independent exact field-Zstd control | 40.19–40.40 MiB* | 70.19–70.40 MiB* |

The working-cache column adds a 256-token output tail. ExactKV also needs verifier candidates, reserved 4-bit tail capacity, and a reconstructed BF16 layer. These are **projections from measured serialized bytes**, not measured process-memory peaks. *The field-Zstd control omits its tensor index and has no measured GPU decoding time.*

ExactKV's prompt numbers count the complete container, headers, alignment, and indexes. Even an optimistic projection that omits draft-tail and candidate buffers loses to raw BF16 once one exact layer is staged. A page-local independent exact codec also stored the prompt in fewer bytes. Its runtime remains untested.

That's enough evidence to stop this format at the size gate. There is **no Metal decoder, integrated verifier, end-to-end speed result, or demonstrated physical-memory saving**. Longer contexts, other models, and different refinement formats weren't evaluated after the stop.

[Full report](results/report.md) · [Size evidence](results/evidence/G1/size-512.json) · [Stop decision](results/decisions/G1.json)

## Inspect the codec

Use native ARM Python 3.12. The ordinary suite checks the CPU codec and protocol without downloading a model; GPU and model tests are opt-in.

```bash
git clone https://github.com/arlegotin/ExactKV.git
cd ExactKV
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q
```

The [codec](src/kvrefine/codec.py), [container format](src/kvrefine/format.py), and [exact baselines](src/kvrefine/baselines.py) are the reusable pieces. The [design](docs/superpowers/specs/2026-09-28-exactkv-design.md) describes the intended draft/verification workflow; later runtime stages remain unimplemented. The [source audit](docs/research/2026-09-28-source-audit.md) records prior art and unresolved overlap. No global novelty claim is made.

## Reproduce the size probe

This downloads the pinned original-BF16 checkpoint and public source text, then runs the model. Allow at least 3 GiB of free disk space. Model and input revisions are in [data/model-lock.json](data/model-lock.json) and [data/sources.json](data/sources.json).

```bash
.venv/bin/python -m kvrefine.cli doctor --out results/runs/doctor.json

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

.venv/bin/python -m kvrefine.cli data --split dev --tokens 512 --out data/manifest.jsonl
.venv/bin/python -m kvrefine.cli data --split heldout --tokens 512 --out data/manifest.jsonl
.venv/bin/python -m kvrefine.cli probe --manifest data/manifest.jsonl --tokens 512 --out results/runs/g1-512
.venv/bin/python -m pytest --run-metal --run-model -m 'not slow' -q
```

The probe writes new output under ignored `results/runs/`. The data commands regenerate the pinned manifest; they don't constitute a held-out evaluation. The reported size result uses only the three development prompts.

A future attempt should first demonstrate a refinement small enough to pay for reconstruction on real cache tensors. Only then is a GPU decoder and full-request benchmark worth building.

## Credit and reuse

By **Artem Legotin**. Original code is licensed under [AGPL-3.0-only](LICENSE); original prose and reports under [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Preserve author credit when sharing covered material. [Licensing scope](LICENSING.md) and [third-party notices](THIRD_PARTY_NOTICES.md) retain the separate terms for source excerpts, models, and dependencies.

If you build on the codec, measurements, or ideas, please cite [CITATION.cff](CITATION.cff), link this repository, and identify the commit you used. Citation of independently reused ideas is a scholarly request, separate from the license obligations.
