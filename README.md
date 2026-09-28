# ExactKV

ExactKV investigates whether an unchanged native MLX Q4 KV cache can also support bit-exact recovery of its original BF16 words through a compact conditional refinement stream. The experiment targets one M3 Max and reconstructs one layer at a time for verification.

The specification and implementation plan are being executed gate by gate. G0 capability checks pass on the measured machine, with novelty explicitly provisional. Codec size and benchmark results remain to be measured. A justified early no-go is an intended research outcome.

- [Design specification](docs/superpowers/specs/2026-09-28-exactkv-design.md): codec, page format, ownership, verification, baselines, gates, and completion criteria.
- [Implementation plan](docs/superpowers/plans/2026-09-28-exactkv.md): 18 tasks with interfaces, tests, evidence requirements, and meaningful commits.
- [Source audit notes](docs/research/2026-09-28-source-audit.md): checked implementation sources, supplied prior-art map, and unresolved overlap risks.

Run the current small suite in an ARM-native Python 3.12 environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m kvrefine.cli doctor --out results/environment.json
.venv/bin/python -m pytest -q
.venv/bin/python -m kvrefine.cli doctor --smoke --out results/environment.json
.venv/bin/python -m pytest --run-metal -m 'not model and not slow' -q
```

`doctor` exits with code 4 when a critical measurement is unavailable; its JSON still records the reason. The smoke and native tests require Metal access. The original BF16 model is pinned in `data/model-lock.json`; model weights are downloaded locally and deliberately ignored by Git. To re-resolve the official checkpoint in a fresh checkout, run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from kvrefine.model import resolve_snapshot
from kvrefine.records import write_record
write_record(Path('data/model-lock.json'), resolve_snapshot('Qwen/Qwen3-0.6B'))
PY
.venv/bin/python -m pytest tests/test_model_identity.py --run-model -m model -q
```

Check the newly resolved revision before replacing a committed lock: a changed upstream checkpoint is a new experimental configuration. The model test checks hashes, original BF16 weights, post-RoPE cache dtype, and valid K/V geometry. The other commands in the plan will appear as their gates are implemented; Metal decoding and held-out benchmarks depend on earlier gate results.

G0 measured an Apple M3 Max with 36 GiB RAM, ARM-native Python 3.12.9, MLX 0.32.2, and MLX-LM 0.31.3. The safe application budget, compressibility, and runtime utility remain to be measured. Work stays on the current branch.
