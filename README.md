# ExactKV

ExactKV investigates whether an unchanged native MLX Q4 KV cache can also support bit-exact recovery of its original BF16 words through a compact conditional refinement stream. The experiment targets one M3 Max and reconstructs one layer at a time for verification.

The specification and implementation plan are being executed gate by gate. G0 environment diagnostics are available; native-cache, model, codec, and benchmark results remain to be measured. A justified early no-go is an intended research outcome.

- [Design specification](docs/superpowers/specs/2026-09-28-exactkv-design.md): codec, page format, ownership, verification, baselines, gates, and completion criteria.
- [Implementation plan](docs/superpowers/plans/2026-09-28-exactkv.md): 18 tasks with interfaces, tests, evidence requirements, and meaningful commits.
- [Source audit notes](docs/research/2026-09-28-source-audit.md): checked implementation sources, supplied prior-art map, and unresolved overlap risks.

Run the current small suite in an ARM-native Python 3.12 environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m kvrefine.cli doctor --out results/environment.json
.venv/bin/python -m pytest -q
```

`doctor` exits with code 4 when a critical measurement is unavailable; its JSON still records the reason. The other commands in the plan will appear as their gates are implemented. Metal decoding and held-out benchmarks depend on earlier gate results.

Planning inspection found an Apple M3 Max with 36 GiB RAM and ARM-native Python 3.12.9 available. MLX compatibility, the safe application budget, compressibility, and runtime utility remain to be measured. Work stays on the current branch.
