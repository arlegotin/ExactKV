# ExactKV

ExactKV investigates whether an unchanged native MLX Q4 KV cache can also support bit-exact recovery of its original BF16 words through a compact conditional refinement stream. The experiment targets one M3 Max and reconstructs one layer at a time for verification.

The repository currently contains the specification and implementation plan. Runtime code, environment locks, model downloads, and benchmark results have not been produced. A justified early no-go is an intended research outcome.

- [Design specification](docs/superpowers/specs/2026-09-28-exactkv-design.md): codec, page format, ownership, verification, baselines, gates, and completion criteria.
- [Implementation plan](docs/superpowers/plans/2026-09-28-exactkv.md): 18 tasks with interfaces, tests, evidence requirements, and meaningful commits.
- [Source audit notes](docs/research/2026-09-28-source-audit.md): checked implementation sources, supplied prior-art map, and unresolved overlap risks.

Execution starts with environment/native-layout checks, then three small real-data probes and a raw verification timing control. Metal decoding and held-out benchmarks depend on those results. Commands in the plan describe interfaces to implement; they are not runnable yet.

Planning inspection found an Apple M3 Max with 36 GiB RAM and ARM-native Python 3.12.9 available. MLX compatibility, the safe application budget, compressibility, and runtime utility remain to be measured. Work stays on the current branch.
