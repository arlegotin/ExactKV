"""Small explicit commands for the currently implemented gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .doctor import collect_environment
from .gates import require_gate
from .records import write_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kvrefine", description="ExactKV gated research commands")
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser("doctor", help="Record local platform and package capability")
    doctor.add_argument("--out", type=Path, required=True)
    doctor.add_argument("--smoke", action="store_true")
    data = subcommands.add_parser("data", help="Build pinned, disjoint prompt manifests")
    data.add_argument("--split", choices=("dev", "heldout"), required=True)
    data.add_argument("--tokens", type=int, default=512)
    data.add_argument("--out", type=Path, required=True)
    probe = subcommands.add_parser("probe", help="Measure all-layer exact and native Q4 page sizes")
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--tokens", type=int, required=True)
    probe.add_argument("--out", type=Path, required=True)
    report = subcommands.add_parser("report", help="Render an evidence-checked findings report")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if args.command == "doctor":
        report = collect_environment(smoke=args.smoke)
        write_record(args.out, report)
        print(json.dumps({"out": str(args.out), "ram_bytes": report["ram_bytes"], "smoke": report["smoke"]}))
        if report["critical_issues"]:
            return 4
        return 0
    if args.command == "data":
        from .data import build_manifest

        require_gate("G0", Path("results/decisions"))
        sources = json.loads(Path("data/sources.json").read_text())
        model = json.loads(Path("data/model-lock.json").read_text())
        records = build_manifest(sources, model, args.split, args.out, tokens=args.tokens)
        print(json.dumps({"out": str(args.out), "prompts": len(records), "split": args.split, "tokens": args.tokens}))
        return 0
    if args.command == "probe":
        from .probe import run_probe

        out = args.out if args.out.suffix == ".json" else args.out / "probe.json"
        try:
            result = run_probe(args.manifest, args.tokens)
        except Exception as exc:
            write_record(out, {"schema_version": 1, "kind": "g1-failure", "reason": str(exc), "error_type": type(exc).__name__})
            raise
        write_record(out, result)
        print(json.dumps({"out": str(out), "prompts": len(result["prompts"]), "tokens": args.tokens}))
        return 0
    if args.command == "report":
        from .report import build_report

        summary = build_report(args.results, args.out)
        print(json.dumps({"out": str(args.out), "conclusion": summary["conclusion"], "exact_cpu_pages": summary["exact_cpu_pages"]}))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
