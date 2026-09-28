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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
