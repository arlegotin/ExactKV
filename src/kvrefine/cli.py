"""Small explicit commands for the currently implemented gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .doctor import collect_environment
from .records import write_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kvrefine", description="ExactKV gated research commands")
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser("doctor", help="Record local platform and package capability")
    doctor.add_argument("--out", type=Path, required=True)
    doctor.add_argument("--smoke", action="store_true")
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
