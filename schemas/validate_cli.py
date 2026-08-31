"""Deterministic command-line validation for tuner JSON boundaries."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from schemas.document_io import load_candidate_set, load_job


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    job_parser = subparsers.add_parser("job", help="validate one JobSpec JSON file")
    job_parser.add_argument("path", type=Path)
    candidate_parser = subparsers.add_parser(
        "candidates", help="validate a complete candidate set against its JobSpec"
    )
    candidate_parser.add_argument("job_path", type=Path)
    candidate_parser.add_argument("candidate_path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "job":
            load_job(args.path)
            return 0
        if args.command == "candidates":
            job = load_job(args.job_path)
            load_candidate_set(args.candidate_path, search=job.search)
            return 0
        raise AssertionError(f"unhandled command: {args.command}")
    except ValueError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
