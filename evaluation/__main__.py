"""Evaluation CLI: ``python -m evaluation``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import get_settings
from evaluation.runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Run the local-first RAG evaluation harness.",
    )
    parser.add_argument("--golden-set", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--fail-on-regression", action="store_true", default=True)
    parser.add_argument("--no-fail-on-regression", dest="fail_on_regression", action="store_false")
    parser.add_argument("--json", action="store_true", help="Print the report JSON to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    try:
        report = run_evaluation(
            settings=settings,
            golden_set_path=args.golden_set,
            output_dir=args.output,
            fail_on_regression=args.fail_on_regression,
            k=args.k,
        )
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "passed": report.passed,
                    "metrics": report.metrics,
                    "failures": report.failures,
                    "output": str(report.output_path),
                },
                indent=2,
            )
        )

    print(f"Report: {report.output_path}")
    print(f"Passed: {report.passed}")
    if report.failures:
        for failure in report.failures:
            print(f"  - {failure}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
