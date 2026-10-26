"""Print the validated configuration: ``python -m config``."""

from __future__ import annotations

import argparse
import json

from config import apply_model_cache_environment, get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m config",
        description="Validate and display the local RAG configuration (secrets are redacted).",
    )
    parser.add_argument(
        "--check-dirs",
        action="store_true",
        help="Create runtime directories and report them.",
    )
    parser.add_argument(
        "--show-cache-env",
        action="store_true",
        help="Apply model cache environment variables and show effective values.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    payload = settings.public_dict()
    meta: dict[str, object] = {
        "project_root": str(settings.project_root),
        "env_file": str(settings.project_root / ".env"),
    }

    if args.check_dirs:
        meta["created_directories"] = [str(path) for path in settings.ensure_directories()]
    if args.show_cache_env:
        meta["model_cache_environment"] = apply_model_cache_environment(settings)

    payload["_meta"] = meta
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
