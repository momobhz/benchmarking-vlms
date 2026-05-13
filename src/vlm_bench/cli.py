from __future__ import annotations

import argparse
import sys

from vlm_bench.commands import run_analyze_config, run_curate_config
from vlm_bench.config import apply_overrides, load_config
from vlm_bench.eval.runner import run_eval_config


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to a YAML workflow config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config value with dotted key syntax, e.g. --set data.max_samples=20.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vlm-bench", description="Run VLM benchmark workflows from config files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Run or dry-run a VLM evaluation config.")
    _add_config_args(eval_parser)
    eval_parser.add_argument("--print-command", action="store_true", help="Print the resolved evaluation command.")

    curate_parser = subparsers.add_parser("curate", help="Run or dry-run a data curation config.")
    _add_config_args(curate_parser)

    analyze_parser = subparsers.add_parser("analyze", help="Run or dry-run an embedding/plot analysis config.")
    _add_config_args(analyze_parser)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = apply_overrides(load_config(args.config), args.overrides)

        if args.command == "eval":
            return run_eval_config(config, dry_run=args.dry_run, print_command=args.print_command)
        if args.command == "curate":
            return run_curate_config(config, dry_run=args.dry_run)
        if args.command == "analyze":
            return run_analyze_config(config, dry_run=args.dry_run)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
