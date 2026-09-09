from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .postprocess import write_xdmf
from .solver import run_analysis
from .types import ModelError


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original ``fe-solver deck.toml`` spelling.
    if arguments and arguments[0] not in {"solve", "postprocess", "-h", "--help"}:
        arguments.insert(0, "solve")
    parser = argparse.ArgumentParser(description="finite-strain FE prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve", help="preprocess and solve a TOML analysis deck")
    solve.add_argument("deck", help="path to a TOML analysis deck")
    solve.add_argument("--stop-time", type=float, default=None)
    solve.add_argument(
        "--num-processes", type=int, default=1,
        help="persistent worker processes for complete element evaluations (default: 1)",
    )
    solve.add_argument(
        "--debug-timing", action="store_true",
        help="record detailed element, sparse-finalization, and line-search timings",
    )
    solve.add_argument(
        "--j2-backend", choices=("python", "numba"), default="python",
        help="J2 numeric kernel backend (default: interpreted Python)",
    )
    post = commands.add_parser("postprocess", help="create temporal XDMF from a run database")
    post.add_argument("database", type=Path, help="path to run.h5")
    post.add_argument("--output", type=Path, default=None, help="output .xdmf path")
    args = parser.parse_args(arguments)
    try:
        if args.command == "postprocess":
            output = write_xdmf(args.database, args.output)
            print(f"wrote {output}")
            return
        result = run_analysis(
            args.deck,
            stop_time=args.stop_time,
            num_processes=args.num_processes,
            debug_timing=args.debug_timing,
            j2_backend=args.j2_backend,
        )
    except ModelError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"converged {len(result.increments)} increments to t={result.t:.12g}; "
        f"max|u|={abs(result.u).max():.6g}; "
        f"force balance={result.verification['force_balance_inf']:.3e}"
    )


if __name__ == "__main__":
    main()
