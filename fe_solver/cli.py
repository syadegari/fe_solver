from __future__ import annotations

import argparse

from .solver import run_analysis
from .types import ModelError


def main() -> None:
    parser = argparse.ArgumentParser(description="finite-strain FE prototype")
    parser.add_argument("deck", help="path to a TOML analysis deck")
    parser.add_argument("--stop-time", type=float, default=None)
    args = parser.parse_args()
    try:
        result = run_analysis(args.deck, stop_time=args.stop_time)
    except ModelError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"converged {len(result.increments)} increments to t={result.t:.12g}; "
        f"max|u|={abs(result.u).max():.6g}; "
        f"force balance={result.verification['force_balance_inf']:.3e}"
    )


if __name__ == "__main__":
    main()
