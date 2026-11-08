"""Measure SuperLU_MT factorization time against the blocking parameters.

Sweeps max_supernode (sp_ienv(3)), rowblk (sp_ienv(4)) and colblk (sp_ienv(5))
on captured KKT systems and reports median factorization time. Only
factorization is timed: the triangular solves are consistently ~two orders of
magnitude cheaper and are not what these parameters affect.

The blocking parameters are coupled -- max_supernode sets how wide a dense
update gets, rowblk/colblk set how that update is tiled -- so the sweep offers
a coupled mode in addition to the one-at-a-time scan.

Resource note: this script honours the project's testing core budget. It
refuses to run above --max-threads (default 8) unless --allow-full-machine is
given, which is intended for a user-launched full-machine measurement rather
than an agent-launched one. See AGENTS.md.

Example:
    python native/benchmark_superlu_mt_blocking.py \
        native/kkt_debug_2.h5 "t_n=0.02/iteration=1" \
        --threads 1 4 8 --repeats 5 --mode scan
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import json
import statistics
import sys

import h5py
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from superlu_mt_backend import SuperLUMTFactor, load_superlu_mt  # noqa: E402


DEFAULT_LIBRARY = (
    Path(__file__).resolve().parent.parent / "build" / "native" / "libsuperlu_mt_bridge.so"
)

#: Upstream generic defaults for the three blocking parameters.
LIBRARY_DEFAULTS = {"max_supernode": 200, "rowblk": 200, "colblk": 100}


def load_kkt(path: Path, group: str) -> sparse.csc_matrix:
    """Rebuild the full KKT matrix from a captured K and C block."""
    with h5py.File(path, "r") as handle:
        grp = handle[group]
        K = sparse.csr_matrix(
            (grp["K/data"][:], grp["K/indices"][:], grp["K/indptr"][:])
        )
        if "C/data" in grp:
            C = sparse.csr_matrix(
                (grp["C/data"][:], grp["C/indices"][:], grp["C/indptr"][:]),
                shape=(len(grp["r_c"]), K.shape[1]),
            )
        else:
            C = sparse.csr_matrix((0, K.shape[1]))
    zero = sparse.csr_matrix((C.shape[0], C.shape[0]))
    return sparse.bmat([[K, C.T], [C, zero]], format="csc")


def time_factorization(lib, A, nprocs: int, repeats: int, **blocking) -> dict:
    """Median factorization seconds over `repeats` runs, plus diagnostics."""
    times: list[float] = []
    diagnostics = None
    for _ in range(repeats):
        factor = SuperLUMTFactor.factor(lib, A, nprocs=nprocs, **blocking)
        times.append(factor.factor_seconds)
        diagnostics = factor.diagnostics
        factor.close()
    return {
        "median_factor_seconds": statistics.median(times),
        "min_factor_seconds": min(times),
        "max_factor_seconds": max(times),
        "samples": times,
        "diagnostics": diagnostics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path, help="captured KKT system")
    parser.add_argument("group", help="HDF5 group, e.g. 't_n=0.02/iteration=1'")
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--mode", choices=("scan", "coupled"), default="scan",
        help="scan: vary one parameter at a time; coupled: full grid",
    )
    parser.add_argument("--max-supernode", nargs="+", type=int, default=[50, 100, 200, 400, 800])
    parser.add_argument("--rowblk", nargs="+", type=int, default=[50, 100, 200, 400])
    parser.add_argument("--colblk", nargs="+", type=int, default=[25, 50, 100, 200])
    parser.add_argument("--max-threads", type=int, default=8,
                        help="testing core budget; see AGENTS.md")
    parser.add_argument("--allow-full-machine", action="store_true",
                        help="permit threads above --max-threads (user-launched runs)")
    parser.add_argument("--json", type=Path, default=None, help="write results here")
    args = parser.parse_args(argv)

    over_budget = [t for t in args.threads if t > args.max_threads]
    if over_budget and not args.allow_full_machine:
        parser.error(
            f"thread counts {over_budget} exceed the testing core budget of "
            f"{args.max_threads}. Re-run with --allow-full-machine only when the "
            f"machine is known to be free (see AGENTS.md)."
        )

    A = load_kkt(args.h5, args.group)
    lib = load_superlu_mt(args.lib.resolve())
    print(f"system: {A.shape[0]} unknowns, {A.nnz} stored entries")
    print(f"repeats: {args.repeats}; reporting median factorization seconds\n")

    if args.mode == "coupled":
        combos = [
            dict(zip(("max_supernode", "rowblk", "colblk"), values))
            for values in product(args.max_supernode, args.rowblk, args.colblk)
        ]
    else:
        combos = []
        for name, values in (
            ("max_supernode", args.max_supernode),
            ("rowblk", args.rowblk),
            ("colblk", args.colblk),
        ):
            for value in values:
                combo = dict(LIBRARY_DEFAULTS)
                combo[name] = value
                if combo not in combos:
                    combos.append(combo)

    results = []
    for nprocs in args.threads:
        baseline = time_factorization(
            lib, A, nprocs, args.repeats, **LIBRARY_DEFAULTS
        )
        base_t = baseline["median_factor_seconds"]
        print(f"--- threads={nprocs} --- baseline (library defaults) {base_t:.4f} s")
        results.append(
            {"threads": nprocs, "parameters": dict(LIBRARY_DEFAULTS),
             "baseline": True, **baseline}
        )
        rows = []
        for combo in combos:
            if combo == LIBRARY_DEFAULTS:
                continue
            record = time_factorization(lib, A, nprocs, args.repeats, **combo)
            speedup = base_t / record["median_factor_seconds"]
            rows.append((speedup, combo, record))
            results.append(
                {"threads": nprocs, "parameters": combo, "baseline": False,
                 "speedup_vs_default": speedup, **record}
            )
        for speedup, combo, record in sorted(rows, key=lambda r: -r[0]):
            label = " ".join(f"{k}={v}" for k, v in combo.items())
            flag = "  <-- faster" if speedup > 1.02 else ""
            print(
                f"  {label:<48} {record['median_factor_seconds']:.4f} s "
                f"({speedup:5.2f}x){flag}"
            )
        print()

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
