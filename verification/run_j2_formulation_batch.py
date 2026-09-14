"""Run the expensive J2 formulation study as a resumable two-phase batch.

The gate phase advances every outstanding case to t=0.5.  After inspecting
those results, the complete phase resumes the same databases to t=1.  All
machine-specific provenance and logs live below the ignored run root.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
from time import time
from typing import Any

import h5py

from verification.check_j2_prism import _load_solver_summary
from verification.run_j2_formulation_study import CASES, ROOT


OUTSTANDING_CASES = (
    "soft_bulk_hex8_fbar",
    "soft_bulk_hex8",
    "coarse_hex20",
    "refined_hex8_fbar",
    "refined_hex8",
    "circular_hex8_fbar",
    "refined_circular_hex8_fbar",
)
LOCK_FILE = ROOT / "requirements-j2-study-linux-x86_64.lock"
THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _capture_git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_provenance() -> dict[str, Any]:
    status = _capture_git("status", "--porcelain=v1", "--untracked-files=all")
    difference = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    digest = hashlib.sha256(status.encode("utf-8") + b"\0" + difference).hexdigest()
    return {
        "revision": _capture_git("rev-parse", "HEAD"),
        "branch": _capture_git("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_tree_sha256": digest,
        "dirty_paths": status.splitlines(),
    }


def _environment_provenance() -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy", "scipy", "gmsh", "h5py", "threadpoolctl", "numba",
        "llvmlite", "matplotlib",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "logical_cpus": os.cpu_count(),
        "packages": packages,
        "lock_file": str(LOCK_FILE.relative_to(ROOT)) if LOCK_FILE.is_file() else None,
        "lock_file_sha256": _sha256(LOCK_FILE) if LOCK_FILE.is_file() else None,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _database_final_time(path: Path) -> float | None:
    if not path.is_file():
        return None
    with h5py.File(path, "r") as archive:
        complete = int(archive["results"].attrs["n_complete_steps"])
        return float(archive["results/time"][complete - 1]) if complete else None


def _restart_time(path: Path) -> float:
    with h5py.File(path, "r") as archive:
        return float(archive.attrs["t_n"])


def latest_restart(output_directory: Path, *, at_most: float) -> Path | None:
    candidates = []
    for path in output_directory.glob("restart/restart_*.h5"):
        restart_time = _restart_time(path)
        if restart_time <= at_most + 1.0e-12:
            candidates.append((restart_time, path.name, path))
    return max(candidates, default=(0.0, "", None))[2]


def _next_attempt(manifest: dict[str, Any], case_name: str) -> int:
    return 1 + len(manifest["cases"].get(case_name, {}).get("attempts", []))


def _solver_command(
    case_name: str,
    output_directory: Path,
    *,
    target_time: float,
    restart_from: Path | None,
    num_processes: int,
    resource_log: Path,
    growth_threshold: int | None = None,
) -> list[str]:
    command = [
        "/usr/bin/time", "-v", "-o", str(resource_log),
        sys.executable, "-m", "verification.run_j2_formulation_study",
        "--case", case_name,
        "--num-processes", str(num_processes),
        "--debug-timing",
        "--output-directory", str(output_directory),
    ]
    if target_time < 1.0 - 1.0e-12:
        command.extend(("--stop-time", str(target_time)))
    if restart_from is not None:
        command.extend(("--restart-from", str(restart_from)))
    if growth_threshold is not None:
        command.extend(("--grow-if-newton-iterations-le", str(growth_threshold)))
    return command


def _run_and_tee(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("command: " + shlex.join(command) + "\n\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            return process.wait()
        except KeyboardInterrupt:
            # The solver may own a fork-server and element workers.  Signal the
            # complete session so no descendants keep the tee pipe open.
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            raise


def _parse_growth_thresholds(values: list[str]) -> dict[str, int]:
    thresholds: dict[str, int] = {}
    for value in values:
        case_name, separator, count_text = value.partition("=")
        if not separator or not case_name or not count_text:
            raise ValueError(
                f"invalid growth threshold {value!r}; expected CASE=COUNT"
            )
        if case_name not in OUTSTANDING_CASES:
            raise ValueError(f"unknown growth-threshold case {case_name!r}")
        if case_name in thresholds:
            raise ValueError(f"duplicate growth threshold for {case_name!r}")
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(
                f"growth threshold for {case_name!r} must be an integer"
            ) from exc
        if count < 0:
            raise ValueError(
                f"growth threshold for {case_name!r} must be nonnegative"
            )
        thresholds[case_name] = count
    return thresholds


def _existing_or_seed_restart(
    case_name: str,
    output_directory: Path,
    *,
    target_time: float,
    circular_restart: Path | None,
    cold_circular: bool,
) -> Path | None:
    restart = latest_restart(output_directory, at_most=target_time)
    if restart is not None:
        return restart.resolve()
    if case_name != "circular_hex8_fbar" or target_time > 0.5 + 1.0e-12:
        return None
    if circular_restart is not None:
        if not circular_restart.is_file():
            raise SystemExit(f"circular restart does not exist: {circular_restart}")
        restart_time = _restart_time(circular_restart)
        if restart_time > target_time + 1.0e-12:
            raise SystemExit("circular restart lies beyond the gate target")
        return circular_restart.resolve()
    if not cold_circular:
        raise SystemExit(
            "the circular gate requires --circular-restart or --cold-circular"
        )
    return None


def _load_or_create_manifest(
    path: Path,
    *,
    phase: str,
    num_processes: int,
    growth_thresholds: dict[str, int],
    argv: list[str],
) -> dict[str, Any]:
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "created_unix_time": time(),
            "repository": _repository_provenance(),
            "environment": _environment_provenance(),
            "thread_limits": {name: "1" for name in THREAD_LIMIT_VARIABLES},
            "invocations": [],
            "cases": {},
        }
    manifest["invocations"].append(
        {
            "unix_time": time(),
            "phase": phase,
            "num_processes": num_processes,
            "growth_thresholds": growth_thresholds,
            "repository": _repository_provenance(),
            "environment": _environment_provenance(),
            "argv": argv,
        }
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("gate", "complete"), required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--cases", nargs="+", choices=OUTSTANDING_CASES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--circular-restart", type=Path, default=None)
    parser.add_argument("--cold-circular", action="store_true")
    parser.add_argument(
        "--growth-threshold",
        action="append",
        default=[],
        metavar="CASE=COUNT",
        help="case-specific override of grow_if_newton_iterations_le; repeatable",
    )
    args = parser.parse_args(argv)
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    try:
        growth_thresholds = _parse_growth_thresholds(args.growth_threshold)
    except ValueError as exc:
        parser.error(str(exc))

    run_root = args.run_root.resolve()
    manifest_path = run_root / "manifest.json"
    if run_root.exists() and any(run_root.iterdir()) and not args.resume:
        parser.error("nonempty --run-root requires --resume")
    if args.phase == "complete" and not manifest_path.is_file():
        parser.error("complete phase requires an existing gated run root")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "resources").mkdir(exist_ok=True)
    (run_root / "results").mkdir(exist_ok=True)

    selected = tuple(args.cases) if args.cases else OUTSTANDING_CASES
    target_time = 0.5 if args.phase == "gate" else 1.0
    manifest = _load_or_create_manifest(
        manifest_path,
        phase=args.phase,
        num_processes=args.num_processes,
        growth_thresholds=growth_thresholds,
        argv=sys.argv if argv is None else [sys.argv[0], *argv],
    )
    _write_manifest(manifest_path, manifest)

    environment = os.environ.copy()
    environment.update({name: "1" for name in THREAD_LIMIT_VARIABLES})
    for case_name in selected:
        if case_name not in CASES:  # defensive if the two case registries diverge
            raise SystemExit(f"unknown formulation-study case: {case_name}")
        output_directory = run_root / "results" / case_name
        database = output_directory / "run.h5"
        final_time = _database_final_time(database)
        if final_time is not None and abs(final_time - target_time) <= 1.0e-12:
            print(f"{case_name}: already complete through t={target_time:g}; skipping")
            continue
        if args.phase == "gate" and final_time is not None and final_time > target_time:
            raise SystemExit(
                f"{case_name}: existing result at t={final_time:g} exceeds gate target"
            )
        restart = _existing_or_seed_restart(
            case_name,
            output_directory,
            target_time=target_time,
            circular_restart=args.circular_restart,
            cold_circular=args.cold_circular,
        )
        if args.phase == "complete" and restart is None:
            raise SystemExit(
                f"{case_name}: no retained gate restart is available for completion"
            )

        attempt_number = _next_attempt(manifest, case_name)
        log_path = run_root / "logs" / f"{case_name}.attempt_{attempt_number:02d}.log"
        resource_log = (
            run_root / "resources" / f"{case_name}.attempt_{attempt_number:02d}.txt"
        )
        command = _solver_command(
            case_name,
            output_directory,
            target_time=target_time,
            restart_from=restart,
            num_processes=args.num_processes,
            resource_log=resource_log,
            growth_threshold=growth_thresholds.get(case_name),
        )
        case_record = manifest["cases"].setdefault(case_name, {"attempts": []})
        attempt = {
            "attempt": attempt_number,
            "phase": args.phase,
            "target_time": target_time,
            "restart_from": str(restart) if restart is not None else None,
            "command": command,
            "log": str(log_path),
            "resource_log": str(resource_log),
            "started_unix_time": time(),
            "status": "running",
        }
        case_record["attempts"].append(attempt)
        _write_manifest(manifest_path, manifest)
        print(f"\n=== {case_name}: {args.phase} to t={target_time:g} ===", flush=True)
        try:
            return_code = _run_and_tee(command, log_path, environment)
        except KeyboardInterrupt:
            attempt.update(status="interrupted", finished_unix_time=time())
            _write_manifest(manifest_path, manifest)
            raise SystemExit(130) from None
        final_time = _database_final_time(database)
        solver_summary = (
            _load_solver_summary(database, "run_log.json")
            if final_time is not None
            else None
        )
        attempt.update(
            status="complete" if return_code == 0 else "failed",
            exit_status=return_code,
            finished_unix_time=time(),
            database=str(database),
            database_final_time=final_time,
            solver_summary=solver_summary,
        )
        _write_manifest(manifest_path, manifest)
        if return_code != 0:
            raise SystemExit(f"{case_name} failed with exit status {return_code}")
        if final_time is None or abs(final_time - target_time) > 1.0e-12:
            attempt["status"] = "invalid_output"
            _write_manifest(manifest_path, manifest)
            raise SystemExit(
                f"{case_name} exited successfully but database ended at {final_time!r}"
            )
        case_record[f"{args.phase}_complete"] = True
        case_record[f"{args.phase}_database"] = str(database)
        _write_manifest(manifest_path, manifest)

    print(f"{args.phase} phase complete; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
