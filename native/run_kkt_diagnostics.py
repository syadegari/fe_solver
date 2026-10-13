"""Run a small, sequential diagnostic sequence with per-run wall-time limits.

Usage: conda run -n py3.14 python run_kkt_diagnostics.py
All commands and full logs are saved; failed convergence is expected evidence.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--ranks', type=int, default=8)
parser.add_argument('--output-dir', type=Path, default=Path('kkt_diagnostics'))
parser.add_argument('--timeout', type=float, default=30)
parser.add_argument('--stage', choices=('initial', 'followup'), default='initial')
args = parser.parse_args()
cases = [
    ('a0_b0', []),
    ('a0_b1', ['--rhs', '1']),
    ('a1_b0', ['--matrix', '1']),
    ('a1_b1', ['--matrix', '1', '--rhs', '1']),
    ('constraint', ['--block', 'constraint']),
    ('force', ['--rhs', '1', '--block', 'force']),
    ('random_force', ['--block', 'random-force']),
    ('refine', ['-bench_ksp_gmres_cgs_refinement_type', 'refine_always']),
    ('restart100', ['-bench_ksp_gmres_restart', '100']),
    ('scaled', ['--scale']),
    ('compatible', ['--guess', 'compatible']),
]
if args.stage == 'followup':
    cases = [
        ('full', ['-bench_pc_fieldsplit_schur_fact_type', 'full']),
        ('full_restart100', ['-bench_pc_fieldsplit_schur_fact_type', 'full', '-bench_ksp_gmres_restart', '100']),
        ('selfp_lu', ['-bench_fieldsplit_lambda_pc_type', 'redundant',
                      '-bench_fieldsplit_lambda_redundant_ksp_type', 'preonly',
                      '-bench_fieldsplit_lambda_redundant_pc_type', 'lu', '-bench_ksp_view']),
    ]
args.output_dir.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, HWLOC_COMPONENTS='-gl', BENCH_NUM_THREADS='1')
for name, extra in cases:
    path = args.output_dir / f'{name}.json'
    # A failed launch must not be mistaken for a result from an earlier run.
    path.unlink(missing_ok=True)
    command = ['mpiexec', '-n', str(args.ranks), sys.executable,
               'diagnose_petsc_kkt.py', '--output', str(args.output_dir / f'{name}.json'), *extra]
    # Record a portable reproduction command, omitting the interpreter's path.
    portable = [*command[:3], 'python', *command[4:]]
    with (args.output_dir / f'{name}.log').open('w') as log:
        log.write(shlex.join(portable) + '\n')
        log.flush()
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 'timeout'
    if code != 'timeout' and path.exists():
        result = json.loads(path.read_text())
        print(name, result['iterations'], result['reason'], f"{result['petsc_residual']:.6e}", flush=True)
    else:
        print(name, code, 'see log', flush=True)
