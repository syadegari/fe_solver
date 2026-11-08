# SuperLU_MT blocking parameter scan

Measured effect of `max_supernode` (`sp_ienv(3)`), `rowblk` (`sp_ienv(4)`) and
`colblk` (`sp_ienv(5)`) on factorization time, after exposing them through the
bridge's `sp_ienv()` override.

**Outcome: no material speedup was found. The shipped defaults are unchanged.**

## Setup

- System: `native/kkt_debug_1.h5`, group `t_n=0.02/iteration=2` — a captured FE
  KKT system with 17,193 unknowns and 1,068,651 stored entries.
- Library defaults (generic branch of upstream `SRC/sp_ienv.c`):
  `max_supernode = 200`, `rowblk = 200`, `colblk = 100`.
- One parameter varied at a time from the default configuration.
- Median of 3 factorizations per configuration; factorization time only
  (`factor_seconds` from the bridge). Solve time is excluded: it is
  consistently ~two orders of magnitude smaller and is not what these
  parameters affect.
- Threads limited to 8 per the project's testing core budget (`AGENTS.md`).
- Machine had unrelated background load (load average 3-4 of 16 cores) during
  the measurement, so differences below roughly 5% are not meaningful.

## Results

| Configuration | 4 threads | vs default | 8 threads | vs default |
| --- | --- | --- | --- | --- |
| defaults | 8.320 s | 1.00x | 4.928 s | 1.00x |
| `max_supernode = 100` | 8.252 s | 1.01x | 4.829 s | 1.02x |
| `max_supernode = 400` | 8.617 s | 0.97x | 5.184 s | 0.95x |
| `rowblk = 100` | 8.334 s | 1.00x | 5.001 s | 0.99x |
| `rowblk = 400` | 8.592 s | 0.97x | 4.999 s | 0.99x |
| `colblk = 50` | 8.497 s | 0.98x | 5.067 s | 0.97x |
| `colblk = 200` | 8.449 s | 0.98x | 5.035 s | 0.98x |

Every variation falls within +-5% of the default, which is inside the noise
floor for this machine state. The only consistently positive direction,
`max_supernode = 100` at 1.01-1.02x, is too small to act on. Increasing any
parameter above its default was mildly unfavourable on both thread counts,
which is consistent with larger dense blocks losing cache residency and
coarsening thread granularity.

## Conclusion and guidance

- Defaults are kept. No shipped default was changed on the strength of this
  scan.
- The parameters remain exposed as expert knobs because the optimum is
  system-dependent: a much larger or differently structured KKT system may
  respond where this one did not.
- Anyone tuning them should measure on their own system with
  `native/benchmark_superlu_mt_blocking.py`, vary them together rather than
  individually (`--mode coupled`), and discount differences below ~5% unless
  the machine is otherwise idle.

## Correctness

Independently of timing, the overrides were verified not to change results:
factorizations with `max_supernode = 1`, with
`max_supernode = 4, rowblk = 8, colblk = 4`, and with
`rowblk = colblk = 1000` all reproduce the SciPy reference solution to
`rtol = 1e-9`. A test also asserts that the bridge's `sp_ienv()` fallback
values still match upstream, and that SuperLU_MT's internals actually resolve
`sp_ienv()` to the bridge's definition (verified by a call counter: a single
small factorization produced 435 internal calls).

## Reproducing

```bash
python native/benchmark_superlu_mt_blocking.py \
    native/kkt_debug_1.h5 "t_n=0.02/iteration=2" \
    --threads 4 8 --repeats 3 --mode scan
```

A full-machine (16-thread) confirmation has not been run; it requires
`--allow-full-machine` and an otherwise idle machine.
