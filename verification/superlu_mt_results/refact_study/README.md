# `refact` symbolic-reuse study (2026-11-13)

**Outcome: `refact` was not added to the solver.** The reuse it offers is worth
0–2% of a factorization and shrinks as problems grow, while it fixes the factor
storage at the first factorization's size and makes a later overflow an
unrecoverable `exit(-1)`. The reasoning is in
[../../../docs/SUPERLU_MT_INTERNALS.md](../../../docs/SUPERLU_MT_INTERNALS.md),
section 6.

This directory is the minimal set needed to reproduce the measurement.

## Files

| file | role |
| --- | --- |
| `refact_probe.c` | standalone probe: factor, perturb the values, factor again `fresh` / `refact` / `refact0` |
| `dump_kkt.py` | writes a KKT snapshot (`native/kkt_debug_*.h5`) to the probe's flat binary format |
| `sweep.sh` | interleaved `fresh`/`refact0` sampling at a list of thread counts |
| `analyze.py` | summarises `RESULT` lines: per-mode means, the `f2/f1` ratio, Welch `t` |
| `results/kkt1.txt`, `results/kkt2.txt` | the `RESULT` lines behind the published numbers |

The probe is deliberately not part of the CMake build: it is a one-off
experiment, not a maintained target. It links the same static archive the
bridge does.

## What the probe measures

One run factors the matrix, replaces the values with a +-1% perturbation of
themselves (same sparsity pattern, standing in for the next Newton tangent),
and factors a second time in one of three modes:

- `fresh` — full `get_perm_c` + `pdgstrf_init(refact=NO)` + `pdgstrf`;
- `refact` — `pdgstrf_init(refact=YES)` + `pdgstrf`;
- `refact0` — `pdgstrf` alone. `pdgstrf_init()` is skipped because `AC->nzval`
  aliases `A->nzval`, and `perm_c` and `AC`'s column pointers are already the
  composed, postordered ones from the first call. This is the cheapest reuse
  the library's API permits, so it bounds what any `refact` integration could
  save.

Each `RESULT` line carries both factorizations' total time (`f1`, `f2`) and
their stages (`p` = `get_perm_c`, `i` = `pdgstrf_init`, `g` = `pdgstrf`), plus
the solve residual of the second one, so a saving that comes with a wrong
answer is visible. Comparing `f2/f1` within one process cancels the
process-to-process speed drift, which on this machine reaches 20%.

## Reproducing

Build the bridge first (see `native/README.md`), then:

```bash
cc -O3 -DNDEBUG -o /tmp/refact_probe refact_probe.c \
  -I build/_deps/superlu_mt-src/SRC -D__PTHREAD \
  build/_deps/superlu_mt-build/SRC/libsuperlu_mt_PTHREAD.a \
  build/_deps/superlu_mt-build/CBLAS/libblas.a -lpthread -lm

conda run -n py3.14 python dump_kkt.py \
  native/kkt_debug_1.h5 't_n=0.02/iteration=2' /tmp/kkt1.bin

# 5 interleaved fresh/refact0 pairs at 4 and 8 threads, library default fill
./sweep.sh /tmp/refact_probe /tmp/kkt1.bin 5 0 4 8 | tee /tmp/kkt1_run.txt
conda run -n py3.14 python analyze.py /tmp/kkt1_run.txt
```

Partial reproduction is fine: one thread count and three pairs already shows
the size of the effect against its noise.

`sweep.sh` sets `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1`;
without them the Python-side BLAS pool saturates every core and the timings are
meaningless. Submit the run through the machine's resource manager rather than
launching it loose, and keep the thread budget in `AGENTS.md` in mind.

The larger snapshot needs a raised storage budget — `fill = -80`, the fifth
argument of `sweep.sh` — or it aborts before producing any timing:

```bash
conda run -n py3.14 python dump_kkt.py \
  native/kkt_debug_2.h5 't_n=0.02/iteration=1' /tmp/kkt2.bin
./sweep.sh /tmp/refact_probe /tmp/kkt2.bin 3 -80 16 8
```

## Measured result

Saving of `refact0` over a `fresh` second factorization, split into the part
that needs `refact` and the part that does not. `n` is pairs per thread count.

| matrix | threads | n | fresh `f2` | total | ordering (no `refact` needed) | `refact`-only | noise (sd/`f2`) | Welch `t` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kkt1 | 1 | 3 | 29.23 s | +0.4% | 0.24% | +0.2% | 3.2% | 0.2 |
| kkt1 | 2 | 3 | 16.05 s | +3.2% | 0.46% | +2.7% | 2.1% | 2.8 |
| kkt1 | 4 | 5 | 9.22 s | +2.6% | 0.83% | +1.8% | 4.8% | 0.8 |
| kkt1 | 8 | 5 | 5.50 s | +2.6% | 1.40% | +1.2% | 1.5% | 7.2 |
| kkt1 | 16 | 5 | 4.10 s | +3.8% | 1.76% | +2.0% | 6.2% | 1.0 |
| kkt2 | 8 | 3 | 72.14 s | +0.5% | 0.59% | -0.1% | 0.7% | 0.9 |
| kkt2 | 16 | 3 | 78.47 s | +2.6% | 0.57% | +2.0% | 7.1% | 0.4 |

kkt1 is `17,193 x 17,193` with 1,068,651 stored entries; kkt2 is
`52,281 x 52,281` with 3,522,771. Every residual was 1e-11 or smaller, and no
timing run hit a storage abort.

The only well-resolved points are kkt1 at 8 threads (+2.6%) and kkt2 at
8 threads (+0.5%); the rest sit inside their own noise. The two conclusions
that survive:

- **The gain shrinks with problem size.** On the 3.3x larger matrix at
  8 threads it is +0.5%, and all of that is the reused ordering: the
  `refact`-only part is -0.1%.
- **What grows with thread count is the ordering share**, not `refact`. The
  serial setup is a fixed ~0.08 s (kkt1) or ~0.46 s (kkt2) against a `pdgstrf`
  that gets shorter, so its share rises from 0.24% to 1.76% on kkt1. Caching
  `perm_c` captures it with `refact=NO`.

## Caveats

- The 16-thread blocks are the noisiest, and unmanaged desktop load (a
  software-GL render pool across all 32 logical CPUs) was active during part of
  the kkt2 16-thread block. Interleaving keeps the comparison fair, but the
  absolute times in that block drift by 15%.
- Thread placement outweighs everything measured here: 8 threads confined to
  one CCD of the 5950X factored kkt2 in 98 s, against 72 s for the same 8
  threads spread over a 16-core allocation. The 16-thread runs were no faster
  than that.
- `fill = -80` is the smallest working budget for kkt2 in `-10` steps; `-50`
  (the default) and `-60`, `-70` all abort. U then occupies 264.4-264.8M of the
  281.8M reserved entries, and the two factorizations of a single run differ by
  342k entries, i.e. the requirement moves under `refact`'s frozen allocation.
