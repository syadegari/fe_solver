# SuperLU_MT factor storage: `fill_lusup`, `fill_ucol`, `fill_lsub`

A practical guide to the three parameters that control how much memory
SuperLU_MT reserves for the LU factors. Read this when a run has died with a
`Storage for ... exceeded` message, or before launching a large job you expect
to be memory-hungry.

No sparse-solver background is needed. If you want to know *why* these arrays
exist and what they hold, see
[SUPERLU_MT_INTERNALS.md](SUPERLU_MT_INTERNALS.md); this document is only about
using them.

## 1. The symptom

When SuperLU_MT runs out of room for the factors, the run does not raise a
Python exception. The library prints to stderr and then **terminates the whole
process immediately**:

```
Storage for  L supernodes exceeded; Current column 48213; Need at least 92716481;
You may set it by the 6-th parameter in routine sp_ienv().
Memory allocation failed
```

What you will observe:

- no Python traceback and no `ModelError`;
- a non-zero exit status (`-1`), often reported by a shell as `255`;
- no restart file and no results for the increment in progress;
- the solver appears to "vanish" part-way through a long run.

This is a hard `exit()` inside the C library. The FE solver cannot catch it, so
there is nothing to fix on the Python side — the only remedy is to give the
factorization more room before it starts, which is what these three parameters
do.

The `sp_ienv()` advice in the message is the library's own wording and refers
to editing and recompiling the library. Ignore it: use the parameter below that
corresponds to the quoted number.

## 2. Which parameter to raise

The message names the array that ran out. Map it directly:

| Message says | Parameter number | Set this | Holds |
|---|---|---|---|
| `Storage for  L supernodes exceeded` | 6 | `fill_lusup` | the numerical values of `L` |
| `Storage for    U columns exceeded`  | 7 | `fill_ucol`  | the values and indices of `U` |
| `Storage for L subscripts exceeded`  | 8 | `fill_lsub`  | the row indices of `L` |

Raise the one named in the message. Raising all three is harmless but reserves
more memory than necessary.

## 3. How a value is interpreted

Each parameter accepts three kinds of value, and the sign is what selects the
meaning. This is the single most important thing to get right:

| Value | Meaning | Example |
|---|---|---|
| **Negative** | A **growth factor**. The library reserves `abs(value) x nnz(A)` entries, where `nnz(A)` is the number of stored entries in your KKT matrix. | `-100` with a 2,000,000-entry matrix reserves 200,000,000 entries |
| **Positive** | An **absolute number of entries**. No scaling is applied. | `250000000` reserves exactly 250,000,000 entries |
| **`0`** (default) | Use the library's built-in value. | — |

The built-in defaults are growth factors:

| Parameter | Default | Meaning |
|---|---|---|
| `fill_lusup` | `-50` | 50 x nnz(A) entries |
| `fill_ucol`  | `-50` | 50 x nnz(A) entries |
| `fill_lsub`  | `-30` | 30 x nnz(A) entries |

**Negative values are almost always what you want.** They scale with the
problem, so the same deck works across mesh refinements. Use a positive
absolute count only when you are deliberately pinning memory for one specific
matrix — and note that the solver prints an extra warning when you do, because
an absolute value that is too small produces exactly the fatal abort above.

## 4. Choosing a value

The defaults are a guess that the factors will be ~50x denser than the original
matrix. Some problems are far worse, and that is not a sign of a mistake in
your model: factor density depends on mesh connectivity and ordering, not just
on size.

A practical procedure:

1. **Double the default for the array that failed.** `fill_lusup = -100`
   (from `-50`), or `fill_lsub = -60` (from `-30`).
2. **If it still aborts, double again.** `-200`, then `-400`. Growth is
   problem-specific; there is no formula worth deriving.
3. **Use the number in the message as a floor.** `Need at least 92716481` tells
   you the factorization needed at least that many entries *at the column it
   reached*, and it had not finished — so the true requirement is larger.
   Divide by your matrix's stored-entry count to get the growth factor it had
   already exceeded, and go comfortably above it.
4. **Check stderr if raising the value does not help.** See section 6.

A worked example, from the storage sweep in
[../verification/superlu_mt_results/refact_study/](../verification/superlu_mt_results/refact_study/):
a `52,281 x 52,281` KKT system with 3,522,771 stored entries (the snapshot in
`native/kkt_debug_2.h5`) aborts on `fill_ucol` at the default `-50`, and again
at `-60` and `-70`. It completes at `-80`, where `U` occupies 264.4M of the
281.8M reserved entries. Each failed attempt reached a later column and
reported a larger floor — 176.1M, 211.4M, then 246.6M — which is step 3 above
in action: the floor in the message trails the true requirement, so stepping up
one notch at a time costs a run per notch. The same directory holds the probe
and scripts used, if you want to measure a problem family's real growth before
a long job rather than after a crash.

The solver reports your matrix's stored-entry count on the `linear system:`
line of every run, so you can compute what a given growth factor will reserve.

## 5. What it costs

Reserved entries are allocated up front, whether or not they are used. Per
entry, on the 32-bit-index build:

| Parameter | Bytes per entry |
|---|---|
| `fill_lusup` | 8 (one double) |
| `fill_ucol`  | 12 (one double + one index) |
| `fill_lsub`  | 4 (one index) |

So `fill_lusup = -100` on a matrix with 2,000,000 stored entries reserves
`100 x 2,000,000 x 8 bytes = 1.6 GB` for that array alone. Budget accordingly
on a shared machine, and remember that each concurrent solver process pays this
separately.

Over-reserving costs memory, not speed: these parameters set the size of the
initial allocation and do not change the arithmetic the factorization performs.
Factorization time is effectively unaffected. The practical risk of a very
large value is not slowness but failing to allocate at all — which leads to the
next section.

## 6. When a raised value still aborts

If the enlarged request does not fit in available memory, SuperLU_MT does not
fail outright. It **halves the request and retries**, printing to stderr:

```
.. pdgstrf_MemInit(): Not enough memory for the requested arrays
   (nzlmax 300000000, nzumax 500000000); halving the request, retry 1
```

The run then continues with a smaller allocation than you asked for, and may
hit the same fatal abort anyway. So if raising a parameter appears to have no
effect, **search stderr for `halving the request`** before raising it further:
the increase may never have been granted. In that case the problem is physical
memory, and the options are a machine with more RAM, fewer concurrent
processes, or a smaller problem — not a larger parameter.

## 7. How to set them

These are expert-level parameters: they are opt-in, and every change is
announced at startup. There is no command-line path to them; they belong to
the deck. Full contract and precedence rules are in [RUNNING.md](RUNNING.md);
in brief, either have the deck point at a parameter file:

```toml
# in the deck
[linear_solver]
superlu_mt_params_file = "../native/superlu_mt_params.toml"   # relative to the deck
```

```toml
# native/superlu_mt_params.toml
[superlu_mt]
fill_lusup = -100
```

or pin them in the deck, which takes precedence over a parameter file:

```toml
[linear_solver.superlu_mt]
fill_lusup = -100
fill_lsub = -60
```

A run that changes nothing prints no banner. A run that does prints an
unmissable block naming each changed parameter, its default, and where the
value came from, and records the same provenance in `run_log.json`:

```
************************************************************************
*** EXPERT SuperLU_MT OVERRIDES ACTIVE (1 of 6 parameters) ***
    parameter file: /path/to/superlu_mt_params.toml
    sha256:         eb89d524...
    fill_lusup = -100  <-- OVERRIDDEN by file (default 0); L supernode value
                           storage; <0 factor on nnz(A), >0 entries, 0 = sp_ienv(6)
    ...
************************************************************************
```

Unknown or malformed parameter names are a hard error, so a typo cannot
silently leave you on the defaults.

## 8. Monitoring headroom before it becomes a crash

Every `superlu_mt` run prints one diagnostics line after its first
factorization and records the same numbers in `run_log.json` under
`linear_system.factor_diagnostics`:

```
factor diagnostics: nnz(L)=9102, nnz(U)=1.021e+04; fill growth 1.7x nnz(KKT);
7.3e+05 flops; LU storage 0.2 MB, total needed 0.9 MB; expansions 0
```

Two of these numbers tell you how close a run is to the abort *before* a
larger mesh triggers it:

- **`fill growth`** is `(nnz(L) + nnz(U)) / nnz(KKT)` — the measured version of
  the quantity the `fill_*` growth factors estimate. If a mesh family shows
  growth trending toward the budget (default 50x for values, 30x for
  subscripts), raise the matching parameter before scaling up further.
- **`expansions`** counts how many times the library had to grow the factor
  arrays past their initial estimate. Non-zero prints an explicit warning: the
  run survived, but the estimate was wrong, and a larger problem in the same
  family may not get the chance to expand. Treat it as the early form of the
  `Storage for ... exceeded` abort.

`nnz(L)` includes supernode padding, so it can exceed the mathematically
necessary fill — it is the right number to compare against the storage budget,
which pads the same way.

## 9. Quick reference

- Run died with no traceback and exit status 255 → suspect this.
- Read which array the message names → raise the matching `fill_*`.
- Use negative values (growth factor on `nnz(A)`); start by doubling the default.
- Positive values are absolute entry counts — rarely what you want.
- Over-reserving costs memory, not speed.
- Still aborting? Check stderr for `halving the request`.
