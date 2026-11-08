# SuperLU_MT internals: supernodes, the elimination tree, and the tuning knobs

This note explains the data structures behind the expert-level SuperLU_MT
parameters exposed by the FE solver (see [RUNNING.md](RUNNING.md), *Expert-level
factorization parameters*). It assumes familiarity with LU factorization and
basic multithreading primitives, but not with sparse direct solvers.

It is background reading, not a troubleshooting guide. To diagnose and fix a
`Storage for ... exceeded` abort, read
[SUPERLU_MT_FACTOR_STORAGE.md](SUPERLU_MT_FACTOR_STORAGE.md) instead; it is
self-contained and requires none of the material below.

All source references are to the pinned upstream SuperLU_MT v4.1.0
(`xiaoyeli/superlu_mt`, commit `f2e50bf`).

## 1. The problem supernodes solve

Eliminating column `j` of a sparse matrix means updating every later column
that has a nonzero in a row where column `j` does. Done column by column, each
update is a sparse `axpy`: one multiply-add per memory reference, indirect
indexing through row-index arrays, no vectorization. The factorization becomes
memory-bandwidth bound and the floating-point units idle.

A **supernode** is a run of consecutive columns `r..s` of `L` whose nonzero row
structure below the diagonal block is identical (and whose diagonal block is
dense). Because the columns match, SuperLU stores the run as one dense
rectangle:

```
          w = s - r + 1 columns
         ┌──────────┐
         │ █ ▓ ▓ ▓  │   dense w x w triangular diagonal block
         │   █ ▓ ▓  │
         │     █ ▓  │
         │       █  │
         ├──────────┤
         │ █ █ █ █  │
  h rows │ █ █ █ █  │   dense h x w block: the shared sparsity
         │ █ █ █ █  │   pattern of all w columns
         │ █ █ █ █  │
         └──────────┘
```

Two storage consequences, which map one-to-one onto the Phase-1 memory knobs:

| Array (in `GlobalLU_t`) | Contents | Expert knob | `sp_ienv` | Default |
|---|---|---|---|---|
| `lusup` | numerical values of the L supernode rectangles | `fill_lusup` | 6 | `-50` |
| `lsub`  | row indices, stored **once per supernode**, not per column | `fill_lsub` | 8 | `-30` |
| `ucol`/`usub` | columns of U (plain sparse, *not* supernodal) | `fill_ucol` | 7 | `-50` |

`lsub`'s smaller default growth factor (`-30` vs `-50`) is the index-sharing at
work: one subscript list serves all `w` columns of a supernode.

The payoff: eliminating a supernode of width `w` updates its targets with a
dense matrix-matrix product (`GEMM`, BLAS-3) on the stored rectangle instead of
`w` separate sparse `axpy`s (BLAS-1). That is roughly `O(w)` arithmetic per
memory reference, vectorizable and cache-blockable — the single largest
performance lever in sparse direct factorization. It is also why the solve
phase is consistently ~two orders of magnitude cheaper than factorization in
our measurements: triangular solves stream through `nnz(L)` once, while
factorization does the `GEMM`-heavy `O(flops)` work that supernodes exist to
accelerate.

## 2. The elimination tree

The **column elimination tree** (etree) compresses "which columns feed which"
into a tree: the parent of column `j` is the first column `k > j` that
column `j` fills into (the row of its first subdiagonal nonzero in `L`).
Column `j`'s values flow into its ancestors — and into nothing else.
For our unsymmetric KKT systems SuperLU_MT uses the etree of `AᵀA`
(we never assume KKT symmetry).

A 9-column example:

```
                 (9)            <- root: receives updates from everything
                  |
                 (8)
                /   \
             (3)     (7)
            /   \     |
          (1)   (2)  (6)
                      |
                     (4)(5)     <- leaves: depend on nothing
```

Reading it as a dependency graph:

- factoring a column requires all its **descendants** to be finished;
- columns in **disjoint subtrees never touch each other's data**.

Columns 1, 2, 4, 5 can be eliminated by different threads simultaneously with
no communication. This is the entire basis of the parallelism in SuperLU_MT.

Columns are processed in **postorder** of the etree, which makes every subtree
a *contiguous* range of columns — the reason a supernode (or a relaxed subtree,
below) can be "a run of consecutive columns" at all.

## 3. Fill-in flows up the tree — and creates the supernodes

The link between the two concepts: **fill-in accumulates upward.** When column
3 is eliminated it inherits the union of the structures of columns 1 and 2;
column 8 inherits from both subtrees under it. Therefore:

- **Near the leaves**, columns are sparse and structurally diverse (siblings
  come from unrelated parts of the mesh), so *natural* supernodes would be
  width 1 — pure BLAS-1, the worst case.
- **Near the root**, so much fill has merged that consecutive columns become
  structurally identical, and natural supernodes grow wide on their own.

Laying the columns of `L` left-to-right in postorder:

```
 columns:   1  2  3    4  5  6  7    8  9
            └──┬──┘    └────┬────┘   └┬─┘
          relaxed        relaxed +  natural snode
          supernode      natural    (dense chain at
          (small subtree snodes     the root;
           padded dense)            maxsuper caps it)

 supernode   thin, padded ──────────► wide, natural
 width:      (set by `relax`)         (capped by `maxsuper`)

 parallelism: many independent ─────► few huge tasks
              cheap tasks             (2-D blocking helps)

 memory:     ~nnz(A) ───────────────► nnz(A) x growth factor
             (fill accumulates upward; `fill_*` size the arrays)
```

Three parameter families are three views of this same picture:

**`relax` (`sp_ienv(2)`, default 6) — floor at the leaves.** Any etree subtree
with at most `relax` nodes is declared one supernode *regardless of row
structure*, padding the rectangle with explicit zeros: some wasted storage and
flops on zeros, bought back many times over by dense kernels.
`SRC/pxgstrf_relax_snode.c` computes descendant counts
(`desc[parent] += desc[j] + 1`) and cuts wherever the count fits — the unit is
a subtree, not a column interval, which postorder then makes contiguous.

**`maxsuper` (`sp_ienv(3)`, default 200) — ceiling at the root.**
`SRC/pdgstrf_column_dfs.c` stops extending a supernode at `maxsuper` columns
even when the structure would continue. Why cap at all, if wider means more
BLAS-3?

1. *Cache:* the dense `h x w` rectangle is the update kernel's working set;
   past a point it stops fitting and the `GEMM` degrades toward
   bandwidth-bound again.
2. *Memory:* wider padded supernodes store more explicit zeros.
3. *Parallel granularity:* near the root the tree narrows to a chain and the
   remaining work concentrates in a few very large supernodes. Unbounded,
   the endgame becomes one enormous serial task while other threads idle;
   capped, it stays divisible into schedulable units.

**`fill_lusup` / `fill_ucol` / `fill_lsub` (`sp_ienv(6/7/8)`) — the memory
guess.** Because fill accumulates upward, `lusup`, `lsub` and `ucol` end up far
larger than `nnz(A)`; the defaults `-50 / -50 / -30` are growth-factor guesses
(`|value| x nnz(A)`) about how much fatter the top of the tree gets than the
original matrix. Semantics of each value: negative = growth factor on
`nnz(A)`; positive = absolute entry count; `0` = use the `sp_ienv` default.
Section 5 covers what happens when the guess is wrong.

## 4. How the etree drives the threading

The scheduling unit is the **panel** (`panel_size`, `sp_ienv(1)`, default 20):
a run of consecutive columns symbolically processed and updated together.
`SRC/pxgstrf_scheduler.c` hands each thread a panel from a global work queue in
priority order:

1. the just-finished panel's **parent**, if runnable without pipelining — the
   thread's freshly computed data is still hot in its cache;
2. any **`CANGO`** panel: all etree descendants finished;
3. any **`CANPIPE`** panel: descendants still in flight, run *pipelined*.

Pipelining is where the synchronization machinery lives: `fb_cols` tracks each
panel's farthest busy descendant and a per-column busy array (`lbusy`) records
what other threads are still computing, so a thread can start a panel whose
dependencies are only partially complete and block on exactly the columns it
needs.

Supernodes enter as **shared, read-mostly data**: a thread updating its panel
reads supernodes produced by other threads; the dense contiguous layout makes
those reads cheap, and supernode completeness is what the pipelining guards.

At the top of the tree, where single updates become huge, `SRC/pdgstrf_bmod2D.c`
and `SRC/pdgstrf_panel_bmod.c` tile the update two-dimensionally with
`rowblk = sp_ienv(4)` and `colblk = sp_ienv(5)`, so several threads can
cooperate on one large supernode-panel update and the tiles are sized to cache.
`maxsuper`, `rowblk` and `colblk` are therefore coupled — a reason to tune them
as one experiment, not three knobs (the planned Phase 2).

## 5. How the blocking parameters are exposed

`relax`, `panel_size` and the `fill_*` estimates all reach the library through
`superlumt_options_t`. `maxsuper`, `rowblk` and `colblk` do not: they are read
through `sp_ienv()`, which takes no context argument, so there is no per-call
way to set them.

The bridge therefore supplies its own `sp_ienv()` definition
(`native/superlu_mt.c`). This is upstream's own pattern — `EXAMPLE/sp_ienv.c`
and `TESTING/sp_ienv.c` both replace the `SRC/` one — and it needs no change
to upstream. Because SuperLU_MT is linked as a static archive and
`sp_ienv.o` defines no other symbol, the linker resolves every internal
reference to the bridge's definition and never pulls in the upstream object.

Two consequences worth knowing:

- **The fallback values are a copy.** When no override is set, the bridge's
  `sp_ienv()` returns the generic (non-`MACH`) values from upstream
  `SRC/sp_ienv.c`: `20, 6, 200, 200, 100, -50, -50, -30`. A test asserts they
  still match, so an upstream change or a build that starts defining `MACH`
  surfaces as a failure rather than as silent drift.
- **The overrides are process-global.** `sp_ienv()` has no context, so the
  values live in file-scope variables that `slumt_factor()` sets before
  `pdgstrf()` and leaves alone while it runs. That is safe for one
  factorization at a time, which is how the FE solver uses the backend, but
  concurrent factorizations in a single process would race and are not
  supported.

## 6. When the storage guess is wrong

If the factors outgrow the allocated arrays mid-factorization, the expansion
path (`SRC/pmemory.c`, `XPAND_HINT`) prints
`Storage for ... exceeded; ... Need at least ...` and then calls
`superlu_abort_and_exit`, which is **`exit(-1)`**: the entire Python process is
killed — no exception, no `info` code, no restart file. The `fill_*` knobs are
the only in-process mitigation.

The converse failure is quieter: if the *initial* allocation request is too
large for available memory, `p?gstrf_MemInit` (`SRC/pdmemory.c`) halves
`nzlmax`/`nzumax` and retries, printing a warning to stderr (made unconditional
upstream in commit `a05e0a0`). A user who raises a fill factor and still hits
the abort should check stderr for that halving message: the request may never
have been granted.

## 7. One-paragraph summary

The etree orders elimination; fill-in flows up it; supernodes are the dense
blocks that fill-in creates along its branches — padded into existence at the
leaves (`relax`), capped near the root (`maxsuper`), updated by `GEMM`s tiled
to cache and threads (`panel_size`, `rowblk`, `colblk`), and stored in arrays
whose size is a guess about how much the tree fattens toward its root
(`fill_lusup`, `fill_ucol`, `fill_lsub`). Disjoint subtrees give threads
independent work; the narrow top of the tree is where parallelism, cache
behavior and memory growth all collide, and where every one of these knobs
earns its keep.
