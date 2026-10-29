# SuperLU_MT bridge and benchmarks

Run all commands below from the repository root. Build the bridge and run the
upstream SuperLU_MT tests with:

```shell
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -Denable_internal_blaslib=ON \
  -DTHREAD_API=PTHREAD \
  -DPLAT=_PTHREAD

cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

I recommend keeping `-Denable_internal_blaslib=ON` in the build since combined with the release build, it can often beat the system's BLAS with a non-negligible margin. On my test system with the benchmark matrix, a 16 thread run with CBLAS build was roughly 20% faster than BLAS build. 

Retrieve the supplied KKT snapshots with Git LFS, then benchmark the generated
`build/native/libsuperlu_mt_bridge.so` against SciPy:

```shell
git lfs pull

OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python native/benchmark_superlu_mt.py \
    native/kkt_debug_1.h5 \
    't_n=0.02/iteration=2' \
    --lib build/native/libsuperlu_mt_bridge.so \
    --threads 1 2 4 8 16
```

The environment settings above apply only to that benchmark command. The FE
solver limits nested BLAS pools to one thread around SuperLU_MT factorization and
solve, and in each element worker. SuperLU_MT's own thread count remains explicit.
The Python test suite compares the built bridge against SciPy; this check is
skipped if the shared library has not been built.

## PETSc frozen-system benchmarks

From this folder, with PETSc/MPI installed in `py3.14`:

```shell
for N in 8 16; do
  HWLOC_COMPONENTS=-gl BENCH_NUM_THREADS=1 BENCH_MODE=validate \
    conda run --no-capture-output -n py3.14 mpiexec --bind-to none -n "$N" \
    python benchmark_petsc_kkt.py kkt_debug_1.h5 \
    --group 't_n=0.02/iteration=2' \
    -bench_fieldsplit_u_pc_asm_overlap 0
done
```

See [the diagnostic results](PETSC_KKT_DIAGNOSTIC_RESULTS.md) for the small
first-increment RHS/restart study and its reproduction commands. These are
frozen-system checks, not completed nonlinear FE validation.
