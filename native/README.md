Brdige files for building against superlu_mt (multi-threaded version of superlu)

From the root of repository build and test using the followings:

```shell
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release\
  -Denable_internal_blaslib=ON\  
  -DTHREAD_API=PTHREAD \
  -DPLAT=_PTHREAD

cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

I recommend keeping `-Denable_internal_blaslib=ON` in the build since combined with the release build, it can often beat the system's BLAS with a non-negligible margin. On my test system with the benchmark matrix, a 16 thread run with CBLAS build was roughly 20% faster than BLAS build. 

Test the generated `.so` object with the followings

```shell
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

python benchmark_superlu_mt.py \
    kkt_debug.h5 \
    't_n=0.02/iteration=2' \
    --lib ../build/native/libsuperlu_mt_bridge.so \
    --threads 1 2 4 8 16 \
```

__NOTE__: The `.h5` files are tracked using `git lfs` and before running the benchmark, you should pull them using `git lfs pull`. 

TODO: One important question comes up here: 
  - Make sure that the above three exports are enforced inside the solver. Somehow running the solver without setting the three thread vars went through without problem, at least for the benchmark 16x16x16 mesh.
  - Compare these with the env variables that are set (if any) when running the element parallel pool
  - Do we need to make sure that these env variables are unset after the factor/solve is done? 
