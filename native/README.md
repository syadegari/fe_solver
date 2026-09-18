Brdige files for building against superlu_mt (multi-threaded version of superlu)

From the root of repository build and test using the followings:

```shell
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHREAD_API=PTHREAD \
  -DPLAT=_PTHREAD

cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```
