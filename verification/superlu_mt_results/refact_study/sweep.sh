#!/bin/bash
# Phase-0c: repeated interleaved fresh/refact0 samples.
#   sweep.sh PROBE DUMP REPS FILL NPROCS...
set -u
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
PROBE="$1"; DUMP="$2"; REPS="$3"; FILL="$4"; shift 4
echo "START $(date -Is) affinity=$(taskset -cp $$ | sed 's/.*: //') fill=$FILL"
for nprocs in "$@"; do
  for rep in $(seq 1 "$REPS"); do
    for mode in fresh refact0; do
      echo "----- $(date -Is) rep=$rep nprocs=$nprocs mode=$mode"
      stdbuf -oL "$PROBE" "$DUMP" "$nprocs" "$mode" "$FILL" 2>&1 \
        || echo "RESULT $mode nprocs=$nprocs fill=$FILL FAILED rc=$?"
    done
  done
done
echo "END $(date -Is)"
