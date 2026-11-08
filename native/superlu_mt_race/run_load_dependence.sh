#!/bin/sh
# Measure whether the SuperLU_MT wrong-solve rate depends on machine load.
#
# Motivation: on an idle machine, two user-run 5,000-repetition runs (8 and 16
# threads) produced zero wrong solves, while the same reproducer on the same
# host produced 3-5/5,000 at 8 threads whenever unrelated processes kept the
# load average near 6 -- across internal-CBLAS/OpenBLAS and -O3/-O0 builds, so
# neither BLAS nor optimization level explains the difference.  This script
# runs the reproducer with and without a synthetic background load and reports
# both rates side by side.
#
# Run it on an OTHERWISE IDLE machine (check with `uptime` first):
#
#   sh native/superlu_mt_race/run_load_dependence.sh [REPRO] [REPORT]
#
# REPRO  defaults to build/native/superlu_mt_race_repro
# REPORT defaults to superlu_mt_race_load_report.txt in the current directory.
#
# It uses up to 16 threads in the 16-thread phases and starts 8 busy-loop
# workers in the loaded phases, so it is a full-machine measurement: intended
# to be launched by the user, not by an agent (see AGENTS.md).

set -u

REPRO="${1:-build/native/superlu_mt_race_repro}"
REPORT="${2:-superlu_mt_race_load_report.txt}"
MATRIX="$(dirname "$0")/kkt_4x4x4_0017.mtx.gz"
REPS=5000
RUNS_PER_CONFIG=3
LOAD_WORKERS=8

[ -x "$REPRO" ] || { echo "reproducer not found or not executable: $REPRO" >&2; exit 2; }
[ -f "$MATRIX" ] || { echo "matrix not found: $MATRIX" >&2; exit 2; }

log() { printf '%s\n' "$*" | tee -a "$REPORT"; }

start_load() {
    LOAD_PIDS=""
    i=0
    while [ "$i" -lt "$LOAD_WORKERS" ]; do
        ( while :; do :; done ) &
        LOAD_PIDS="$LOAD_PIDS $!"
        i=$((i + 1))
    done
}

stop_load() {
    for pid in ${LOAD_PIDS:-}; do kill "$pid" 2>/dev/null; done
    wait ${LOAD_PIDS:-} 2>/dev/null
    LOAD_PIDS=""
}
trap stop_load EXIT INT TERM

run_config() {
    threads="$1"; condition="$2"
    run=1
    while [ "$run" -le "$RUNS_PER_CONFIG" ]; do
        summary=$(zcat "$MATRIX" | "$REPRO" "$threads" "$REPS" 2>&1 | grep '^summary:')
        log "threads=$threads load=$condition run=$run/$RUNS_PER_CONFIG  $summary"
        run=$((run + 1))
    done
}

: > "$REPORT"
log "SuperLU_MT wrong-solve load dependence  --  $(date -u '+%Y-%m-%d %H:%M UTC')"
log "host: $(uname -n), $(nproc) CPUs; reproducer: $REPRO"
log "pre-run load: $(uptime)"
log ""

log "== phase 1: idle machine =="
run_config 8 idle
run_config 16 idle
log ""

log "== phase 2: $LOAD_WORKERS busy-loop background workers =="
start_load
sleep 2
log "load during phase 2: $(uptime)"
run_config 8 loaded
run_config 16 loaded
stop_load
log ""
log "post-run load: $(uptime)"
log "done; report written to $REPORT"
