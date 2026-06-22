#!/usr/bin/env bash
# Master orchestrator for all MBH campaigns. Designed to run FULLY DETACHED so it
# survives SSH/VS Code disconnects (launch with setsid+nohup, see bottom note).
#
# Runs sequentially (single GPU, and Gemini runs share one working dir so they
# must not overlap):
#   1. Gemini baseline loop runs 2..4  (run1 already done) -> _run{2,3,4}
#   2. Gemini neutral runs 1..5                            -> _neutral_run{1..5}
#   3. Kimi baseline (charged allowed)                     -> _original
#   4. Kimi neutral run 1                                  -> _neutral_run1
#
# Safe-by-design: every run writes to a shared per-model dir then gets renamed to
# a distinct dir; we refuse to overwrite an existing target.
set -u
cd /home/miquelaperez/Agentic-AI-for-Catalyst-Design

GEM="output/saturn_mbh/run_google_gemini-3.5-flash"
KIMI="output/saturn_mbh/run_moonshotai_kimi-k2.6"
RUN="conda run --no-capture-output -n ppchem python main.py --mode saturn-mbh"

log() { echo "[$(date '+%F %T')] $*"; }

run_and_save() {
  # $1 = working dir (BASE), $2 = destination dir, rest = extra CLI args
  local base="$1"; local dst="$2"; shift 2
  if [[ -e "$dst" ]]; then log "SKIP: $dst already exists."; return 0; fi
  if [[ -e "$base" ]]; then log "WARN: $base exists before run; moving aside."; mv "$base" "${base}_stale_$(date +%s)"; fi
  log "START: $dst   (args: $*)"
  $RUN "$@" || { log "FAIL: run for $dst failed (exit $?)."; return 1; }
  if [[ -e "$dst" ]]; then log "WARN: $dst appeared mid-run; not overwriting."; return 1; fi
  mv "$base" "$dst" && log "SAVED: $dst"
}

log "===== ORCHESTRATOR START ====="

# 1. Gemini baseline loop, runs 2..4 (run1 already complete).
for i in 2 3 4; do
  run_and_save "$GEM" "${GEM}_run${i}" --budget 500 || break
done

# 2. Gemini neutral runs 1..5 (MW<=250, rotbonds<=3, reject charged).
for i in 1 2 3 4 5; do
  run_and_save "$GEM" "${GEM}_neutral_run${i}" --budget 500 --require-neutral || break
done

# 3. Kimi baseline (charged allowed) -> _original.
run_and_save "$KIMI" "${KIMI}_original" \
  --model moonshotai/kimi-k2.6 --budget 1200 \
  --request-timeout 1800 --full-batch-attempts 2

# 4. Kimi neutral run 1.
run_and_save "$KIMI" "${KIMI}_neutral_run1" \
  --model moonshotai/kimi-k2.6 --budget 1200 \
  --request-timeout 1800 --full-batch-attempts 2 --require-neutral

log "===== ORCHESTRATOR DONE ====="
