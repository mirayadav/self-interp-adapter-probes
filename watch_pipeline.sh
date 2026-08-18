#!/usr/bin/env bash
# Emits one line per new pipeline step, plus failure signatures, then exits.
cd /Users/mirayadav/self-interp-adapter-probes
n_prev=0
quiet=0
while true; do
  steps=$(./pod.sh 'cat /workspace/PIPELINE_STEPS 2>/dev/null' 2>/dev/null || echo "")
  n=$(printf '%s\n' "$steps" | grep -c . || true)
  if [ "$n" -gt "$n_prev" ]; then
    printf '%s\n' "$steps" | tail -n $((n - n_prev)) | sed 's/^/STEP: /'
    n_prev=$n; quiet=0
  else
    quiet=$((quiet+1))
  fi
  # failure signatures in the log (emit once per occurrence window)
  errs=$(./pod.sh 'grep -hoE "Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|Killed|GatedRepoError|ConnectionError|AssertionError|FAILED:[A-Z]+" /workspace/pipeline.log 2>/dev/null | tail -3' 2>/dev/null || echo "")
  if [ -n "$errs" ] && [ "$errs" != "$last_errs" ]; then
    printf '%s\n' "$errs" | sed 's/^/ERROR: /'
    last_errs="$errs"
  fi
  case "$steps" in
    *PIPELINE_COMPLETE*) echo "DONE: pipeline complete"; exit 0;;
    *FAILED*)            echo "DONE: pipeline failed"; exit 1;;
  esac
  # hang detection: if nothing new for ~20 min, surface where it is
  if [ "$quiet" -ge 20 ]; then
    tail_line=$(./pod.sh 'tail -c 300 /workspace/pipeline.log 2>/dev/null | tr "\n" " " | tail -c 200' 2>/dev/null)
    gpu=$(./pod.sh 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' 2>/dev/null)
    echo "HEARTBEAT: no new step in ~20min | gpu=$gpu | log=...$tail_line"
    quiet=0
  fi
  sleep 60
done
