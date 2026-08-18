#!/usr/bin/env bash
# Full pipeline runner, designed to be launched detached on the pod:
#   setsid nohup bash run_all.sh > /workspace/pipeline.log 2>&1 < /dev/null &
# Writes a STEP marker after each phase so progress can be polled over SSH.
set -uo pipefail

REPO=/workspace/self-interp-adapter-probes
PY=/workspace/venv/bin/python
cd "$REPO"
export HF_HOME=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mark() { echo "$1" >> /workspace/PIPELINE_STEPS; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }
fail() { echo "FAILED:$1" >> /workspace/PIPELINE_STEPS; exit 1; }

mark "START"

# ---- 0. rebuild the AxBench pair table (73MB, regenerable, not synced) -------
if [ ! -f data/axbench/concept_pairs.parquet ]; then
  mark "CONCEPTS_BUILD"
  $PY selfie_steering/concepts.py --out-dir data/axbench || fail CONCEPTS
fi
mark "CONCEPTS_OK"

# ---- 1. Phase 2: reproduce recall@k (THE GATE) -------------------------------
mark "REPRO_START"
$PY -m selfie_steering.repro \
    --n-eval "${N_EVAL:-1000}" --n-gen 6 --batch-size 96 \
    --out results/repro.json --gen-out results/repro_generations.parquet \
  || fail REPRO
mark "REPRO_OK"

# ---- 2. Phase 3a: concept steering vectors ----------------------------------
mark "VECTORS_START"
$PY -m selfie_steering.vectors \
    --n-concepts "${N_CONCEPTS:-60}" --batch-size 24 --split-half \
    --out results/concept_vectors.pt \
  || fail VECTORS
mark "VECTORS_OK"

# ---- 3. Phase 3b: behavioural screening == B(lambda) ------------------------
mark "BEHAVIORAL_START"
$PY -m selfie_steering.behavioral \
    --vectors results/concept_vectors.pt --n-prompts 20 --batch-size 16 \
  || fail BEHAVIORAL
mark "BEHAVIORAL_OK"

# ---- 4. Phase 3c/3d: the lambda sweep ---------------------------------------
mark "SWEEP_START"
$PY -m selfie_steering.selfie_sweep \
    --n-concepts "${SWEEP_CONCEPTS:-10}" --n-topics "${SWEEP_TOPICS:-30}" \
    --n-gen 6 --batch-size 96 --out results/sweep.parquet \
  || fail SWEEP
mark "SWEEP_OK"

# ---- 5. Phase 3e: analysis ---------------------------------------------------
mark "ANALYSIS_START"
$PY -m selfie_steering.analysis \
    --sweep results/sweep.parquet --behavioral results/behavioral.parquet \
    --out results/analysis.json --curves-out results/curves.parquet \
  || fail ANALYSIS
mark "ANALYSIS_OK"

mark "PIPELINE_COMPLETE"
