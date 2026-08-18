# self-interp-adapter-probes

Reproduction of **"Self-interpretation in language models via adapter probes"**
(arXiv 2602.10352, AE Studio / AI Alignment Foundation) plus a steering-sensitivity
extension.

## The question

The paper trains a `d+1`-parameter adapter `f(h) = alpha*h + b` that maps a frozen
LM's internal activations into its own embedding space so the model can describe
them. It reports that the learned bias `b` alone accounts for **~85%** of the
improvement, reading `b` as a general interpretation prior while `h` supplies
instance-specific semantics.

The paper's evaluations show descriptions *correlate* with activations, but never
make a controlled change to a single activation and check that the description
responds. So prior-driven interpretation is not excluded:

> **How much of a SelfIE interpretation actually tracks controlled changes in `h`,
> versus being dominated by the learned prior `b`?**

Approach: take a steering direction `v` for concept X, independently validate that
it changes behaviour, then sweep `SelfIE(h + lambda*v)` bidirectionally and measure
whether the description moves toward X in step with the behavioural effect.

This is a **sensitivity test, not a causal test** of interpretation — steering
vectors are not clean semantic representations. Causal intervention on verified
circuits is the natural next step and is explicitly out of scope.

## Status

| phase | state | output |
|---|---|---|
| 0. adapter geometry (laptop) | **done** | `results/phase0_findings.md` |
| 3a-prep. concept catalog + confound audit | **done** | `results/phase3a_style_confound.md` |
| code drafts + smoke tests | **done** — all 6 modules run end-to-end on CPU | `selfie_steering/` |
| 2. reproduce recall@k | needs GPU | `results/repro.json` |
| 3. the extension | needs GPU | `results/analysis.json` |

## Findings so far (no GPU used)

**Adapter geometry.** `normalize_input=true`, so `f(h) = alpha*(h/||h||) + b`: the
instance term has *fixed norm*. For `wikipedia-scalar-affine`, alpha=7.17 and
||b||=20.87, so the instance term is **10.6%** of the soft token's second moment
and no two interpretations can differ by more than **40 degrees**. Closed form and
simulation agree to 5 decimals.

This bounds the instance **budget**, not its **usage** — the paper resolves 1-in-50,000
topics inside that cone, so the model must be highly sensitive to small angular
shifts. It therefore *predicts* a non-flat `S(lambda)`; a flat one would be the
surprising result.

**A confound that would have faked a positive result.** AxBench positives carry a
concept-independent style signature: a classifier on **held-out concepts** separates
positives from negatives at **0.969 AUC** (`"akin"`: 17.2% vs 0.0%; positives +46%
longer). A naive `mean_pos - mean_neg` would encode style, not the concept — and
would **survive the random-direction control**. Fixed by mean-centring across
concepts, plus a dedicated style-direction control arm. In activation space the
shared component is **0.977** of typical vector norm, far worse than the lexical
proxy suggested.

## Layout

```
selfie_steering/
  geometry.py        Phase 0 — adapter geometry (numpy only, no torch)
  concepts.py        build the AxBench concept catalog (~2000 concepts)
  select_concepts.py CPU pre-screen -> shortlist for the GPU screen
  core.py            model plumbing: extraction, SelfIE injection, steering hooks
  adapter.py         scalar-affine adapter load / random-init
  scoring.py         GTE embeddings, rank-based concept scoring, fluency
  vectors.py         Phase 3a — concept steering vectors + confound diagnostics
  behavioral.py      Phase 3b — behavioural validation == concept screening
  selfie_sweep.py    Phase 3c/3d — the lambda sweep and its control arms
  analysis.py        Phase 3e — S/T curves, bootstrap, variance decomposition
  repro.py           Phase 2 — recall@k reproduction
  smoke_test.py      CPU plumbing test on Qwen2.5-0.5B
```

## Environments

- `.venv`   Python 3.13, numpy/pandas only. Phase 0 + all data prep.
- `.venv311` Python 3.11 + `torch==2.2.2` (last Intel-Mac build), `numpy<2`.
  CPU smoke tests only.

Every GPU script takes `--smoke` to run on Qwen2.5-0.5B on CPU. Debug locally,
then run for real.

## Running

```bash
# no GPU
.venv/bin/python selfie_steering/geometry.py
.venv/bin/python selfie_steering/concepts.py
.venv/bin/python selfie_steering/select_concepts.py
.venv311/bin/python -m selfie_steering.analysis --selftest

# smoke (CPU)
PYTHONPATH=. .venv311/bin/python selfie_steering/smoke_test.py
PYTHONPATH=. .venv311/bin/python -m selfie_steering.vectors --smoke

# GPU (24GB is enough; Llama-3.1-8B bf16 ~16GB)
python -m selfie_steering.repro        --n-eval 1000
python -m selfie_steering.vectors      --n-concepts 60 --split-half
python -m selfie_steering.behavioral
python -m selfie_steering.selfie_sweep --n-concepts 10 --n-topics 30
python -m selfie_steering.analysis
```

## Reading the result

`concept` must beat **both** `random` and `style`. Wherever `concept` ~= `pure_v`,
`h` has stopped mattering and those lambdas say nothing about activation grounding.
A negative result is a real finding, not a failure.

## Credits

Upstream code and checkpoints: [agencyenterprise/selfie-adapters](https://github.com/agencyenterprise/selfie-adapters) (MIT).
Steering data: [AxBench CONCEPT500](https://huggingface.co/datasets/pyvene/axbench-concept500) (CC-BY-4.0).
Topics: [keenanpepper/fifty-thousand-things](https://huggingface.co/datasets/keenanpepper/fifty-thousand-things).
