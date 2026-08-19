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

All phases are complete. The full run is 134,406 generations on one rented
24 GB GPU.

| phase | state | output |
|---|---|---|
| 0. adapter geometry (laptop) | **done** | `results/phase0_findings.md` |
| 3a-prep. concept catalog + confound audit | **done** | `results/phase3a_style_confound.md` |
| 2. reproduce recall@k | **done** | `results/repro.json` |
| 3. the extension (40 concepts) | **done** | `results/analysis40.json` |
| write-up | **done** | `results/RESULTS.md`, `results/METHODOLOGY.md` |

Write-up: **[Reading the Activation, Not the Prior](https://mirayadav.github.io/projects/self-interp-adapter-probes)**

## Findings

**The reproduction holds.** 79.3% recall@1 over 49,637 candidate topics against
the paper's 82.9%, on a 1,000-topic held-out sample. The untrained baseline and
the bias-only baseline both sit at 0.0%.

**The description tracks controlled changes to the activation.** Adding a concept
direction moves the description toward that concept and subtracting it moves the
description away; a norm-matched random direction does nothing. Paired bootstrap
over 40 concepts, concept minus random:

| lambda | difference | 95% CI | p |
|---:|---:|---:|---:|
| -1.0 | -0.108 | [-0.130, -0.088] | <0.0001 |
| -0.3 | -0.019 | [-0.027, -0.013] | <0.0001 |
| 0.0 | -0.000 | [-0.000, +0.000] | 0.49 |
| +0.3 | +0.082 | [+0.068, +0.097] | <0.0001 |
| +1.0 | +0.516 | [+0.460, +0.569] | <0.0001 |

The zero row is the control that matters: both conditions run the same
computation there and agree to four decimals.

**It moves at the same strength that behaviour moves.** The self-interpretation
curve and the behavioural curve correlate at **r=0.997** with a mean absolute
difference of 0.022 (lambda50 0.517 vs 0.576). A description that only registered
gross changes would give a flatter, later curve.

**But the activation's identity still dominates.** Variance decomposition of the
description embeddings: prior 72.1% of a typical soft token; within the remaining
spread, topic identity 37.8%, concept 1.39%, lambda 0.74%, sampling noise 60.1%.

**Adapter geometry (laptop, no GPU).** `normalize_input=true`, so
`f(h) = alpha*(h/||h||) + b`: the instance term has *fixed norm*. For
`wikipedia-scalar-affine`, alpha=7.17 and ||b||=20.87, so the instance term is
**10.6%** of the soft token's second moment and no two interpretations can differ
by more than **40 degrees**. Closed form and simulation agree to 5 decimals.
This bounds the instance **budget**, not its **usage** — the paper resolves
1-in-50,000 topics inside that cone, so it predicted a non-flat `S(lambda)`.

**A confound that would have faked a positive result.** AxBench positives carry a
concept-independent style signature: a classifier on **held-out concepts**
separates positives from negatives at **0.969 AUC** (`"akin"`: 17.2% vs 0.0%;
positives +46% longer). A naive `mean_pos - mean_neg` would encode style, not the
concept — and would **survive the random-direction control**. Fixed by
mean-centring across concepts, plus a dedicated style-direction control. Measured
on the text the shared component looked like 23% of a typical vector; measured on
the actual Llama activations it was **96%**.

**Two errors caught during the run**, both documented in `results/RESULTS.md`: a
rank-against-alternatives control pinned to exactly 0.5 by symmetry (targets and
distractors were drawn from the same set), and a coherence gate that selected
against the effect it was meant to protect (kept concepts had a median behavioural
effect of 0.283 against 0.526 for those it excluded).

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
  make_figures.py    result charts -> results/fig_*.svg
  smoke_test.py      CPU plumbing test on Qwen2.5-0.5B
```

The write-up itself lives in the website repo, not here; this repo holds
the code, the data, and the numbers it reports.

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
python -m selfie_steering.selfie_sweep --n-concepts 40 --n-topics 30
python -m selfie_steering.analysis     --vectors results/concept_vectors.pt
```

## Reading the result

`concept` must beat **both** `random` and `style`. Wherever `concept` ~= `pure_v`,
`h` has stopped mattering and those lambdas say nothing about activation grounding.
A negative result is a real finding, not a failure.

## Credits

Upstream code and checkpoints: [agencyenterprise/selfie-adapters](https://github.com/agencyenterprise/selfie-adapters) (MIT).
Steering data: [AxBench CONCEPT500](https://huggingface.co/datasets/pyvene/axbench-concept500) (CC-BY-4.0).
Topics: [keenanpepper/fifty-thousand-things](https://huggingface.co/datasets/keenanpepper/fifty-thousand-things).
