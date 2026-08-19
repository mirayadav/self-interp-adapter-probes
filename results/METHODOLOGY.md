# Methodology, datasets, reproduction, and extension — full detail

Companion to `RESULTS.md`. Everything here is what was actually run, including the
things that went wrong and what was done about them.

---

## 1. Setup

| item | value |
|---|---|
| Model | `meta-llama/Meta-Llama-3.1-8B-Instruct` (gated; `hidden_size=4096`, `num_hidden_layers=32`) |
| Layer | 19, residual stream — read as `output_hidden_states[layer+1]` (index 0 is the embedding output) |
| Adapter | `keenanpepper/selfie-adapters-llama-3.1-8b-instruct` → `wikipedia-scalar-affine.safetensors` |
| Adapter form | `f(h) = α·(h/‖h‖) + b`, α = **7.175**, ‖b‖ = **20.869**, `normalize_input=true`, d+1 = 4097 params |
| Mean vector | `mean-vectors.safetensors` → `layer_19` (mean activation over the 50k-topic dataset) |
| Embedder | `thenlper/gte-large` (the paper's choice) — **the same model for every score in every experiment**: retrieval, behavioural screen, and λ-sweep |
| Hardware | 1 × NVIDIA A40 48GB, RunPod Secure Cloud, CA-MTL-1 |
| Stack | torch 2.8.0+cu128, transformers **5.15.0**, sentence-transformers 5.7.0 |
| Runtime | ~1h45m end to end; **79,206 generations** total |

**No adapter was trained.** The paper's released checkpoints are used as-is, so this
is a reproduction of their artifact, not of their training run.

**SelfIE injection.** The explanation-seeking prompt is the upstream template,
containing two `<|reserved_special_token_0|>` slots. The prompt is embedded, the
soft token `f(·)` is written into both slots, and generation runs from
`inputs_embeds`. Sampling: temperature 0.5, 40 new tokens, 6 generations per vector.

**Activation extraction convention** (kept identical to upstream so numbers stay
comparable):
- *Topic activations* `h`: prompt `"Tell me about {title}."` through the chat
  template with `add_generation_prompt=True`; take the **last token**.
- *Contrastive vector*: `h − mean_vector[19]`. This is what the adapter consumes.
- *Concept activations* (extension only): full user+assistant turn; **mean-pool over
  the response span only**, located by character offsets so the instruction tokens
  are excluded.

**One code change was required** for transformers 5.x: `from_pretrained(torch_dtype=)`
is renamed to `dtype=`. Handled by signature inspection so the code runs on both.

---

## 2. Datasets

### 2.1 `keenanpepper/fifty-thousand-things` — topics
49,637 Wikipedia vital-article topics; columns `original_title`, `prompt`, `labels`
(6–20 alternate descriptions, avg 17), `split` (44,673 train / 4,964 val). 55 MB,
public, no auth. Used for both the reproduction and as the base activations `h` in
the sweep.

Retrieval document per topic = `"{title}. " + first 8 labels`.

### 2.2 `pyvene/axbench-concept500` — steering-vector source
CC-BY-4.0. **Not used by the paper** — introduced here to supply concept directions.

Four partitions (`2b/l10`, `2b/l20`, `9b/l20`, `9b/l31`), each 500 concepts drawn
from GemmaScope SAE auto-interp labels. Per concept: 72 positives in `train`, plus
36 positives and 36 **concept-specific** negatives in `test`.

### 2.3 Adapters
Four released `.safetensors` (19 kB each): `wikipedia-scalar-affine`,
`goodfire-sae-scalar-affine`, `llamascope-sae-scalar-affine`, `mean-vectors`.
Public, no auth.

---

## 3. Dataset failure modes found, and what was done

These were found before any GPU time was spent, except (F7) which required it.

**F1 — `load_dataset` cannot read AxBench at all.**
`train` has 7 columns, `test` has 9 (`sae_link`, `sae_id` extra), so HuggingFace
`datasets` raises `CastError: Couldn't cast … column names don't match`.
*Fix:* read the eight parquet files directly with pandas, per partition.

**F2 — the pool is 2,000 concepts, not 500.**
The four partitions are near-disjoint: pairwise label overlap is 0–1 concepts,
union = 1,995 distinct labels across 2,000 rows. All four were pooled, giving 108
positives + 36 negatives per concept.

**F3 — style confound (the dangerous one).**
AxBench positives are LLM-generated to express a concept; negatives are ordinary
responses. A TF-IDF + logistic-regression classifier trained on some concepts and
evaluated on **held-out concepts** (grouped CV) separates positives from negatives
at **0.969 ± 0.005 AUC** — on style alone.

| token | in positives | in negatives |
|---|---|---|
| `"akin"` | 17.2% | **0.0%** |
| `"just"` | 15.8% | 0.6% |
| `"like"` | 29.3% | 6.3% |

*Why it matters:* a naive `mean_pos − mean_neg` vector would encode this register
rather than the concept — and because style is a *real, consistent* direction, it
would **pass a norm-matched random-direction control**. That is a false positive
for the paper's faithfulness, produced by an artifact of the steering data.

**F4 — instruction-pool confound.**
The instructions alone separate positives from negatives at **0.929 AUC**, and the
two instruction pools are **0% overlapping** within a concept (the pairs are
unmatched). Worse, instructions are concept-*matched*: a concept's instructions
predict its own label at 35.8% top-50 versus ~0.2% chance, so cross-concept
mean-centring cannot remove this component.
*Judgement:* less dangerous than F3, because instruction topic points in the *same*
semantic direction as the label — "prompts about parks" and "references to parks"
are not in conflict. Recorded as a scope limit: `v_c` blends "concept in the
response" with "topic of its associated prompts."

**F5 — length imbalance, and a fix that does not work.**
Positives are 46% longer (501 vs 344 chars; length alone gives 0.704 AUC). The
obvious remedy — truncating everything to the negative median — leaves style
separability essentially unchanged (**0.966** vs 0.969). Length-matching was
therefore **dropped from the plan**; it buys nothing here.

**F6 — genre is clean.** Text/code/math proportions are *identical* between
positives and negatives (0.6767 / 0.2500 / 0.0733 each). No genre confound.

**F7 — the lexical proxy understated the problem 4×.**
The checks above are all TF-IDF. Measured on real Llama layer-19 activations:

| | shared component / typical norm | raw mean pairwise cos |
|---|---|---|
| TF-IDF proxy | 0.233 | +0.051 |
| **Llama-3.1-8B L19** | **0.960** | **+0.919** |

Dense activations carry register far more strongly than sparse text statistics do.
Uncorrected, all 60 concept vectors would have been nearly the same vector.

*How the shared component is measured* (`vectors.py::diagnostics`):
`‖mean(V)‖ / mean(‖v‖)` — average the 60 raw vectors and compare the length of
that average to a typical individual length. For 60 *unrelated* vectors the
average shrinks to ~`1/sqrt(60)` = 0.129; we measured **0.960**, ~7x that. It
cross-checks against the pairwise cosine: `sqrt((1+(n-1)c)/n)` with c=0.919,
n=60 gives 0.959 vs the measured 0.960, so the two diagnostics are one fact.

Note 0.960 is a ratio of *lengths*, not energy — the residual is near-orthogonal
to the shared part, so what remains is `sqrt(1-0.96²)` = **28% by length, 8% by
energy**. Real signal, submerged under a ~9x larger common-mode component; the
split-half check (0.939) confirms the residual is signal rather than subtraction
noise.

### Corrections applied
1. **Cross-concept mean-centring**, `v_c ← v_c − mean_{c'}(v_{c'})` — the same
   operation the paper applies to topic vectors. Result: mean pairwise cosine
   **−0.0132** against the mechanical floor of −1/(n−1) = −0.0169 for n=60, i.e. an
   excess of +0.0037 — as orthogonal as centring permits.
2. **Positives-only construction**, `v_c = mean(resid | positives of c)`, rather
   than `pos − neg`. Both are statistically equivalent after centring, but
   positives-only never differences two disjoint instruction pools (F4), and it
   mirrors the paper's own topic-vector construction.
3. **Split-half stability check**: vectors rebuilt from alternating halves of the
   positives. Median cosine **0.939**, min 0.826, 100% above 0.5.
4. **A style-direction control arm** was added to every experiment (see §5.4).

### Concept filtering funnel

| stage | remaining | criterion |
|---|---|---|
| all AxBench concepts | 2,000 | — |
| genre = `text` | 1,341 | drops code/math |
| not surface-form | −180 flagged | keyword filter on the label: punctuation, capitalisation, tokens, grammar, `instances of the word "X"`. These are poor SelfIE targets — a null on them confounds "insensitive to `h`" with "has no vocabulary for this". |
| separability AUC ≥ 0.90 | 1,947 passed | per-concept TF-IDF CV AUC, positives vs its own negatives (median 0.988) |
| dedupe near-identical labels | — | normalised label string |
| **CPU shortlist** | **60** | top by AUC |
| vectors built + stability-checked | 60 | all passed |
| **behavioural screen: fluency preserved** | **11** | see §5.3 |
| **swept** | **10** | top 10 of the 11 by behavioural rise |

---

## 4. What was reproduced, exactly

**Scope, chosen deliberately:** only the **Wikipedia contrastive-vector retrieval**
result. *Not* reproduced: SAE detection/generation scoring, bridge-entity
extraction, the taboo baseline, the Qwen scaling curves, the adapter-architecture
sweep, or the LoRA comparison. None of them bear on the critique being tested.

**Protocol.** 1,000 topics sampled from the val split (seed 0). Retrieval index
built over **all 49,637** topics with GTE-large. For each topic: extract `h`,
subtract the mean vector, apply the adapter, inject, sample 6 descriptions, embed
each, and count a hit@k if **any** of the 6 lands the true topic in the top k
(best-of-6, matching the paper's multi-candidate protocol).

**How "identify the topic" is checked.** Nothing compares strings, and nothing
requires the description to name the topic:

1. Build an index once — each of the 49,637 topics becomes a document (title +
   first 8 alternate labels), embedded with GTE-large into a unit vector.
2. Embed each generated description the same way.
3. Cosine-similarity it against all 49,637 documents (a dot product, since
   everything is unit-length) and sort descending.
4. Find the rank of the true topic in that list; hit@k if rank < k.
5. Best-of-6 — the topic counts as a hit if *any* of its six descriptions hits.

So the columns are three cutoffs, against these chance rates:

| column | meaning | by luck |
|---|---|---|
| R@1 | the single nearest of 49,637 is correct | 0.002% |
| R@5 | correct topic among the 5 nearest | 0.010% |
| R@100 | among the 100 nearest | 0.201% |

The untrained arm's 2.0% at R@100 is ~10x chance but still negligible.

Worked examples from `results/repro_generations.parquet`:

- **Robert Menzies** — *hit*. Five of six say "Robert Menzies, Australian Prime
  Minister from 1939 to 1941 and 1949 to 1966"; embeds almost onto his document.
- **John Dory** — *hit*. "the fish John Dory" is short but unambiguous.
- **Chu Suiliang** — *miss*. All six say "Chen Shou, the Eastern Jin historian
  who compiled the Records of the Three Kingdoms" — a real person, but the wrong
  one, so it retrieves Chen Shou instead. This is what the missing 20.7% looks
  like: fluent, confident, plausibly Chinese-historical, and wrong.

Two caveats on the metric. **Best-of-6 flatters it** — single-shot accuracy would
be noticeably lower; the paper also uses 6 candidates (varying injection scale
where we vary by sampling), so the comparison to their figure is fair. And
because matching is semantic, a description that never names the topic can still
rank it first — which is the point, since the model is describing an activation.

Three arms, identical topics and seeds:
- `trained` — the released adapter
- `untrained` — `f(h) = 1.0 · h/‖h‖`, i.e. scale-only at unit scale (the paper's
  untrained baseline)
- `bias_only` — `f(0) = b`, feeding a zero vector

### Results

| arm | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---|---|---|---|---|---|
| **trained** | **79.3%** | 86.6% | 89.2% | 91.7% | 94.5% | **95.9%** |
| untrained | 0.0% | 0.1% | 0.4% | 1.1% | 1.5% | 2.0% |
| bias-only | 0.0% | 0.0% | 0.1% | 0.4% | 0.7% | 1.0% |

### Comparison to the paper

| quantity | paper | here | delta |
|---|---|---|---|
| trained R@1 (no scale tuning) | 82.9% | 79.3% | −3.6 pt |
| trained R@100 | 98.4% | 95.9% | −2.5 pt |
| untrained R@1 | 0.04% | 0.0% | ≈0 |

**Verdict: reproduces.** Both trained figures land a few points under the paper's.
Plausible causes, none of which were chased: a 1,000-topic subsample versus their
full validation set, sampling variance at temperature 0.5, and possible differences
in how the topic document is assembled for the index (we use title + 8 labels).
The qualitative claim — ~80% versus ~0% — is fully reproduced.

---

## 5. The extension — every experiment not in the paper

The paper shows descriptions *correlate* with activations. None of the following
appear in it; each was added to test whether descriptions *respond* to a controlled
change in the activation.

### 5.1 Adapter geometry in closed form (no GPU)

**Why.** The paper reports that the bias accounts for ~85% of the improvement but
never characterises the geometry that claim implies.

**Method.** `normalize_input=true` means the instance term `α·(h/‖h‖)` has *fixed
norm*, so every soft token lies on a sphere of radius α centred at `b`. Both
quantities come from the released 19 kB checkpoint alone, no model run needed.

*Share.* Squaring gives `‖f‖² = α² + 2α⟨ĥ,b⟩ + ‖b‖²`. Averaged over activation
directions the cross term vanishes — `b` is one fixed vector while `ĥ` ranges over
the whole sphere, pointing with and against it equally — leaving an orthogonal
split `α² + ‖b‖²`. Share = `α²/(α²+‖b‖²)` = 51.48/486.98 = **10.6%**.

*Gap.* Viewed from the origin, the furthest a direction can swing is where the
line of sight is tangent to that sphere: `sin θ = α/‖b‖` → θ = 20.11°, so two soft
tokens differ by at most `2θ` = **40.2°**. Requires `α < ‖b‖` (origin outside the
sphere), which holds: 7.17 < 20.87.

**Only the ratio is interpretable.** α = 7.175 and ‖b‖ = 20.869 are in arbitrary
units — there is no canonical scale for the model's internal space — so neither is
meaningful on its own. Both derived quantities are functions of `r = α/‖b‖ = 0.344`
and nothing else: share = `r²/(1+r²)`, gap = `2·arcsin(r)`. Quote the ratio, not
the raw sizes.

| r = α/‖b‖ | share | widest gap |
|---|---|---|
| 0.10 | 1.0% | 11.5° |
| 0.20 | 3.8% | 23.1° |
| **0.344 (actual)** | **10.6%** | **40.2°** |
| 0.50 | 20.0% | 60.0° |
| 0.90 | 44.8% | 128.3° |

In d=4096 random directions are near-orthogonal, so
`E[cos(f(h_i), f(h_j))] = ‖b‖²/(α²+‖b‖²)` — checked empirically below.

**Result.**

| adapter | α | ‖b‖ | instance share | max angle | predicted E[cos] | empirical (1000 dirs) |
|---|---|---|---|---|---|---|
| `wikipedia-scalar-affine` | 7.175 | 20.869 | **10.6%** | **40.2°** | 0.89429 | 0.89426 |
| `goodfire-sae-scalar-affine` | 4.032 | 19.750 | 4.0% | 23.6° | 0.95998 | 0.96000 |
| `llamascope-sae-scalar-affine` | 4.021 | 19.008 | 4.3% | 24.4° | 0.95718 | 0.95717 |

Theory and simulation agree to 5 decimals; empirical spread sits inside the
analytic `arcsin` bound.

**Meaning.** Every one of ~50,000 interpretations is launched from inside a single
**40° cone**. Crucially this bounds the instance **budget**, not its **usage** —
§4 shows the model resolves 1-in-49,637 *inside* that cone. So the geometry alone
cannot settle the question, which is exactly why the sweep is needed. It also makes
a falsifiable prediction: a model this sensitive to small angular displacement
*should* register a steering vector, so a flat `S(λ)` would be the surprise.

### 5.2 Bias-only retrieval

**Why.** The paper notes that applying the adapter to a zero vector yields generic
descriptions matching the training distribution, but does not score them.

**Method.** Run the full retrieval protocol on `f(0) = b`.

**Result.** **0.0% R@1**, 1.0% R@100 — yet the outputs are fluent, confident and
specific: *"the director of the 1967 film Blow-Up"*, *"the journalist who coined the
term 'tropism'"*, *"the opera singer whose voice was used in the recording of the
Hallelujah Chorus"*.

**Meaning.** This is the sharpest qualification of the paper's own framing. "The
bias accounts for ~85% of the improvement" is fully compatible with the bias
carrying **zero instance information**: it supplies format, register and confident
fluency, and the instance term does all the identifying. It also shows the prior is
an active confabulator, not a neutral scaffold — relevant to the paper's own
deceptive-alignment motivation.

### 5.3 Behavioural validation, doubling as concept screening

**Why.** The extension's claim is `intervention → internal change → SelfIE report`.
Without independent evidence that the intervention changes *behaviour*, the sweep
would be pure semantic correlation. Screening and measurement are the same
operation, so it was run once.

**Method.** A forward hook adds `λ·‖resid‖_typ·v̂` to the layer-19 residual stream at
all token positions during `.generate()` (CAA-style), for 20 neutral prompts, over
λ ∈ {−2, −1.5, −1, −0.6, −0.3, 0, 0.3, 0.6, 1, 1.5, 2}. λ is expressed in units of
the typical residual norm so the behavioural and SelfIE curves share an x-axis.
Scored with the same rank-against-40-distractors metric, plus `distinct-2` and
per-token NLL under the *unsteered* model as fluency guards.

**Result** (mean over the 60 shortlisted concepts):

| λ | B(λ) rank | distinct-2 | NLL/token |
|---|---|---|---|
| −2.0 | 0.362 | 0.657 | 2.94 |
| −1.0 | 0.390 | 0.963 | 2.00 |
| 0.0 | 0.502 | 0.984 | 1.28 |
| +0.6 | 0.754 | 0.977 | 1.80 |
| +1.0 | 0.951 | 0.889 | 2.59 |
| +2.0 | 0.957 | 0.522 | 2.99 |

The **keyword rate** — pure substring matching, no embeddings — traces the same
curve (0.039 / 0.350 / 0.598 / **0.874** / 0.769 at λ = −2 / 0 / +0.6 / +1 / +2),
so the behavioural effect is not an artifact of GTE-large. This matters because
the same embedder scores both the screen that *selects* concepts and the sweep
that *measures* the effect.

**Only `distinct2` gated anything.** `fluency_ok` requires distinct2 > 0.6× its
λ=0 value at every λ; `nll_per_token` and `keyword` were recorded for
interpretation but never used to filter. The 60 → 11 cut was distinct2 alone.

Steering works, strongly and monotonically. **But fluency degrades with |λ|**:
distinct-2 falls from 0.984 to 0.522 and NLL more than doubles by λ=2. Only **11 of
60** concepts kept fluency within the threshold across the whole grid; the sweep
used the top 10 of those by behavioural rise.

**Meaning.** The intervention is behaviourally real, and the informative band is
|λ| ≲ 1 where the model is still producing coherent text. Effects at λ=2 are
partly the model breaking down.

### 5.4 The λ-sweep, with four arms

**Why.** This is the core experiment: does the description track a controlled,
independently validated change to `h`?

**Method.** 10 screened concepts × 30 held-out topics × 11 λ × 6 generations
(19,800 per arm, 79,206 total including the bias-only floor). For each:
`soft = f(h_contrastive + λ‖h‖v̂)`. Injection scale is **held fixed across λ** so
the paper's best-of-n scale search cannot be confused with the λ effect.

Scored as **rank against a fixed pool of other concept labels** rather than raw
cosine — raw cosine drifts upward whenever descriptions merely get longer or more
generic, which would read as a false positive.

Four arms:

| arm | direction | what it isolates |
|---|---|---|
| `concept` | `v̂_c` (centred) | the effect under test |
| `random` | norm-matched random | direction-specificity vs generic perturbation response |
| `style` | the shared component removed in §3 | response to *any real consistent* direction the model represents — a random direction cannot test this |
| `pure_v` | `λ‖h‖v̂` with **no `h`** | degeneracy: where `concept ≈ pure_v`, `h` no longer matters |

**Pool size — correction.** The intent was 40 distractors, and §5.3's behavioural
screen did use 40 (drawn from all 60 screened concepts). But `analysis.py` built
the sweep's pool from the labels present in `sweep.parquet`, which contains only
the 10 swept concepts — so S(λ) is a **9-way** comparison, not 40-way. The pool
is drawn once per concept with a fixed seed and reused across every topic, λ and
arm. Nothing enforces semantic distance: the 10 include near-duplicates
(*"calls to action for clicking links"* vs *"phrases that prompt clicking on
links"*, cosine 0.62). Smaller pool inflates absolute values; near-duplicates
deflate them; both apply uniformly across λ and to all four arms, so the shape
and the flat controls are unaffected. Fixed in `analysis.py` via `--vectors`,
which widens the pool to all 60 screened concepts and warns when short.

**Result — S(λ), rank vs the distractor pool (0.5 = chance):**

| λ | concept | random | style | pure_v |
|---|---|---|---|---|
| −2.0 | **0.3510** | 0.5087 | 0.5000 | 0.3131 |
| −1.5 | 0.3763 | 0.5090 | 0.5000 | 0.3151 |
| −1.0 | **0.4149** | 0.5072 | 0.5000 | 0.3131 |
| −0.6 | 0.4556 | 0.5014 | 0.5000 | 0.3143 |
| −0.3 | 0.4792 | 0.4984 | 0.5000 | 0.3142 |
| 0.0 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| +0.3 | **0.5584** | 0.5037 | 0.5000 | 0.7957 |
| +0.6 | 0.6405 | 0.4996 | 0.5000 | 0.7957 |
| +1.0 | **0.7227** | 0.5041 | 0.5000 | 0.7960 |
| +1.5 | 0.7599 | 0.5011 | 0.5000 | 0.7960 |
| +2.0 | **0.7788** | 0.5018 | 0.5001 | 0.7960 |

**T(λ) — similarity to the *original* topic:** concept falls 0.9084 → 0.7641 across
λ=0→2, versus 0.8022 (random) and 0.7975 (style). Concept-ness rises *while*
topic-ness falls, and falls further than controls.

**Meaning.**
1. `S(+λ) > S(0) > S(−λ)` — monotone and **bidirectional**. Adding the direction
   makes the description more like the concept; subtracting it makes it less.
2. `random` flat ⇒ direction-specific, not a magnitude response. The control that
   could have killed the result does not.
3. `style` flat ⇒ the model does not drift concept-ward for *any* real direction,
   only this one. This is the control the random arm cannot substitute for, and it
   exists only because the audit in §3 found the confound.
4. `pure_v` saturates at λ=0.3 and never moves, while `concept` climbs gradually
   (0.558 → 0.640 → 0.723). At λ=0.3, concept 0.558 ≪ pure_v 0.796, so `h` still
   dominates; by λ=2 they converge and `h` is washed out. **The informative regime
   is λ ≲ 1.**

Because the adapter L2-normalises its input, adding `λv` can only *rotate* the
direction — never change its magnitude — so no norm confound can explain any of it.

### 5.5 Sensitivity comparison

**Why.** The point of the extension: does the self-report respond at the same
intervention strength that changes behaviour?

**Method.** λ at half-max of each normalised curve, linearly interpolated.

| curve | λ₅₀ |
|---|---|
| **SelfIE (concept)** | **0.596** |
| **behavioural** | **0.576** |
| style | 1.75 (flat/noisy) |
| random | — (flat) |
| pure_v | 0.150 (saturates instantly) |

**Meaning.** A 3% difference. SelfIE reports the change at essentially the same
steering strength that changes the model's behaviour, and both sit inside the band
where fluency is preserved (distinct-2 ≈ 0.98 at λ=0.6). This is the closest the
experiment gets to `intervention → internal change → report`.

### 5.6 Variance decomposition

**Why.** To answer "how much of the output is responsive to `h`" as one number —
the direct quantitative counterpart to the paper's 85% claim, in *output* space.

**Method.** Description embeddings are unit-norm, so the second moment about the
origin splits exactly as `1 = ‖ē‖² + Var`. The variance term is then partitioned
factorially over concept, topic and λ (balanced design). Validated beforehand
against synthetic data with closed-form ground truth, including a null case that
must not fire (`--selftest`: predicted prior share 0.9493 vs measured 0.9553;
null-case λ share 0.005).

**Result** (concept arm, 19,800 generations):

| component | share |
|---|---|
| prior / constant | **71.8%** of second moment |
| responsive (variance) | 28.2% |
| ↳ topic identity | 31.4% of variance |
| ↳ concept identity | 1.2% |
| ↳ λ along v | **0.75%** |
| ↳ residual (sampling noise) | 66.6% |

**Meaning.** The λ effect is real, monotone, direction-specific and behaviourally
calibrated — and **small**. Descriptions stay dominated by which topic `h` came from
(31.4%) and by sampling noise at temperature 0.5 (66.6%). Anyone who cares about
effect size rather than existence should weight this over §5.4.

---

## 6. Limitations, stated plainly

- **Not causal.** Steering vectors are not clean semantic representations. This
  tests sensitivity to a controlled, independently validated change — nothing more.
- **Selection effect, twice over.** Concepts were filtered for lexical separability
  (§3) *and* chosen because they steer well and preserve fluency (§5.3). Results
  describe the favourable case.
- **Weak concepts in the swept set.** Only **5 of the 10** swept concepts had
  behavioural rise > 0.3 (median 0.377), and the two weakest (0.240, 0.219) are not
  far above the style control's 0.153. The fluency filter (11 survivors of 60) bound
  harder than the steering-strength filter, so the final set trades effect size for
  coherence. All 10 did have high monotonicity (0.86–0.98).
- **Behavioural control arms are not absolutely comparable.** `__STYLE__` and
  `__RANDOM__` were scored against a placeholder label ("generic descriptive text"),
  so their absolute rank values (0.75–0.98) mean nothing; only their *rise* (0.153
  and −0.049, versus up to 0.945 for concepts) is interpretable.
- **Topic displacement is partly generic.** T(λ) also declines for random and style,
  so not all of the concept arm's decline is concept substitution.
- **Instruction confound persists** (F4) and cannot be removed by centring; `v_c`
  blends concept-in-response with prompt topic.
- **Narrow.** One model, one layer, one adapter architecture, 10 concepts, 30 topics.
- **Reproduction is partial by design** — only the retrieval result, on a subsample.

## 7. What would strengthen this next

In rough order of value: an instruction-only control arm (built from prompt-only
forward passes) to bound F4 empirically; the same sweep on `wikipedia-full-rank` and
the SAE adapters, whose instance share is less than half (§5.1); more concepts, to
turn the 0.75% variance figure into something with an interval; and the genuinely
stronger test — intervening on features or circuits whose causal role is
independently established, and asking whether the self-report tracks *that*.

---

## 8. Implementation

### 8.1 Code layout

~2,000 lines across 12 modules. Nothing from the upstream repo is vendored; it was
cloned for reference only, and the pieces we needed (adapter format, injection
template, extraction convention) were reimplemented so the pipeline runs without
its GPU-heavy dependency set (`sae-lens`, `nnsight`, `vllm`).

| file | lines | role |
|---|---|---|
| `selfie_steering/core.py` | 282 | model loading, activation extraction (last / mean / response-span pooling), SelfIE soft-token injection, steering hooks |
| `selfie_steering/analysis.py` | 314 | S/T curves, group-level bootstrap, variance decomposition, `--selftest` |
| `selfie_steering/vectors.py` | 179 | concept vectors, cross-concept centring, activation-space confound diagnostics |
| `selfie_steering/behavioral.py` | 176 | steering hook sweep, B(λ), fluency, concept screening |
| `selfie_steering/selfie_sweep.py` | 174 | the λ sweep and its four arms |
| `selfie_steering/repro.py` | 146 | Phase 2 recall@k reproduction |
| `selfie_steering/geometry.py` | 143 | Phase 0, numpy only — no torch |
| `selfie_steering/scoring.py` | 137 | GTE embeddings, rank-vs-distractors, fluency metrics |
| `selfie_steering/select_concepts.py` | 129 | CPU concept pre-screen |
| `selfie_steering/smoke_test.py` | 102 | CPU plumbing test |
| `selfie_steering/concepts.py` | 100 | AxBench catalog build |
| `selfie_steering/adapter.py` | 71 | scalar-affine load / random-init |
| `run_all.sh` | 64 | pod-side orchestration with step markers |
| `pod.sh` | 9 | SSH/scp wrapper (port lives in one place) |

**Key implementation choices**

- **`Adapter` reads the released `.safetensors` directly** — tensors `bias` (4096,)
  and `log_scale` (1,), metadata `normalize_input`, `projection_type` — rather than
  importing `selfie_adapters`. Removes a dependency and lets `Adapter.random_like()`
  fabricate an untrained adapter at the measured geometry (α=7.17, ‖b‖=20.87) for
  smoke runs on a model with a different `d`.
- **Model-agnostic SelfIE template.** Upstream hard-codes Llama's
  `<|reserved_special_token_0|>`. `build_selfie_template()` derives an equivalent
  prompt from any tokenizer's own chat template, falling back to another
  single-token placeholder. This is what makes the CPU smoke path possible.
- **`selfie_describe` batches across soft tokens.** The first version issued one
  `generate()` per vector; flattening `(B, n)` into one batched call is what makes
  79,206 generations finish in under two hours.
- **Steering via a context-managed forward hook** on the layer's module, handling
  both tuple and tensor block outputs, with the handle removed in a `finally`. The
  smoke test explicitly asserts the hook is removed cleanly and that `coeff=0` is a
  true no-op — both silent-corruption failure modes.
- **Rank-based scoring, not cosine**, for the reason in §5.4.

### 8.2 Two local environments

The 2019 Intel Mac has no CUDA and no MPS, and PyTorch ships no macOS x86_64 wheel
past 2.2.x (cp311 max). So:

- **`.venv`** — system Python 3.13, `numpy` / `pandas` / `safetensors` /
  `huggingface_hub` / `scikit-learn`. No torch. Ran **all of Phase 0 and the entire
  dataset audit** — `safetensors.numpy.load_file` reads the checkpoints without
  torch at all.
- **`.venv311`** — `brew install python@3.11`, then `torch==2.2.2` with
  `numpy<2` (the 2.2.2 binary is built against numpy 1.x and segfaults otherwise)
  and `transformers==4.46.3`. CPU smoke tests only.

Every GPU script takes `--smoke`, which swaps in **Qwen2.5-0.5B-Instruct** (ungated,
~1 GB, d=896) with a random adapter on CPU. Outputs are nonsense; the point is
shape, hook-placement and template validation before spending GPU time. The full
chain — vectors → behavioral → sweep → analysis, plus repro — was run end to end
this way locally first.

### 8.3 RunPod

**Provisioning** was done through the RunPod MCP server rather than the console.
Selection was driven by a constraint that only shows up if you check: **no 48GB GPU
has a network volume in its region.** A40 exists only in CA-MTL-1 and EU-SE-1,
neither of which supports network volumes; A6000 likewise; L40S has one region with
volumes but at 2–3× the price and LOW stock. The only HIGH-stock GPU in a
volume-capable region is the 24GB RTX 4090.

Chosen: **A40 48GB, CA-MTL-1, Secure Cloud, $0.44/hr, no network volume** — VRAM
headroom over persistence, since re-setup is ~15 minutes and scriptable while a
24GB ceiling is permanent. Container disk 20GB, pod volume 100GB at `/workspace`,
image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.

```
create-pod  name=selfie-adapter-probes  gpuTypeIds=["NVIDIA A40"]
            dataCenterIds=["CA-MTL-1"]  cloudType=SECURE
            containerDiskInGb=20  volumeInGb=100  volumeMountPath=/workspace
            ports=["22/tcp","8888/http"]  sshPublicKey=<ed25519 public key>
            env={HF_HOME:/workspace/hf_cache, HF_HUB_ENABLE_HF_TRANSFER:1}
```

**Credentials.** The HF token was supplied as a RunPod **Secret**, referenced as
`HF_TOKEN={{ RUNPOD_SECRET_hf_token }}` in the pod env, so it never entered the
session transcript. The account-scoped OAuth token on the laptop was deliberately
*not* copied to rented hardware.

**Code transfer** by `rsync` over SSH (excluding venvs, `.git`, and the 73MB
`concept_pairs.parquet`, which `concepts.py` regenerates from HuggingFace in about a
minute). Results came back the same way.

**Execution.** Everything ran through `run_all.sh`, launched detached:

```bash
setsid nohup env N_EVAL=1000 N_CONCEPTS=60 SWEEP_CONCEPTS=10 SWEEP_TOPICS=30 \
  bash run_all.sh > /workspace/pipeline.log 2>&1 < /dev/null &
```

The script appends a marker to `/workspace/PIPELINE_STEPS` after each phase
(`REPRO_START`, `REPRO_OK`, …, `PIPELINE_COMPLETE`, or `FAILED:<phase>`), so progress
is a one-line SSH poll rather than a held-open connection. Total: 07:52 → 09:37 UTC.

**Cost.** ~1h45m at $0.44/hr ≈ **$0.80** for all 79,206 generations.

### 8.4 Six things that cost real time

1. **`update-pod` replaces the env wholesale — it does not merge.** Setting
   `HF_TOKEN` while omitting `PUBLIC_KEY` deleted the variable the image uses to
   install the SSH key and start sshd. Instant lockout; two restarts to recover.
   Always resend every variable.
2. **Pod env vars are absent from SSH sessions.** They live on PID 1. `echo $HF_TOKEN`
   returned empty and looked exactly like an unresolved secret; the secret had in
   fact resolved correctly all along. Read them with
   `tr '\0' '\n' < /proc/1/environ`, and bridge them into shells via
   `/etc/profile.d/`.
3. **The public SSH port changes on every restart** (22104 → 22019 → 22015 → 22179).
   Re-read it from `get-pod` after any restart.
4. **Container disk resets on restart; only `/workspace` persists.** `/etc/profile.d`
   and the sshd hardening had to be re-applied.
5. **`/workspace` is MooseFS network storage.** 453 MB/s sequential (fine for a 16GB
   model) but very slow on many-small-file work — `pip install` there took ~20
   minutes and twice appeared to hang. Container disk `/` is 1.5 GB/s but only 20GB.
   Related: PEP 668 blocks system-wide pip on this image, so a venv is required; and
   killing a duplicate pip mid-install corrupted the shared venv and forced a clean
   rebuild.
6. **`rsync -a` fails on `/workspace`** with `chown … Operation not permitted`. Use
   `scp`, or `rsync -rlptz --no-owner --no-group`. (macOS ships rsync 2.6.9, which
   also lacks `--info=stats2`.)

For any long pod job: `setsid nohup … &` plus a completion-marker file. Plain
`nohup` over SSH dies when the session drops, which is how the first two installs
were lost.

### 8.5 Reproducing this run

```bash
# laptop, no GPU, no torch
.venv/bin/python selfie_steering/geometry.py          # Phase 0
.venv/bin/python selfie_steering/concepts.py          # AxBench catalog (~1 min)
.venv/bin/python selfie_steering/select_concepts.py   # CPU pre-screen -> 60
.venv311/bin/python -m selfie_steering.analysis --selftest

# CPU smoke, Qwen2.5-0.5B
PYTHONPATH=. .venv311/bin/python selfie_steering/smoke_test.py
PYTHONPATH=. .venv311/bin/python -m selfie_steering.vectors --smoke

# GPU pod (24GB suffices; A40 48GB used here)
python -m selfie_steering.repro        --n-eval 1000 --n-gen 6 --batch-size 96
python -m selfie_steering.vectors      --n-concepts 60 --batch-size 24 --split-half
python -m selfie_steering.behavioral   --n-prompts 20 --batch-size 16
python -m selfie_steering.selfie_sweep --n-concepts 10 --n-topics 30 --batch-size 96
python -m selfie_steering.analysis
```

Seeds are fixed at 0 throughout (topic sampling, distractor-pool selection, random
control direction, generation). Sampling at temperature 0.5 is still stochastic, so
exact strings will differ between runs; the aggregate curves should not.
