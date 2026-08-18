# Phase 3a — AxBench style confound (found before spending any GPU time)

## The problem

AxBench positives are LLM-generated to express a concept; negatives are ordinary
responses. That difference leaves a **concept-independent style signature** in
every positive, so the naive steering vector

    v_c = mean(resid | positives of c) - mean(resid | negatives of c)

is contaminated by "AxBench synthetic-positive style" rather than isolating c.

Evidence (CPU only, `data/axbench/concept_pairs.parquet`, 300 concepts, TF-IDF +
logistic regression, **grouped CV so eval concepts are never seen in training**):

| measure | value |
|---|---|
| global pos-vs-neg ROC-AUC on **held-out concepts** | **0.969 ± 0.005** |
| `"akin"` occurrence | 17.2% of positives vs **0.0%** of negatives |
| `"just"` | 15.8% vs 0.6% |
| `"like"` | 29.3% vs 6.3% |
| mean output length | 501 chars vs 344 chars (**+46%**) |

A classifier that has never seen concept *c* still separates its positives from
its negatives at 0.97 AUC. That separation is pure style.

## Why it matters more than an ordinary nuisance

An uncorrected `v_c` would **manufacture a false positive for the paper**. The
sweep would show descriptions changing with lambda, the effect would be
direction-specific, and it would survive the norm-matched random-direction
control — because style is a *real, consistent* direction, just not the concept.
The headline claim ("SelfIE tracks controlled changes in h") would look
supported while actually demonstrating sensitivity to verbosity and register.
The random-direction control cannot catch this. It is the single most dangerous
artifact identified so far.

## The fix

Mean-centre the steering vectors across concepts — precisely the trick the paper
itself uses for topic vectors (`mean-vectors.safetensors` subtracts the mean
activation over ~50k topics):

    v_c  <-  (mean_pos_c - mean_neg_c) - mean_over_c'(mean_pos_c' - mean_neg_c')

Lexical proxy check (400 concepts, TF-IDF diff-of-means):

| | mean pairwise cos | own-label token in top-50 |
|---|---|---|
| raw | **+0.0512** | 88.8% |
| mean-centred | **-0.0024** | 86.8% |

The shared component is 23.3% of a typical vector's norm, its top tokens are
`like, akin, where, that, just, much` — style, not content — and removing it
drives distinct concepts to near-orthogonality while costing only 2 points of
concept identity. The fix is close to free.

## Three consequences for the plan

1. **Mean-centre every steering vector.** Non-optional.
2. **Add a style-direction control arm.** Sweep lambda along the shared component
   itself. It tests whether SelfIE reports *any* consistent direction or the
   concept specifically — a question the random-direction control cannot answer,
   since a random direction is not a direction the model has ever been trained to
   represent. Prediction: a faithful interpreter tracks concept directions more
   strongly than the style direction.
3. **Length-match positives and negatives** before extraction (+46% is large, and
   the activation is read at the last token, where length effects concentrate).

## Caveat

This is a **lexical** proxy (TF-IDF), not activation space. Dense 4096-d
activations may carry the style component far more strongly than sparse TF-IDF
cosines suggest. Re-verify in activation space during Phase 3a on the GPU:
report `||mean_c v_c|| / mean_c ||v_c||` and the pairwise-cosine table above,
recomputed on real residuals, before running any sweep.
