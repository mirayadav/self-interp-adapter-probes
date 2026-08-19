# Results — does SelfIE track the activation, or the prior?

Llama-3.1-8B-Instruct, layer 19, released `wikipedia-scalar-affine` adapter.
79,206 generations. Pod: 1×A40, ~1h45m, ~$0.80 of compute.

---

## 1. Reproduction gate (Phase 2) — PASSED

1,000 held-out topics, retrieval index over all 49,637, best-of-6 generations.

| arm | R@1 | R@5 | R@10 | R@100 |
|---|---|---|---|---|
| **trained** | **79.3%** | 86.6% | 89.2% | 95.9% |
| untrained (`f(h)=h/‖h‖`) | 0.0% | 0.1% | 0.4% | 2.0% |
| **bias-only** (`f(0)=b`) | **0.0%** | 0.0% | 0.1% | 1.0% |

Paper: 82.9% R@1 / 98.4% R@100 without scale tuning, ~0.04% untrained. We land a
few points under on a subsample — the pipeline reproduces.

**The bias-only arm is the first real result.** It scores **0.0% R@1** while
producing fluent, confident, specific text:

> "the director of the 1967 film Blow-Up"
> "the journalist who coined the term 'tropism'"
> "the opera singer whose voice was used in the recording of the Hallelujah Chorus"

These come from a **zero input vector**. The prior alone invents plausible,
entirely ungrounded descriptions and identifies nothing. So whatever the bias
contributes, it is not topic information.

## 2. Adapter geometry (Phase 0, no GPU)

`normalize_input=true` ⇒ `f(h) = α·(h/‖h‖) + b`, so the instance term has fixed
norm. α=7.17, ‖b‖=20.87 ⇒ instance is **10.6%** of the soft token's second
moment; no two interpretations can differ by more than **40°**. Theory and
simulation agreed to 5 decimals.

This bounds the instance **budget**, not its **usage** — and §1 shows the model
resolves 1-in-49,637 inside that cone while the prior alone resolves nothing.

## 3. The confound that would have faked a positive

AxBench positives carry a concept-independent style signature: a classifier on
**held-out concepts** separates positives from negatives at **0.969 AUC**
(`"akin"` 17.2% vs 0.0%; positives +46% longer). Instructions alone separate at
0.929 AUC.

In **activation space this was far worse than the lexical proxy predicted**:

| | shared component | raw pairwise cos |
|---|---|---|
| lexical (TF-IDF) proxy | 0.233 | +0.051 |
| **real Llama layer-19** | **0.960** | **+0.919** |

Uncorrected, every `v_c` would have been ~the same vector, and the sweep would
have shown a strong, direction-specific effect that **passes the random-direction
control** — a false positive for the paper. Fixed by cross-concept mean-centring
(→ −0.013 against a mechanical floor of −0.017). Split-half stability of the
resulting vectors: median **0.939**, min 0.826, all >0.5.

## 4. Main result — S(λ), concept-ness of the description

Rank against a fixed 40-distractor pool (0.5 = chance). 10 behaviourally-screened,
fluency-preserving concepts × 30 topics × 11 λ × 6 generations.

| λ | **concept** | random | style | pure_v |
|---|---|---|---|---|
| −2.0 | **0.351** | 0.509 | 0.500 | 0.313 |
| −1.0 | **0.415** | 0.507 | 0.500 | 0.313 |
| −0.3 | **0.479** | 0.498 | 0.500 | 0.314 |
| 0.0 | 0.500 | 0.500 | 0.500 | 0.500 |
| +0.3 | **0.558** | 0.504 | 0.500 | 0.796 |
| +1.0 | **0.723** | 0.504 | 0.500 | 0.796 |
| +2.0 | **0.779** | 0.502 | 0.500 | 0.796 |

**Four things, in order of importance:**

1. **`concept` is monotone and bidirectional**: 0.351 → 0.500 → 0.779, i.e.
   `S(+λ) > S(0) > S(−λ)`. Adding the direction makes the description more like
   the concept; subtracting it makes it *less*. Directional consistency holds.
2. **`random` is flat** (0.498–0.509 across the whole range). The effect is
   direction-specific, not a response to perturbation magnitude. The control that
   could have killed the result does not.
3. **`style` is flat** (0.500 everywhere) — and this is the stronger control. The
   style direction is a *real* direction the model represents, unlike a random one.
   SelfIE does not drift concept-ward for it. This separates "responds to any
   consistent real direction" from "responds to this concept", which the random
   arm alone cannot do.
4. **`pure_v` saturates instantly** (0.796 at λ=0.3, flat thereafter) whereas
   `concept` climbs gradually. At λ=0.3 concept is 0.558 vs pure_v 0.796, so `h`
   is still doing most of the work; by λ=2.0 they converge (0.779 vs 0.796) and
   `h` has been washed out. **The informative regime is λ ≲ 1**; beyond that the
   model is just describing `v`.

`T_cos` (similarity to the *original* topic) falls 0.908 → 0.764 on the concept
arm versus 0.802 (random) and 0.798 (style) — so concept-ness rises *while*
topic-ness falls, and falls further than controls. A trade-off, not free drift.

## 5. Sensitivity comparison — the point of the extension

λ at half-max:

| curve | λ₅₀ |
|---|---|
| **SelfIE (concept)** | **0.596** |
| **behavioural** | **0.576** |
| style | 1.75 (flat/noisy) |
| random | — (flat) |

**SelfIE reports the change at essentially the same steering strength that
changes the model's behaviour** (0.596 vs 0.576, a 3% difference). This is the
`intervention → internal change → SelfIE report` chain the extension was built to
test, and the two legs line up.

## 6. Qualitative

Topic **"Chu Suiliang"** (Tang-dynasty calligrapher); concept **"awards and
nominations related to performances in film and television"**:

| λ | description |
|---|---|
| −2.0 | "the Chinese herbal medicine used to treat fever" |
| −1.0 | "the Chinese herbalist who developed the concept of qi" |
| 0.0 | "Chen Shou, the Eastern Jin historian who compiled the Records of the Three Kingdoms" |
| +0.6 | "Chen Kaige, the Chinese film director" |
| +2.0 | "Chen Kaige, the Chinese film director" |

"Chinese historical figure" is retained from `h` throughout while the film/media
content enters with λ. The output is a **blend**, which is what activation-
grounded interpretation should look like — not a replacement.

## 7. Variance decomposition (concept arm, 19,800 generations)

| component | share |
|---|---|
| prior / constant | **71.8%** of second moment |
| responsive (variance) | 28.2% |
| ↳ topic identity | 31.4% of variance |
| ↳ λ along v | **0.75%** of variance |
| ↳ concept identity | 1.2% |
| ↳ residual (sampling noise) | 66.6% |

**The honest headline.** The λ effect is real, monotone, direction-specific and
behaviourally calibrated — but it is *small*: 0.75% of description-embedding
variance, against 31.4% for which topic `h` came from. Descriptions remain
dominated by `h`'s identity and by sampling noise at temperature 0.5.

---

## What this establishes, and what it doesn't

**Supports the paper.** Interpretations are *not* prior-anchored. They track
controlled changes to `h` in a direction-specific, bidirectional way, at the same
intervention strength that moves behaviour. The prior alone (§1) identifies
nothing at all. The three controls that could have produced a false positive —
random direction, style direction, and pure-`v` degeneracy — all behave as a
genuine effect requires.

**Confirms the paper's own account of the bias.** The authors already report the
zero-vector behaviour (Appendix J) and already conclude that the bias captures
format while the activation contributes semantics — with a stronger test than
ours (ALL-CAPS label training, Appendix I). Our contribution here is a number
rather than a reframing: 0.0% R@1 shows the bias carries *no* instance
information, which is what the paper says. Read in isolation, "~85% of the
improvement" might suggest the prior does most of the work; the paper does not
claim that and neither does this.

**Limits.**
- Not causal. Steering vectors are not clean semantic representations; this tests
  *sensitivity to a controlled, independently validated change*, nothing more.
- **Selection effect, declared**: the 10 concepts were chosen *because* they steer
  well and preserve fluency. Results describe the favourable case.
- 10 concepts × 30 topics, one model, one layer, one adapter architecture.
- The λ variance share is small; a reader who cares about effect size rather than
  existence should weight §7 over §4.
- `T_cos` also declines for random/style, so some of the topic-displacement is
  generic perturbation damage rather than concept substitution.

**Next step** (out of scope here): intervene on features or circuits whose causal
role is independently established, and test whether SelfIE reports *that*.
