# Phase 0 — Adapter geometry (laptop, no GPU)

Run: `python selfie_steering/geometry.py`

The trained adapters use `normalize_input=true`, so the real map is
`f(h) = alpha * (h/||h||) + b`. The instance term has **fixed norm alpha**; the
learned prior `b` is an unconstrained constant. Every soft token therefore lies
on a sphere of radius `alpha` centred at `b`.

| adapter | alpha | \|\|b\|\| | alpha/\|\|b\|\| | instance frac | prior frac | max angle |
|---|---|---|---|---|---|---|
| `wikipedia-scalar-affine`   | 7.175 | 20.869 | 0.344 | **10.6%** | **89.4%** | 40.2 deg |
| `goodfire-sae-scalar-affine`| 4.032 | 19.750 | 0.204 | **4.0%**  | **96.0%** | 23.6 deg |
| `llamascope-sae-scalar-affine`| 4.021 | 19.008 | 0.212 | **4.3%** | **95.7%** | 24.4 deg |

`instance frac = alpha^2/(alpha^2+||b||^2)`; `max angle = 2*arcsin(alpha/||b||)`.

In d=4096 random directions are near-orthogonal, giving the closed form
`E[cos(f(h_i), f(h_j))] = ||b||^2/(alpha^2+||b||^2) = 1 - instance_frac`.
Predicted vs empirical (1000 random directions) agree to 5 decimals, and the
empirical off-axis spread sits just inside the analytic `arcsin` bound. The
geometry is exactly as derived.

## Findings

**1. The prior dominates the soft token by norm.** For the Wikipedia adapter
89.4% of the soft token's second moment about the origin comes from the constant
`b`. All 50k topic interpretations are launched from inside a single 40-degree
cone; for the SAE adapters that cone is 24 degrees and the prior share is ~96%.

**2. This lands remarkably close to the paper's ~85%** — from a completely
independent direction (closed-form geometry, zero generation). **But they are
not the same quantity.** The paper's 85% is a share of *benchmark improvement*;
this is a share of *input-signal norm*. The numerical agreement is suggestive,
not a reproduction, and should not be reported as one.

**3. The important consequence is the opposite of the obvious reading.**
A small-norm perturbation is not the same as an uninformative one. The paper
gets 94% recall@1 over ~50,000 topics using only this cone, so the model must be
*extremely* sensitive to small angular displacements of the soft token. Geometry
bounds the instance **budget**, not its **usage** — which is precisely why the
lambda-sweep is needed and cannot be replaced by this calculation.

**4. It also makes a prediction.** If the model resolves 1-in-50,000 topics
inside a 40-degree cone, it should register a steering vector that rotates `h`
within that same cone. A *flat* S(lambda) would then be genuinely surprising and
would point at prior-anchoring rather than at an insufficient signal.

**5. Adapter contrast.** The SAE adapters allocate the instance term less than
half the relative norm the Wikipedia adapter does (4.0-4.3% vs 10.6%), despite
the paper reporting SAE labels that beat their own training labels. Worth
carrying into Phase 3 as a secondary comparison.

## Caveat

Cosine in embedding space is a proxy for how differently the LM *reads* a soft
token, not a measurement of it — the LM is deeply nonlinear and these vectors are
far off the token-embedding manifold. Phase 2 should re-run this on real topic
vectors (which may cluster more tightly than random directions, tightening the
cone further) and compare `||b||` against Llama-3.1-8B's actual input-embedding
norms to see how far off-distribution the soft tokens sit.
