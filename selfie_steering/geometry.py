#!/usr/bin/env python3
"""
Phase 0: adapter geometry. Laptop-only, no GPU, no torch.

The trained SelfIE adapters use normalize_input=true, so the actual map is

    f(h) = alpha * (h / ||h||) + b            (scalar_affine, d+1 params)

The instance-specific term therefore has FIXED norm alpha, while the learned
prior b is an unconstrained constant. Every soft token lies on a sphere of
radius alpha centred at b. That geometry lets us bound, in closed form and
without generating a single token, how far apart two interpretations can be.

Reported quantities
-------------------
alpha, ||b||, alpha/||b||
    Raw prior-vs-instance scale.

instance_fraction = alpha^2 / (alpha^2 + ||b||^2)
    Fraction of the soft token's second moment about the origin contributed by
    the instance term. The direct quantitative counterpart to the paper's claim
    that the bias accounts for ~85% of the improvement.

max_angle = 2*arcsin(alpha/||b||)     [valid when alpha < ||b||]
    Tangent from the origin to the sphere: the largest angle ANY two soft
    tokens can subtend. A hard ceiling on interpretation diversity.

E[cos] ~= ||b||^2 / (alpha^2 + ||b||^2)
    In d=4096, random unit directions are near-orthogonal to each other and to
    b, so this is the expected cosine between two soft tokens. Note it equals
    exactly 1 - instance_fraction; the two views agree.

Caveat: cosine in embedding space is a proxy for "how differently the LM will
read the soft token", not a measure of it. It bounds the input signal, not the
downstream response. Phase 2 checks real topic vectors against these numbers.
"""
import argparse, glob, json, os
import numpy as np
from safetensors.numpy import load_file
from safetensors import safe_open


def load_adapter(path):
    d = load_file(path)
    with safe_open(path, framework="np") as f:
        meta = f.metadata() or {}
    if "bias" not in d or "log_scale" not in d:
        return None
    return {
        "name": os.path.basename(path).replace(".safetensors", ""),
        "alpha": float(np.exp(d["log_scale"][0])),
        "b": d["bias"].astype(np.float64),
        "normalize_input": meta.get("normalize_input", "?"),
        "type": meta.get("projection_type", "?"),
        "desc": meta.get("description", ""),
    }


def analyse(ad, n_samples, seed):
    alpha, b = ad["alpha"], ad["b"]
    d = b.shape[0]
    bnorm = float(np.linalg.norm(b))
    ratio = alpha / bnorm

    inst_frac = alpha**2 / (alpha**2 + bnorm**2)
    max_angle = 2*np.arcsin(min(1.0, ratio)) if ratio < 1 else np.pi
    pred_cos = bnorm**2 / (alpha**2 + bnorm**2)

    # Empirical: random unit directions -> soft tokens -> pairwise cosine.
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n_samples, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    S = alpha*U + b
    Sn = S/np.linalg.norm(S, axis=1, keepdims=True)
    C = Sn @ Sn.T
    iu = np.triu_indices(n_samples, k=1)
    cos = C[iu]

    # Angle of each soft token off the bias direction.
    bhat = b/bnorm
    off = np.degrees(np.arccos(np.clip(Sn @ bhat, -1, 1)))

    return {
        "adapter": ad["name"], "type": ad["type"],
        "normalize_input": ad["normalize_input"], "dim": d,
        "alpha": alpha, "bias_norm": bnorm, "alpha_over_bnorm": ratio,
        "instance_fraction": inst_frac,
        "prior_fraction": 1-inst_frac,
        "max_angle_deg": float(np.degrees(max_angle)),
        "predicted_mean_cos": pred_cos,
        "empirical_mean_cos": float(cos.mean()),
        "empirical_min_cos": float(cos.min()),
        "empirical_max_cos": float(cos.max()),
        "empirical_mean_angle_deg": float(np.degrees(np.arccos(np.clip(cos.mean(),-1,1)))),
        "max_offaxis_angle_deg": float(off.max()),
        "theory_offaxis_bound_deg": float(np.degrees(np.arcsin(min(1.0, ratio)))),
        "n_samples": n_samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="adapters/*.safetensors")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/phase0_geometry.json")
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(a.adapters)):
        ad = load_adapter(p)
        if ad is None:
            print(f"skip {os.path.basename(p)} (not a scalar_affine adapter)")
            continue
        rows.append(analyse(ad, a.n_samples, a.seed))

    for r in rows:
        print("="*72)
        print(f"{r['adapter']}   [{r['type']}, normalize_input={r['normalize_input']}, d={r['dim']}]")
        print(f"  alpha (instance term norm)        {r['alpha']:.4f}")
        print(f"  ||b||  (prior term norm)          {r['bias_norm']:.4f}")
        print(f"  alpha/||b||                       {r['alpha_over_bnorm']:.4f}")
        print(f"  instance fraction of 2nd moment   {r['instance_fraction']*100:.2f}%")
        print(f"  prior    fraction of 2nd moment   {r['prior_fraction']*100:.2f}%")
        print(f"  MAX angle between any two soft tokens   {r['max_angle_deg']:.2f} deg")
        print(f"  predicted E[cos] (theory)         {r['predicted_mean_cos']:.5f}")
        print(f"  empirical  E[cos] ({r['n_samples']} random dirs)  {r['empirical_mean_cos']:.5f}"
              f"   (min {r['empirical_min_cos']:.5f})")
        print(f"  empirical mean angle              {r['empirical_mean_angle_deg']:.2f} deg")
        print(f"  max off-bias-axis angle           {r['max_offaxis_angle_deg']:.2f} deg"
              f"   (theory bound {r['theory_offaxis_bound_deg']:.2f})")
        ok = r['max_offaxis_angle_deg'] <= r['theory_offaxis_bound_deg'] + 1e-6
        print(f"  [check] empirical spread within analytic bound: {'PASS' if ok else 'FAIL'}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print("="*72)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
