#!/usr/bin/env python3
"""
Phase 3e: turn the sweep's generations into the numbers that answer the question.

Metrics
-------
S(lambda)  concept-ness of the descriptions. Primary form is `rank_pct` against a
           fixed distractor pool (see scoring.py), because raw cosine drifts
           upward whenever descriptions merely get longer or more generic.
T(lambda)  similarity to the ORIGINAL topic. S rising while T falls is far more
           informative than S alone: it shows a trade-off rather than a
           free-floating increase.
lambda_50  the lambda at which a curve reaches half its range. Comparing
           lambda_50 for SelfIE against lambda_50 for behaviour is the
           "sensitivity comparison" the extension is built around.

Variance decomposition (the headline number)
--------------------------------------------
Description embeddings are unit-norm, so E||e||^2 = 1 and the second moment about
the origin splits exactly as

    1 = ||e_bar||^2            <- CONSTANT / prior component
      + (variance about e_bar) <- everything that responds to any input

The variance term is then split factorially (the design is balanced: every
concept x topic x lambda x generation cell is filled) into topic, lambda,
concept, their interactions, and generation-level residual noise.

`prior_share` is the direct output-space counterpart to the Phase 0 input-space
number (instance term = 10.6% of the soft token's second moment) and to the
paper's claim that the bias accounts for ~85% of the improvement. All three are
DIFFERENT quantities and must not be reported as reproductions of one another.

Run `python -m selfie_steering.analysis --selftest` to verify the decomposition
against synthetic data with known ground-truth shares.
"""
from __future__ import annotations

import argparse, json, os
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ------------------------------------------------------- variance decomposition

def decompose(E: np.ndarray, factors: dict[str, np.ndarray]) -> dict:
    """
    E: (N, d) embeddings (assumed unit-norm). factors: name -> (N,) label array.

    Returns the constant/prior share plus each factor's share of the remaining
    variance. Factor shares are computed as main effects about the grand mean; on
    a balanced design they are close to orthogonal, and `residual` absorbs
    interactions and generation noise.
    """
    N = len(E)
    ebar = E.mean(0)
    total_second_moment = float((E ** 2).sum(1).mean())     # 1.0 for unit-norm
    prior = float(ebar @ ebar)
    var_total = float(((E - ebar) ** 2).sum(1).mean())

    out = {"n": N,
           "second_moment": total_second_moment,
           "prior_share": prior / total_second_moment,
           "variance_share": var_total / total_second_moment,
           "factors": {}}

    explained = 0.0
    for name, lab in factors.items():
        ss = 0.0
        for v in np.unique(lab):
            m = lab == v
            if m.sum() == 0:
                continue
            diff = E[m].mean(0) - ebar
            ss += m.sum() * float(diff @ diff)
        ss /= N
        out["factors"][name] = ss / var_total if var_total > 0 else np.nan
        explained += ss
    out["factors"]["residual"] = (var_total - explained) / var_total if var_total > 0 else np.nan
    return out


def _selftest():
    """Synthetic data with ANALYTICALLY derived ground truth.

    Each component is built with an explicit norm so the expected shares can be
    predicted in closed form rather than guessed:

        e = base + t_i + c_j + lam * amp * u + noise      (then L2-normalised)

    E||e||^2 = |base|^2 + |t|^2 + |c|^2 + E[lam^2]*amp^2 + |noise|^2
    prior_share  ~= |base|^2 / E||e||^2
    factor share ~= that factor's squared norm / total variance
    """
    rng = np.random.default_rng(0)
    d, nC, nT, nG = 64, 5, 8, 4
    lams = np.linspace(-1, 1, 7)

    def unit(*shape):
        v = rng.standard_normal(shape)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    n_base, n_t, n_c, amp, n_noise = 3.0, 0.6, 0.2, 0.4, 0.1
    base = unit(d) * n_base
    tvec = unit(nT, d) * n_t
    cvec = unit(nC, d) * n_c
    uvec = unit(d)

    rows, fc, ft, fl = [], [], [], []
    for ci in range(nC):
        for ti in range(nT):
            for li, lam in enumerate(lams):
                for _ in range(nG):
                    e = (base + tvec[ti] + cvec[ci] + lam * amp * uvec
                         + unit(d) * n_noise)
                    rows.append(e / np.linalg.norm(e))
                    fc.append(ci); ft.append(ti); fl.append(li)
    E = np.array(rows)
    r = decompose(E, {"concept": np.array(fc), "topic": np.array(ft),
                      "lam": np.array(fl)})

    lam2 = float((lams ** 2).mean())
    var_pred = n_t**2 + n_c**2 + lam2 * amp**2 + n_noise**2
    tot_pred = n_base**2 + var_pred
    exp = {"prior_share": n_base**2 / tot_pred,
           "topic": n_t**2 / var_pred,
           "concept": n_c**2 / var_pred,
           "lam": lam2 * amp**2 / var_pred}

    print("SELFTEST -- synthetic data, closed-form ground truth")
    print(f"  {'quantity':16s} {'predicted':>10s} {'measured':>10s}")
    print(f"  {'prior_share':16s} {exp['prior_share']:10.4f} {r['prior_share']:10.4f}")
    for k in ("topic", "concept", "lam"):
        print(f"  {k:16s} {exp[k]:10.4f} {r['factors'][k]:10.4f}")
    print(f"  {'residual':16s} {'~0':>10s} {r['factors']['residual']:10.4f}")

    assert abs(r["prior_share"] - exp["prior_share"]) < 0.05, "prior share off"
    for k in ("topic", "concept", "lam"):
        assert abs(r["factors"][k] - exp[k]) < 0.08, f"{k} share off"
    assert r["factors"]["topic"] > r["factors"]["lam"] > r["factors"]["concept"], \
        "factor ordering wrong"

    # NULL: identical construction but with NO lambda effect
    rows2, fl2 = [], []
    for i in range(len(E)):
        e = base + unit(d) * n_noise
        rows2.append(e / np.linalg.norm(e)); fl2.append(fl[i])
    r2 = decompose(np.array(rows2), {"lam": np.array(fl2)})
    print(f"\n  NULL case (no real lambda effect): lam share = "
          f"{r2['factors']['lam']:.4f}  (must be small)")
    assert r2["factors"]["lam"] < 0.15, "false positive on null data"

    # lambda_50 on a known curve
    l50 = lambda_50([0, .25, .5, .75, 1.0], [0.0, 0.1, 0.5, 0.9, 1.0])
    assert abs(l50 - 0.5) < 1e-6, f"lambda_50 wrong: {l50}"
    print(f"  lambda_50 on a symmetric ramp     = {l50:.3f}  (expected 0.500)")

    print("\nSELFTEST PASSED")


# ------------------------------------------------------------------ curve stats

def lambda_50(lams: Sequence[float], vals: Sequence[float]) -> float:
    """lambda at which the curve first reaches half its range (positive arm)."""
    l = np.asarray(lams, float); v = np.asarray(vals, float)
    o = np.argsort(l); l, v = l[o], v[o]
    pos = l >= 0
    l, v = l[pos], v[pos]
    if len(v) < 2 or not np.isfinite(v).all():
        return np.nan
    lo, hi = v[0], v.max()
    if hi <= lo:
        return np.nan
    half = lo + 0.5 * (hi - lo)
    for i in range(1, len(v)):
        if v[i] >= half:
            if v[i] == v[i - 1]:
                return float(l[i])
            f = (half - v[i - 1]) / (v[i] - v[i - 1])
            return float(l[i - 1] + f * (l[i] - l[i - 1]))
    return np.nan


def bootstrap_ci(values_by_group: dict, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05):
    """Resample GROUPS (concepts), not individual generations -- generations
    within a cell are strongly correlated and would give false precision."""
    rng = np.random.default_rng(seed)
    keys = list(values_by_group)
    if not keys:
        return np.nan, np.nan, np.nan
    arr = np.array([np.nanmean(values_by_group[k]) for k in keys], float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    boots = np.array([rng.choice(arr, len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    return float(arr.mean()), float(np.quantile(boots, alpha / 2)), \
           float(np.quantile(boots, 1 - alpha / 2))


# ----------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sweep", default="results/sweep.parquet")
    ap.add_argument("--behavioral", default="results/behavioral.parquet")
    ap.add_argument("--embedder", default="thenlper/gte-large")
    ap.add_argument("--n-distractors", type=int, default=40)
    ap.add_argument("--out", default="results/analysis.json")
    ap.add_argument("--curves-out", default="results/curves.parquet")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return

    from selfie_steering.scoring import Embedder, ConceptScorer

    df = pd.read_parquet(a.sweep)
    emb = Embedder(a.embedder)
    print(f"{len(df)} generations | arms {sorted(df.arm.unique())}")

    E = emb.encode(df.desc.fillna("").tolist())
    labels = sorted(df[df.uid != "__BIAS__"].label.unique())
    L = emb.encode(labels)
    lab_i = {l: i for i, l in enumerate(labels)}

    # S(lambda): rank of the true concept against a fixed distractor pool
    rng = np.random.default_rng(0)
    S_cos = np.full(len(df), np.nan)
    S_rank = np.full(len(df), np.nan)
    for lab in labels:
        m = (df.label == lab).values
        others = [i for l, i in lab_i.items() if l != lab]
        pool = rng.permutation(others)[:a.n_distractors]
        ct = E[m] @ L[lab_i[lab]]
        cd = E[m] @ L[pool].T
        S_cos[m] = ct
        S_rank[m] = (ct[:, None] > cd).mean(1)
    df["S_cos"], df["S_rank"] = S_cos, S_rank

    # T(lambda): similarity to the ORIGINAL topic
    topics = sorted(t for t in df.topic.unique() if t)
    if topics:
        TT = emb.encode(topics); ti = {t: i for i, t in enumerate(topics)}
        df["T_cos"] = [float(E[i] @ TT[ti[t]]) if t else np.nan
                       for i, t in enumerate(df.topic)]

    # curves with bootstrap CIs over concepts
    curves = []
    for arm in df.arm.unique():
        if arm == "bias_only":
            continue
        d = df[df.arm == arm]
        for lam in sorted(d.lam.unique()):
            dl = d[d.lam == lam]
            for metric in ["S_rank", "S_cos", "T_cos"]:
                if metric not in dl:
                    continue
                grp = {u: g[metric].values for u, g in dl.groupby("uid")}
                m, lo, hi = bootstrap_ci(grp)
                curves.append({"arm": arm, "lam": lam, "metric": metric,
                               "mean": m, "lo": lo, "hi": hi, "n_concepts": len(grp)})
    cur = pd.DataFrame(curves)
    cur.to_parquet(a.curves_out, index=False)

    # variance decomposition on the concept arm
    ca = df[df.arm == "concept"]
    dec = decompose(E[ca.index.values], {
        "concept": ca.uid.values, "topic": ca.topic.values, "lam": ca.lam.values})

    # sensitivity comparison
    sens = {}
    for arm in df.arm.unique():
        if arm == "bias_only":
            continue
        c = cur[(cur.arm == arm) & (cur.metric == "S_rank")].sort_values("lam")
        sens[f"selfie_lambda50_{arm}"] = lambda_50(c.lam, c["mean"])
    if os.path.exists(a.behavioral):
        b = pd.read_parquet(a.behavioral)
        b = b[~b.uid.str.startswith("__")].groupby("lam").rank_pct.mean().reset_index()
        sens["behavioral_lambda50"] = lambda_50(b.lam, b.rank_pct)

    bias_desc = df[df.arm == "bias_only"].desc.tolist()
    res = {"n_generations": int(len(df)),
           "variance_decomposition": dec,
           "sensitivity": sens,
           "bias_only_descriptions": bias_desc[:10]}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, default=float)

    print("\n=== VARIANCE DECOMPOSITION (concept arm) ===")
    print(f"  prior/constant share of 2nd moment : {dec['prior_share']:.4f}")
    print(f"  responsive (variance) share        : {dec['variance_share']:.4f}")
    for k, v in dec["factors"].items():
        print(f"    {k:10s} share of variance        : {v:.4f}")
    print("\n=== S(lambda), rank_pct, by arm ===")
    piv = cur[cur.metric == "S_rank"].pivot_table(index="lam", columns="arm", values="mean")
    print(piv.round(4).to_string())
    print("\n=== SENSITIVITY (lambda at half-max) ===")
    for k, v in sens.items():
        print(f"  {k:34s} {v}")
    print(f"\nwrote {a.out}, {a.curves_out}")
    print("\nREAD THIS: `concept` must beat `random` AND `style`; wherever "
          "`concept` ~= `pure_v`, h no longer matters and those lambdas say "
          "nothing about activation grounding.")


if __name__ == "__main__":
    main()
