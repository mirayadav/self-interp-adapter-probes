#!/usr/bin/env python3
"""
Phase 3c/3d: the main experiment. Sweep lambda bidirectionally and record what
SelfIE says.

For a base topic activation h (contrastive: raw residual minus the dataset mean,
matching the paper's pipeline) and a unit direction v_hat:

    soft = adapter( h + lambda * ||h|| * v_hat )

lambda is in units of ||h||, the same convention behavioral.py uses, so the two
sensitivity curves can be overlaid.

Note the normalize_input=true geometry (Phase 0): adding lambda*v can only ROTATE
the direction, never change magnitude, so no norm/magnitude confound can explain
any observed effect. lambda is effectively an angle; `cos_to_v` is recorded per
row because it is the more faithful x-axis.

Arms -- the controls carry the argument, not the main sweep:

  concept   h + lambda*||h||*v_hat_c        the effect under test
  random    norm-matched random direction   isolates direction-specificity from
                                            generic perturbation response
  style     the cross-concept shared direction removed in vectors.py. A random
            direction is not one the model represents; style IS. This arm tests
            whether SelfIE responds to any consistent real direction or to the
            concept specifically -- the random arm cannot answer that.
  pure_v    lambda*||h||*v_hat with NO h at all. The degeneracy reference: at
            large lambda, h+lambda*v ~= lambda*v, so SelfIE would merely be
            describing v. Wherever `concept` ~= `pure_v`, h has stopped
            mattering and the result says nothing about activation grounding.
  bias_only f(0) = b. The prior floor: what the adapter says with zero
            instance information.

Writes one row per generation so every downstream statistic can be recomputed
without re-running the model.
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from selfie_steering.core import load_lm, residual_at, format_topic_prompt, build_selfie_template, selfie_describe
from selfie_steering.adapter import Adapter, load_mean_vector

DEFAULT_GRID = [-2.0, -1.5, -1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 1.5, 2.0]


def load_topics(n: int, seed: int, split: str = "val"):
    from datasets import load_dataset
    d = load_dataset("keenanpepper/fifty-thousand-things")["train"]
    df = d.to_pandas()
    df = df[df.split == split] if (df.split == split).any() else df
    return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="results/concept_vectors.pt")
    ap.add_argument("--mode", default="positives")
    ap.add_argument("--adapter", default="adapters/wikipedia-scalar-affine.safetensors")
    ap.add_argument("--mean-vectors", default="adapters/mean-vectors.safetensors")
    ap.add_argument("--concepts", nargs="+", default=None,
                    help="uids to sweep; default = behavioral_summary top N")
    ap.add_argument("--n-concepts", type=int, default=10)
    ap.add_argument("--n-topics", type=int, default=30)
    ap.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID)
    ap.add_argument("--n-gen", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--arms", nargs="+",
                    default=["concept", "random", "style", "pure_v"])
    ap.add_argument("--out", default="results/sweep.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.vectors = "results/smoke_concept_vectors.pt"
        a.n_concepts, a.n_topics, a.n_gen, a.max_new_tokens = 2, 3, 2, 12
        a.grid = [-1.0, 0.0, 1.0]
        a.out = "results/smoke_sweep.parquet"

    blob = torch.load(a.vectors, weights_only=False)
    uids, labels, layer = blob["uids"], blob["labels"], blob["layer"]
    V = blob[a.mode]
    shared = blob[f"{a.mode}_shared"]

    # pick concepts: prefer the behaviourally screened ones
    if a.concepts:
        chosen = a.concepts
    else:
        bs = "results/behavioral_summary.parquet"
        if os.path.exists(bs):
            s = pd.read_parquet(bs)
            s = s[~s.uid.str.startswith("__") & s.fluency_ok]
            chosen = list(s.sort_values("rise", ascending=False).uid)[:a.n_concepts]
            print(f"using {len(chosen)} behaviourally-screened concepts")
        else:
            chosen = uids[:a.n_concepts]
            print("[WARN] no behavioral_summary.parquet -- concepts NOT screened; "
                  "run behavioral.py first or results are uninterpretable")

    lm = load_lm(blob["model"], layer=layer)
    if a.smoke:
        adapter = Adapter.random_like(lm.dim)
        mean_vec = torch.zeros(lm.dim)
    else:
        adapter = Adapter.load(a.adapter)
        mean_vec = load_mean_vector(a.mean_vectors, layer)
        assert adapter.dim == lm.dim, f"adapter dim {adapter.dim} != model dim {lm.dim}"
    print(adapter)

    topics = load_topics(a.n_topics, a.seed)
    prompts = [format_topic_prompt(lm, t) for t in topics.original_title]
    H_raw = residual_at(lm, prompts, pool="last", batch_size=8)
    H = H_raw - mean_vec                                    # contrastive vectors
    Hn = H.norm(dim=1, keepdim=True)
    print(f"{len(topics)} topics | ||h_contrastive|| mean {Hn.mean():.2f}")

    rng = np.random.default_rng(a.seed)
    template = build_selfie_template(lm)
    idx = {u: i for i, u in enumerate(uids)}
    rows = []

    # bias-only floor: f(0) = b, independent of every arm
    for j, d in enumerate(selfie_describe(lm, adapter(torch.zeros(1, lm.dim)),
                                          n=a.n_gen, max_new_tokens=a.max_new_tokens,
                                          temperature=a.temperature, template=template,
                                          seed=a.seed, batch_size=a.batch_size)[0]):
        rows.append({"arm": "bias_only", "uid": "__BIAS__", "label": "",
                     "topic": "", "lam": 0.0, "gen": j, "cos_to_v": np.nan,
                     "desc": d})

    for ci, uid in enumerate(chosen, 1):
        vc = F.normalize(V[idx[uid]], dim=-1)
        vr = torch.tensor(rng.standard_normal(lm.dim), dtype=torch.float32)
        vr = F.normalize(vr, dim=-1)
        vs = F.normalize(shared, dim=-1)
        dirs = {"concept": vc, "random": vr, "style": vs, "pure_v": vc}
        print(f"[{ci}/{len(chosen)}] {uid}: {labels[uid][:60]}")

        for arm in a.arms:
            vhat = dirs[arm]
            for lam in a.grid:
                pert = lam * Hn * vhat                       # (T, d)
                X = pert if arm == "pure_v" else H + pert
                soft = adapter(X)
                cos_v = F.cosine_similarity(X, vhat.expand_as(X), dim=-1)
                descs = selfie_describe(lm, soft, n=a.n_gen,
                                        max_new_tokens=a.max_new_tokens,
                                        temperature=a.temperature, template=template,
                                        seed=a.seed, batch_size=a.batch_size)
                for ti, dl in enumerate(descs):
                    for j, d in enumerate(dl):
                        rows.append({"arm": arm, "uid": uid, "label": labels[uid],
                                     "topic": topics.original_title.iloc[ti],
                                     "lam": lam, "gen": j,
                                     "cos_to_v": float(cos_v[ti]), "desc": d})
            print(f"    {arm:9s} done ({len(a.grid)} lambdas x {len(topics)} topics)")

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} generations)")
    print(df.groupby("arm").size().to_string())


if __name__ == "__main__":
    main()
