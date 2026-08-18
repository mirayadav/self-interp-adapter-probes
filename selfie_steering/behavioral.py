#!/usr/bin/env python3
"""
Phase 3b: behavioural validation of the steering vectors == concept screening.

These are the same operation. Screening AxBench concepts for the ones that
actually steer the target model IS the B(lambda) measurement the extension needs,
so it is run once and reused.

Without this leg the whole project is only semantic correlation: the claim
"intervention -> internal change -> SelfIE reports it" requires independent
evidence that the intervention changes behaviour at all.

lambda convention (shared with selfie_sweep.py so the two curves are comparable):

    steered residual = resid + lambda * ||resid||_typ * v_hat

i.e. lambda is measured in units of the typical residual norm at the layer, not
in raw coefficient units. lambda=0.5 means "add half a typical residual".

Outputs one row per (concept, lambda) with:
  rank_pct   primary concept-ness (see scoring.py: robust to generic drift)
  cosine     raw similarity, reported for comparison only
  keyword    non-embedding second opinion
  distinct2 / nll_per_token   fluency guards -- the analysis range is the lambda
                              band where these stay near their lambda=0 values
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from selfie_steering.core import load_lm, residual_at, generate_steered, format_topic_prompt
from selfie_steering.scoring import Embedder, ConceptScorer, keyword_rate, fluency

NEUTRAL_PROMPTS = [
    "Write a short paragraph about what happened yesterday.",
    "Describe something you find interesting.",
    "Explain how two people might solve a disagreement.",
    "Tell me a short story.",
    "What makes a good afternoon?",
    "Describe a place worth visiting.",
    "Give me some general advice.",
    "Write a few sentences about change.",
    "What should someone consider before making a decision?",
    "Describe an ordinary object in detail.",
    "Write a short note to a friend.",
    "Explain something simple to a beginner.",
    "What is worth paying attention to?",
    "Describe a typical morning.",
    "Write a brief reflection on learning.",
    "What do people often overlook?",
    "Describe a conversation between two strangers.",
    "Write a short paragraph about work.",
    "What makes something memorable?",
    "Describe how a project usually begins.",
]

DEFAULT_GRID = [-2.0, -1.5, -1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 1.5, 2.0]


def typical_norm(lm, n: int = 32, batch_size: int = 8) -> float:
    """Mean residual norm at lm.layer over the neutral prompts -- the unit for lambda."""
    prompts = [lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in NEUTRAL_PROMPTS[:n]]
    return residual_at(lm, prompts, pool="last", batch_size=batch_size).norm(dim=1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="results/concept_vectors.pt")
    ap.add_argument("--mode", default="positives")
    ap.add_argument("--screen", default="results/concept_screen.parquet")
    ap.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-distractors", type=int, default=40)
    ap.add_argument("--embedder", default="thenlper/gte-large")
    ap.add_argument("--out", default="results/behavioral.parquet")
    ap.add_argument("--gen-out", default="results/behavioral_generations.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.vectors = "results/smoke_concept_vectors.pt"
        a.grid = [-1.0, 0.0, 1.0]
        a.n_prompts, a.max_new_tokens = 3, 16
        a.embedder = "sentence-transformers/all-MiniLM-L6-v2"
        a.out, a.gen_out = "results/smoke_behavioral.parquet", "results/smoke_behavioral_gen.parquet"

    blob = torch.load(a.vectors, weights_only=False)
    uids, labels = blob["uids"], blob["labels"]
    V = blob[a.mode]                                   # centred vectors
    shared = blob[f"{a.mode}_shared"]                  # style-direction control

    lm = load_lm(blob["model"], layer=blob["layer"])
    emb = Embedder(a.embedder)
    unit = typical_norm(lm, batch_size=a.batch_size)
    print(f"typical residual norm at layer {blob['layer']}: {unit:.2f}  (lambda unit)")

    kw_map = {}
    if os.path.exists(a.screen):
        sc = pd.read_parquet(a.screen).set_index("uid")
        kw_map = {u: str(sc.loc[u, "top_tokens"]).split(", ") for u in uids if u in sc.index}

    prompts = [lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in NEUTRAL_PROMPTS[:a.n_prompts]]

    all_labels = [labels[u] for u in uids]
    rows, gens = [], []

    # arms: every concept, plus the shared/style direction as a control
    arms = [(u, F.normalize(V[i], dim=-1)) for i, u in enumerate(uids)]
    arms.append(("__STYLE__", F.normalize(shared, dim=-1)))
    rng = np.random.default_rng(a.seed)
    arms.append(("__RANDOM__", F.normalize(torch.tensor(
        rng.standard_normal(lm.dim), dtype=torch.float32), dim=-1)))

    for ai, (uid, vhat) in enumerate(arms, 1):
        label = labels.get(uid, "generic descriptive text")
        pool = [l for l in all_labels if l != label]
        rng.shuffle(pool)
        scorer = ConceptScorer(emb, label, pool[:a.n_distractors])
        print(f"[{ai}/{len(arms)}] {uid}: {label[:60]}")
        for lam in a.grid:
            texts = generate_steered(lm, prompts, vhat, lam * unit,
                                     max_new_tokens=a.max_new_tokens,
                                     temperature=a.temperature,
                                     batch_size=a.batch_size, seed=a.seed)
            s = scorer.score(texts)
            f = fluency(texts, lm=lm)
            rows.append({"uid": uid, "label": label, "lam": lam,
                         "rank_pct": s["rank_pct"], "cosine": s["cosine"],
                         "margin": s["margin"],
                         "keyword": keyword_rate(texts, kw_map.get(uid, [])),
                         **f})
            for t in texts:
                gens.append({"uid": uid, "lam": lam, "text": t})
            print(f"    lam={lam:+.2f} rank={s['rank_pct']:.3f} cos={s['cosine']:.3f} "
                  f"d2={f['distinct2']:.2f} nll={f.get('nll_per_token', float('nan')):.2f}")

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_parquet(a.out, index=False)
    pd.DataFrame(gens).to_parquet(a.gen_out, index=False)

    # screening summary: monotone rise on the positive arm, fluency preserved
    base = df[df.lam == 0].set_index("uid")
    summ = []
    for uid in df.uid.unique():
        g = df[df.uid == uid].sort_values("lam")
        pos = g[g.lam >= 0]
        rise = pos.rank_pct.iloc[-1] - pos.rank_pct.iloc[0]
        mono = float(np.corrcoef(pos.lam, pos.rank_pct)[0, 1]) if len(pos) > 2 else np.nan
        d2_ok = bool((g.distinct2 > 0.6 * base.loc[uid, "distinct2"]).all())
        summ.append({"uid": uid, "label": g.label.iloc[0], "rise": rise,
                     "monotonicity": mono, "fluency_ok": d2_ok})
    s = pd.DataFrame(summ).sort_values("rise", ascending=False)
    s.to_parquet("results/behavioral_summary.parquet", index=False)
    print("\n--- screening summary (top 15 by rise in rank_pct) ---")
    print(s.head(15).to_string(index=False))
    print(f"\nCONTROLS: __STYLE__ and __RANDOM__ should sit near the BOTTOM.")
    print(f"wrote {a.out}, {a.gen_out}, results/behavioral_summary.parquet")


if __name__ == "__main__":
    main()
