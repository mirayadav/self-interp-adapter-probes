#!/usr/bin/env python3
"""
Phase 2: reproduce the paper's Wikipedia contrastive-vector retrieval result.

This is the foundation the extension stands on -- if the released adapter does
not reproduce here, nothing downstream is interpretable. Scope is deliberately
narrow: only this result, not SAE scoring / bridge-entity / taboo / scaling.

Three arms on the same topics:
  trained     the released wikipedia-scalar-affine adapter        expect ~80%+ R@1
  untrained   f(h) = 1.0 * h/||h||  (scale_only, init 1.0)        expect ~0-1%
  bias_only   f(0) = b                                            the prior floor

The bias_only arm is a DELIVERABLE, not a check: its descriptions are the literal
text of the learned prior, and every later result is read against them.

Retrieval follows the paper: embed each generated description with GTE-large and
look it up against an index of topic documents; report recall@k.
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import pandas as pd
import torch

from selfie_steering.core import load_lm, residual_at, format_topic_prompt, build_selfie_template, selfie_describe
from selfie_steering.adapter import Adapter, load_mean_vector


def topic_document(title: str, labels) -> str:
    labs = list(labels)[:8] if labels is not None else []
    return f"{title}. " + " ".join(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--adapter", default="adapters/wikipedia-scalar-affine.safetensors")
    ap.add_argument("--mean-vectors", default="adapters/mean-vectors.safetensors")
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-gen", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--embedder", default="thenlper/gte-large")
    ap.add_argument("--arms", nargs="+", default=["trained", "untrained", "bias_only"])
    ap.add_argument("--out", default="results/repro.json")
    ap.add_argument("--gen-out", default="results/repro_generations.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.model, a.layer = "Qwen/Qwen2.5-0.5B-Instruct", 12
        a.n_eval, a.n_gen, a.max_new_tokens = 8, 2, 12
        a.embedder = "sentence-transformers/all-MiniLM-L6-v2"
        a.out, a.gen_out = "results/smoke_repro.json", "results/smoke_repro_gen.parquet"

    from datasets import load_dataset
    from selfie_steering.scoring import Embedder

    ds = load_dataset("keenanpepper/fifty-thousand-things")["train"].to_pandas()
    val = ds[ds.split == "val"] if (ds.split == "val").any() else ds
    ev = val.sample(min(a.n_eval, len(val)), random_state=a.seed).reset_index(drop=True)
    index_df = ds if not a.smoke else ds.head(500)
    print(f"index over {len(index_df)} topics | evaluating {len(ev)}")

    lm = load_lm(a.model, layer=a.layer)
    if a.smoke:
        adapter = Adapter.random_like(lm.dim)
        mean_vec = torch.zeros(lm.dim)
    else:
        adapter = Adapter.load(a.adapter)
        mean_vec = load_mean_vector(a.mean_vectors, a.layer)
        assert adapter.dim == lm.dim
    print(adapter)

    emb = Embedder(a.embedder)
    docs = [topic_document(r.original_title, r.labels) for r in index_df.itertuples()]
    print("building retrieval index ...")
    D = emb.encode(docs)
    pos = {t: i for i, t in enumerate(index_df.original_title)}

    prompts = [format_topic_prompt(lm, t) for t in ev.original_title]
    H = residual_at(lm, prompts, pool="last", batch_size=8) - mean_vec
    template = build_selfie_template(lm)

    def soft_for(arm):
        if arm == "trained":
            return adapter(H)
        if arm == "untrained":
            return torch.nn.functional.normalize(H, dim=-1) * 1.0
        if arm == "bias_only":
            return adapter(torch.zeros_like(H))
        raise ValueError(arm)

    results, gen_rows = {}, []
    ks = [1, 5, 10, 20, 50, 100]
    for arm in a.arms:
        print(f"\narm={arm}")
        descs = selfie_describe(lm, soft_for(arm), n=a.n_gen,
                                max_new_tokens=a.max_new_tokens,
                                temperature=a.temperature, template=template,
                                seed=a.seed, batch_size=a.batch_size)
        flat, owner = [], []
        for i, dl in enumerate(descs):
            for j, d in enumerate(dl):
                flat.append(d); owner.append(i)
                gen_rows.append({"arm": arm, "topic": ev.original_title.iloc[i],
                                 "gen": j, "desc": d})
        Eg = emb.encode(flat)
        sims = Eg @ D.T
        order = np.argsort(-sims, axis=1)
        # best-of-n: a topic counts as hit@k if ANY of its generations hits
        hits = {k: np.zeros(len(ev), bool) for k in ks}
        for r, oi in enumerate(owner):
            tgt = pos.get(ev.original_title.iloc[oi])
            if tgt is None:
                continue
            rank = int(np.where(order[r] == tgt)[0][0])
            for k in ks:
                if rank < k:
                    hits[k][oi] = True
        results[arm] = {f"recall@{k}": float(hits[k].mean()) for k in ks}
        print("  " + "  ".join(f"R@{k}={results[arm][f'recall@{k}']*100:.1f}%" for k in ks))
        if arm == "bias_only":
            print("  --- the learned prior, verbatim ---")
            for d in flat[:8]:
                print(f"    {d[:90]!r}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(gen_rows).to_parquet(a.gen_out, index=False)
    with open(a.out, "w") as f:
        json.dump({"n_eval": int(len(ev)), "n_index": int(len(index_df)),
                   "model": a.model, "layer": a.layer, "adapter": str(adapter),
                   "n_gen": a.n_gen, "results": results}, f, indent=2)
    print(f"\nwrote {a.out}, {a.gen_out}")
    print("\nGATE: trained R@1 within a few points of the paper (~83% without "
          "scale tuning), untrained ~1%. If not, suspect the layer index "
          "(hidden_states[layer+1]) before anything else.")


if __name__ == "__main__":
    main()
