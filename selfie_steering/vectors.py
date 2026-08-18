#!/usr/bin/env python3
"""
Phase 3a: build concept steering vectors from AxBench, with the corrections
required by results/phase3a_style_confound.md.

Construction (default `positives`):

    v_c_raw = mean( resid_L over the RESPONSE span of concept c's positives )
    v_c     = v_c_raw - mean_over_all_concepts( v_c'_raw )

The cross-concept mean subtraction is essential and is the same trick the paper
uses for topic vectors (mean-vectors.safetensors subtracts the mean over ~50k
topics). Without it, v_c is dominated by a concept-independent "AxBench synthetic
positive" direction: style separates positives from negatives at 0.969 AUC on
held-out concepts, and such a vector would survive the random-direction control
while having nothing to do with the concept.

Alternative modes, kept for robustness checks:
  `pos_minus_neg`  classic diff-of-means. Lower shared component before centring
                   (0.233 vs 0.480 lexically) but differences two DISJOINT
                   instruction pools (0% overlap), so it imports a prompt-pool
                   artifact of its own.
  `instruction`    prompt-only forward pass. This is a CONTROL, not a candidate:
                   AxBench instructions are concept-matched (they predict their
                   own concept label at 35.8% top-50), so part of any effect may
                   be prompt topic rather than the concept in the response. If
                   the instruction-only vector reproduces the effect, the result
                   is about prompts, not about concepts.

Everything is written to a single .pt so downstream phases never recompute.
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from selfie_steering.core import load_lm, residual_at, format_qa

MODES = ("positives", "pos_minus_neg", "instruction")


def concept_vectors(lm, pairs: pd.DataFrame, uids, mode: str, batch_size: int,
                    max_per_concept: int = 0, half: str = "all"):
    """Raw (uncentred) per-concept vectors, plus per-concept sample counts."""
    rows, counts = [], []
    for k, uid in enumerate(uids, 1):
        g = pairs[pairs.uid == uid]
        pos = g[g.polarity == "pos"]
        neg = g[g.polarity == "neg"]
        if half in ("a", "b"):                       # split-half stability
            pos = pos.iloc[0::2] if half == "a" else pos.iloc[1::2]
            neg = neg.iloc[0::2] if half == "a" else neg.iloc[1::2]
        if max_per_concept:
            pos, neg = pos.head(max_per_concept), neg.head(max_per_concept)

        def emb(df, prompt_only=False):
            if len(df) == 0:
                return torch.zeros(lm.dim)
            if prompt_only:
                texts = [lm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": r.input}],
                    tokenize=False, add_generation_prompt=True) for r in df.itertuples()]
                return residual_at(lm, texts, pool="last", batch_size=batch_size).mean(0)
            out = [format_qa(lm, r.input, r.output) for r in df.itertuples()]
            texts, spans = [o[0] for o in out], [o[1] for o in out]
            return residual_at(lm, texts, pool="response", batch_size=batch_size,
                               response_spans=spans).mean(0)

        if mode == "positives":
            v = emb(pos)
        elif mode == "pos_minus_neg":
            v = emb(pos) - emb(neg)
        elif mode == "instruction":
            v = emb(pos, prompt_only=True)
        else:
            raise ValueError(mode)
        rows.append(v)
        counts.append({"uid": uid, "n_pos": len(pos), "n_neg": len(neg)})
        if k % 10 == 0:
            print(f"    {k}/{len(uids)} concepts")
    return torch.stack(rows), pd.DataFrame(counts)


def diagnostics(V_raw: torch.Tensor, tag: str) -> dict:
    """Activation-space version of the checks done lexically in phase 3a."""
    shared = V_raw.mean(0)
    frac = (shared.norm() / V_raw.norm(dim=1).mean()).item()
    def pcos(M):
        Mn = F.normalize(M, dim=-1)
        C = Mn @ Mn.T
        iu = torch.triu_indices(len(M), len(M), offset=1)
        return C[iu[0], iu[1]]
    raw_c, cen_c = pcos(V_raw), pcos(V_raw - shared)
    d = {"tag": tag,
         "shared_frac": frac,
         "pairwise_cos_raw_mean": raw_c.mean().item(),
         "pairwise_cos_centred_mean": cen_c.mean().item(),
         "pairwise_cos_centred_absmax": cen_c.abs().max().item(),
         "norm_mean": V_raw.norm(dim=1).mean().item(),
         "norm_std": V_raw.norm(dim=1).std().item()}
    n = len(V_raw)
    floor = -1.0 / (n - 1) if n > 1 else float("nan")
    d["centred_cos_floor"] = floor
    d["centred_cos_excess"] = cen_c.mean().item() - floor
    print(f"  [{tag}] shared component = {frac:.3f} of typical norm | "
          f"pairwise cos {raw_c.mean():+.4f} -> {cen_c.mean():+.4f} after centring")
    print(f"      (mean-centring forces mean cos to -1/(n-1) = {floor:+.4f} for n={n}; "
          f"excess {cen_c.mean().item() - floor:+.4f} is the part that means anything)")
    if frac > 0.8:
        print(f"      [WARN] shared component dominates ({frac:.2f}); centring is "
              "doing nearly all the work -- inspect before trusting the sweep")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--shortlist", default="results/concept_shortlist.json")
    ap.add_argument("--pairs", default="data/axbench/concept_pairs.parquet")
    ap.add_argument("--n-concepts", type=int, default=60)
    ap.add_argument("--max-per-concept", type=int, default=0, help="0 = all 108")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--modes", nargs="+", default=["positives"], choices=MODES)
    ap.add_argument("--split-half", action="store_true",
                    help="also build half-A/half-B vectors for a stability check")
    ap.add_argument("--out", default="results/concept_vectors.pt")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.model, a.layer, a.n_concepts, a.max_per_concept = \
            "Qwen/Qwen2.5-0.5B-Instruct", 12, 6, 4
        a.out = "results/smoke_concept_vectors.pt"

    short = pd.read_json(a.shortlist)
    uids = list(short.uid)[:a.n_concepts]
    pairs = pd.read_parquet(a.pairs)
    pairs = pairs[pairs.uid.isin(uids)]
    print(f"{len(uids)} concepts | model={a.model} layer={a.layer}")

    lm = load_lm(a.model, layer=a.layer)
    blob = {"uids": uids, "model": a.model, "layer": a.layer,
            "labels": {r.uid: r.label for r in short.itertuples() if r.uid in set(uids)},
            "diagnostics": []}

    for mode in a.modes:
        print(f"\nbuilding mode={mode} ...")
        V_raw, counts = concept_vectors(lm, pairs, uids, mode, a.batch_size,
                                        a.max_per_concept)
        blob["diagnostics"].append(diagnostics(V_raw, mode))
        blob[f"{mode}_raw"] = V_raw
        blob[f"{mode}"] = V_raw - V_raw.mean(0, keepdim=True)   # <- the vectors to use
        blob[f"{mode}_shared"] = V_raw.mean(0)                  # <- style control arm
        blob[f"{mode}_counts"] = counts

        if a.split_half:
            VA, _ = concept_vectors(lm, pairs, uids, mode, a.batch_size,
                                    a.max_per_concept, half="a")
            VB, _ = concept_vectors(lm, pairs, uids, mode, a.batch_size,
                                    a.max_per_concept, half="b")
            VA, VB = VA - VA.mean(0, keepdim=True), VB - VB.mean(0, keepdim=True)
            stab = F.cosine_similarity(VA, VB, dim=-1)
            blob[f"{mode}_splithalf_cos"] = stab
            print(f"  [{mode}] split-half cosine: median {stab.median():.3f}, "
                  f"min {stab.min():.3f}, frac>0.5 {(stab > 0.5).float().mean():.2f}")
            print("      (low values mean the vector is noise; drop those concepts)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(blob, a.out)
    print(f"\nwrote {a.out}")
    print(json.dumps(blob["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
