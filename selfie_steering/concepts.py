#!/usr/bin/env python3
"""
Build a unified AxBench concept catalog + contrastive text pairs.

AxBench CONCEPT500 (pyvene/axbench-concept500) ships four near-disjoint
partitions (2b/l10, 2b/l20, 9b/l20, 9b/l31), 500 concepts each, ~1995 distinct
concepts in total. Concepts are GemmaScope SAE autointerp labels, i.e. the same
descriptive register the SelfIE adapters were trained on.

Per concept:
  train split : 72 positive responses (+216 shared concept-independent negatives)
  test  split : 36 positive + 36 negative, negatives are CONCEPT-SPECIFIC
                (same concept_id/sae_id, response simply lacks the concept)

The concept-specific negatives are the better contrast, so the steering vector is
    v_c = mean(resid_19 | positives of c) - mean(resid_19 | negatives of c)
with positives pooled from train+test (108) and negatives from test (36).

Nothing here touches a GPU or a model: it only assembles text.
"""
import argparse, json, os
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "pyvene/axbench-concept500"
PARTITIONS = ["2b/l10", "2b/l20", "9b/l20", "9b/l31"]


def load_partition(sub, split):
    p = hf_hub_download(REPO, f"{sub}/{split}/data.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    df["partition"] = sub
    return df


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cat_rows, pair_rows = [], []

    for sub in PARTITIONS:
        tr = load_partition(sub, "train")
        te = load_partition(sub, "test")

        # concept-specific positives / negatives from test
        te_pn = te[te.category.isin(["positive", "negative"])]
        # train positives (train negatives are shared/concept-independent -> skip)
        tr_pos = tr[tr.category == "positive"]

        meta = (te_pn[te_pn.category == "positive"]
                .drop_duplicates("concept_id")
                .set_index("concept_id")[["output_concept", "concept_genre", "sae_id", "sae_link"]])

        for cid, m in meta.iterrows():
            uid = f"{sub}:{cid}"
            pos = pd.concat([
                tr_pos[tr_pos.concept_id == cid][["input", "output"]],
                te_pn[(te_pn.concept_id == cid) & (te_pn.category == "positive")][["input", "output"]],
            ], ignore_index=True)
            neg = te_pn[(te_pn.concept_id == cid) & (te_pn.category == "negative")][["input", "output"]]

            cat_rows.append({
                "uid": uid, "partition": sub, "concept_id": int(cid),
                "label": m.output_concept, "genre": m.concept_genre,
                "sae_id": int(m.sae_id), "sae_link": m.sae_link,
                "n_pos": len(pos), "n_neg": len(neg),
            })
            for _, r in pos.iterrows():
                pair_rows.append({"uid": uid, "polarity": "pos", "input": r.input, "output": r.output})
            for _, r in neg.iterrows():
                pair_rows.append({"uid": uid, "polarity": "neg", "input": r.input, "output": r.output})

    cat = pd.DataFrame(cat_rows)
    pairs = pd.DataFrame(pair_rows)
    cat.to_parquet(f"{out_dir}/concepts.parquet", index=False)
    pairs.to_parquet(f"{out_dir}/concept_pairs.parquet", index=False)
    cat[["uid", "label", "genre", "n_pos", "n_neg"]].to_json(
        f"{out_dir}/concepts.json", orient="records", indent=2)
    return cat, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/axbench")
    ap.add_argument("--show", type=int, default=25)
    a = ap.parse_args()

    cat, pairs = build(a.out_dir)
    print(f"concepts: {len(cat)}   distinct labels: {cat.label.nunique()}")
    print(f"pairs rows: {len(pairs)}  (pos {(pairs.polarity=='pos').sum()}, neg {(pairs.polarity=='neg').sum()})")
    print("genre:", cat.genre.value_counts().to_dict())
    print("n_pos per concept:", cat.n_pos.value_counts().to_dict())
    print("n_neg per concept:", cat.n_neg.value_counts().to_dict())
    print(f"\n--- {a.show} sample 'text' concept labels ---")
    for i, r in cat[cat.genre == "text"].sample(a.show, random_state=0).iterrows():
        print(f"  [{r.uid:12s}] {r.label}")
    print(f"\nwrote {a.out_dir}/concepts.parquet, concept_pairs.parquet, concepts.json")


if __name__ == "__main__":
    main()
