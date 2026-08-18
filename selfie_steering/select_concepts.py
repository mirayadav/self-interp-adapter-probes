#!/usr/bin/env python3
"""
CPU-only pre-screen of the ~2000 AxBench concepts down to a GPU-screen shortlist.

The *principled* selection is the behavioural screen in Phase 3b (does adding
lambda*v actually steer Llama-3.1-8B?), which needs a GPU. This script is a
cheap pre-filter so that screen only has to look at good candidates.

Three signals, none of which touch a model:

1. separability  - 5-fold CV ROC-AUC of a TF-IDF + logistic-regression classifier
   distinguishing the concept's 108 positive responses from its 36 concept-specific
   negatives. Low AUC means the concept has no consistent textual signature, so
   diff-of-means would be noise and the embedding score could not detect it either.

2. is_surface    - keyword flag on the LABEL for concepts about surface form
   (punctuation, capitalisation, tokens, grammar) rather than semantic content.
   These are poor SelfIE targets: the adapter was trained to emit topical
   descriptions, so a null result on them would confound "insensitive to h" with
   "has no vocabulary for this".

3. top_tokens    - highest positive-weight TF-IDF features. Content nouns indicate
   a topical concept; function words/punctuation indicate a syntactic one. Kept for
   eyeballing, since the is_surface heuristic reads the label rather than the data.

Caveat, to be stated in the writeup: high separability can mean "clear semantic
signature" OR "trivially lexical". It is a screening signal, not a quality score.
"""
import argparse, json, re, os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

SURFACE_PAT = re.compile(
    r"\b(punctuation|capitaliz|capitalis|whitespace|token|suffix|prefix|"
    r"grammatical|syntax|syntactic|spelling|typograph|character|letter|"
    r"apostrophe|comma|semicolon|bracket|parenthes|quotation|newline|"
    r"formatting|markdown|indentation|word ['\"]|the word)\b", re.I)


def score_concept(pos_docs, neg_docs, seed=0):
    docs = list(pos_docs) + list(neg_docs)
    y = np.r_[np.ones(len(pos_docs), int), np.zeros(len(neg_docs), int)]
    vec = TfidfVectorizer(min_df=2, max_features=5000, stop_words="english",
                          sublinear_tf=True)
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        return np.nan, []
    if X.shape[1] < 5:
        return np.nan, []
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    try:
        auc = float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())
    except ValueError:
        return np.nan, []
    clf.fit(X, y)
    names = np.array(vec.get_feature_names_out())
    top = names[np.argsort(clf.coef_[0])[::-1][:10]].tolist()
    return auc, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/axbench")
    ap.add_argument("--out", default="results/concept_screen.parquet")
    ap.add_argument("--shortlist", default="results/concept_shortlist.json")
    ap.add_argument("--min-auc", type=float, default=0.90)
    ap.add_argument("--n-shortlist", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N concepts")
    a = ap.parse_args()

    cat = pd.read_parquet(f"{a.data_dir}/concepts.parquet")
    pairs = pd.read_parquet(f"{a.data_dir}/concept_pairs.parquet")
    if a.limit:
        cat = cat.head(a.limit)
    by_uid = {u: g for u, g in pairs.groupby("uid")}

    rows = []
    for i, r in enumerate(cat.itertuples(), 1):
        g = by_uid[r.uid]
        auc, top = score_concept(g[g.polarity == "pos"].output,
                                 g[g.polarity == "neg"].output)
        rows.append({"uid": r.uid, "label": r.label, "genre": r.genre,
                     "separability_auc": auc,
                     "is_surface": bool(SURFACE_PAT.search(r.label)),
                     "label_len": len(r.label.split()),
                     "top_tokens": ", ".join(top)})
        if i % 250 == 0:
            print(f"  scored {i}/{len(cat)}")

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_parquet(a.out, index=False)

    ok = df[(df.genre == "text") & (~df.is_surface) &
            (df.separability_auc >= a.min_auc)].copy()
    # drop near-duplicate labels (keep the most separable of each token-set)
    ok["key"] = ok.label.str.lower().str.replace(r"[^a-z ]", "", regex=True)
    ok = ok.sort_values("separability_auc", ascending=False).drop_duplicates("key")
    short = ok.head(a.n_shortlist)

    print(f"\nscored {len(df)} concepts")
    print(f"  usable AUC        : {df.separability_auc.notna().sum()}")
    print(f"  genre==text       : {(df.genre=='text').sum()}")
    print(f"  surface-form flag : {df.is_surface.sum()}")
    print(f"  AUC >= {a.min_auc}      : {(df.separability_auc>=a.min_auc).sum()}")
    print(f"  passing all filters: {len(ok)}  -> shortlist {len(short)}")
    print(f"\nseparability AUC quantiles:\n{df.separability_auc.describe(percentiles=[.1,.25,.5,.75,.9]).round(3)}")

    print(f"\n--- top {min(20,len(short))} shortlisted concepts ---")
    for r in short.head(20).itertuples():
        print(f"  auc={r.separability_auc:.3f} [{r.uid:12s}] {r.label}")
        print(f"        top: {r.top_tokens}")

    print(f"\n--- 8 REJECTED (surface-form) for contrast ---")
    for r in df[df.is_surface].head(8).itertuples():
        print(f"  auc={r.separability_auc:.3f} [{r.uid:12s}] {r.label}")

    short[["uid","label","genre","separability_auc","top_tokens"]].to_json(
        a.shortlist, orient="records", indent=2)
    print(f"\nwrote {a.out} and {a.shortlist}")


if __name__ == "__main__":
    main()
