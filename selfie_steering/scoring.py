#!/usr/bin/env python3
"""
Shared scoring: text -> concept-ness and fluency.

Two scoring modes, and the distinction matters for the result.

`cosine`  raw cos(embedding(text), embedding(concept_label)). Simple but NOT
          trustworthy on its own: if descriptions merely become longer or more
          generic as lambda grows, cosine to *any* label can drift upward, which
          would read as a positive result.

`rank`    cosine to the true concept measured against a fixed pool of distractor
          concepts: reported as percentile rank and as the margin over the best
          distractor. This is the primary metric. A generic drift lifts the true
          concept and the distractors together and therefore leaves rank flat,
          whereas genuine concept tracking moves rank. The distractor pool is
          held fixed across all lambda so the comparison is like-for-like.

Fluency guards the interpretation: the paper documents self-interpretation
staying fluent while becoming ungrounded, and steering at high coefficients
degenerates into repetition. `distinct2` is a cheap repetition detector and is
the only one used as a GATE in behavioral.py. `nll_per_token` under the
UNSTEERED model is the more principled measure -- it also catches fluent but
off-distribution text -- but it is recorded for interpretation only, not used to
filter. Both agree in practice (each roughly doubles/halves by |lambda|=2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


# --------------------------------------------------------------------- embedder

class Embedder:
    """Wraps a sentence-transformers model (default: the paper's GTE-large)."""

    def __init__(self, name: str = "thenlper/gte-large", device: Optional[str] = None,
                 batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(name, device=device)
        self.batch_size = batch_size
        self.name = name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        e = self.model.encode(list(texts), batch_size=self.batch_size,
                              convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=False)
        return e.astype(np.float32)


# ------------------------------------------------------------------ concept-ness

@dataclass
class ConceptScorer:
    """
    target_label : the concept whose presence we are scoring
    distractors  : fixed pool of other concept labels (>= 20 recommended)
    """
    embedder: Embedder
    target_label: str
    distractors: Sequence[str]

    def __post_init__(self):
        self._t = self.embedder.encode([self.target_label])[0]
        self._d = self.embedder.encode(list(self.distractors))

    def score(self, texts: Sequence[str]) -> dict:
        if len(texts) == 0:
            return {"cosine": np.nan, "rank_pct": np.nan, "margin": np.nan}
        E = self.embedder.encode(texts)
        cos_t = E @ self._t                       # (n,)
        cos_d = E @ self._d.T                     # (n, n_distract)
        rank_pct = (cos_t[:, None] > cos_d).mean(axis=1)
        margin = cos_t - cos_d.max(axis=1)
        return {"cosine": float(np.mean(cos_t)),
                "rank_pct": float(np.mean(rank_pct)),
                "margin": float(np.mean(margin)),
                "cosine_all": cos_t.tolist(),
                "rank_pct_all": rank_pct.tolist()}


def keyword_rate(texts: Sequence[str], keywords: Sequence[str]) -> float:
    """Non-embedding second opinion: fraction of texts containing any keyword.

    Kept because the embedding metric is also used to *select* concepts; if both
    metrics move together the result is not an artifact of one embedding space.
    """
    if not texts or not keywords:
        return float("nan")
    kw = [k.lower().strip() for k in keywords if len(k.strip()) > 2]
    return float(np.mean([any(k in t.lower() for k in kw) for t in texts]))


# ---------------------------------------------------------------------- fluency

def distinct_n(text: str, n: int = 2) -> float:
    toks = text.split()
    if len(toks) < n + 1:
        return 1.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


@torch.no_grad()
def nll_per_token(lm, texts: Sequence[str], batch_size: int = 8) -> list[float]:
    """Mean NLL per token under the UNSTEERED model. Degenerate or off-manifold
    text scores high. Steering is only interpretable where this stays near the
    unsteered baseline."""
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = [t for t in texts[i:i + batch_size]]
        enc = lm.tokenizer(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(lm.device)
        logits = lm.model(**enc).logits[:, :-1]
        tgt = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].bool()
        ll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1),
            reduction="none").reshape(tgt.shape)
        for b in range(len(chunk)):
            m = mask[b]
            out.append(ll[b][m].mean().item() if m.any() else float("nan"))
    return out


def fluency(texts: Sequence[str], lm=None) -> dict:
    d = {"distinct2": float(np.mean([distinct_n(t, 2) for t in texts])) if texts else np.nan,
         "mean_words": float(np.mean([len(t.split()) for t in texts])) if texts else np.nan,
         "empty_frac": float(np.mean([len(t.strip()) == 0 for t in texts])) if texts else np.nan}
    if lm is not None and texts:
        v = nll_per_token(lm, list(texts))
        d["nll_per_token"] = float(np.nanmean(v))
    return d
