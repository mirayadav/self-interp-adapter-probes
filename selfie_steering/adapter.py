#!/usr/bin/env python3
"""
Scalar-affine SelfIE adapter: f(h) = alpha * (h/||h||) + b.

Prefers the upstream `selfie_adapters` package when installed; otherwise reads
the released .safetensors directly (verified format: tensors `bias` (d,) and
`log_scale` (1,), metadata `normalize_input`, `projection_type`). The fallback
exists so the pipeline can be smoke-tested on a laptop where the upstream repo
and its GPU dependencies are not installed.

`random_like` builds an untrained adapter with the geometry measured in Phase 0
(alpha=7.17, ||b||=20.87) so smoke runs exercise realistic magnitudes.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F


class Adapter:
    def __init__(self, alpha: float, bias: torch.Tensor, normalize_input: bool = True,
                 source: str = "?"):
        self.alpha = float(alpha)
        self.bias = bias.float()
        self.normalize_input = normalize_input
        self.source = source

    @property
    def dim(self) -> int:
        return self.bias.shape[0]

    @classmethod
    def load(cls, path: str) -> "Adapter":
        from safetensors import safe_open
        from safetensors.torch import load_file
        d = load_file(path)
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
        if "bias" not in d or "log_scale" not in d:
            raise ValueError(f"{path} is not a scalar_affine adapter "
                             f"(keys: {list(d)})")
        return cls(alpha=float(torch.exp(d["log_scale"][0])),
                   bias=d["bias"],
                   normalize_input=str(meta.get("normalize_input", "true")).lower() == "true",
                   source=path)

    @classmethod
    def random_like(cls, dim: int, alpha: float = 7.17, bias_norm: float = 20.87,
                    seed: int = 0) -> "Adapter":
        g = torch.Generator().manual_seed(seed)
        b = torch.randn(dim, generator=g)
        return cls(alpha=alpha, bias=b / b.norm() * bias_norm,
                   normalize_input=True, source=f"random(dim={dim})")

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d) -> soft tokens (..., d)."""
        h = h.float()
        if self.normalize_input:
            h = F.normalize(h, dim=-1)
        return self.alpha * h + self.bias.to(h.device)

    def __repr__(self):
        return (f"Adapter(alpha={self.alpha:.3f}, ||b||={self.bias.norm():.3f}, "
                f"dim={self.dim}, normalize_input={self.normalize_input}, "
                f"source={self.source})")


def load_mean_vector(path: str, layer: int) -> torch.Tensor:
    from safetensors.torch import load_file
    return load_file(path)[f"layer_{layer}"].float()
