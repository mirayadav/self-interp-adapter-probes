#!/usr/bin/env python3
"""Smoke test: exercises every code path in core.py on a tiny CPU model.

Descriptions will be nonsense (0.5B model, untrained random adapter). The point
is to validate shapes, hook placement, template construction and span pooling
before spending GPU-hours.

    python selfie_steering/smoke_test.py
"""
import sys
import torch
from selfie_steering.core import (
    LM, load_lm, residual_at, format_topic_prompt, format_qa,
    build_selfie_template, selfie_describe, steering, generate_steered)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER = 12
ok = lambda m: print(f"  [PASS] {m}")


def main():
    print(f"loading {MODEL} on CPU ...")
    lm = load_lm(MODEL, layer=LAYER, device="cpu", dtype=torch.float32)
    print(f"  dim={lm.dim} layers={lm.n_layers} layer={lm.layer}")
    assert len(lm.blocks()) == lm.n_layers
    ok("block discovery")

    # 1. topic activations, last-token pooling
    titles = ["William Wallace", "Photosynthesis", "The Fibonacci sequence"]
    prompts = [format_topic_prompt(lm, t) for t in titles]
    H = residual_at(lm, prompts, pool="last")
    assert H.shape == (3, lm.dim), H.shape
    assert torch.isfinite(H).all()
    ok(f"residual_at(last) -> {tuple(H.shape)}, norms {H.norm(dim=1).tolist()}")

    # distinct prompts must give distinct activations
    c = torch.nn.functional.cosine_similarity(H[0], H[1], dim=0).item()
    assert c < 0.999, f"activations not distinct (cos={c})"
    ok(f"distinct topics give distinct activations (cos={c:.3f})")

    # 2. response-span pooling
    texts, spans = zip(*[format_qa(lm, "What is a cell?", "A cell is the basic unit of life."),
                         format_qa(lm, "Name a river.", "The Nile is a major river in Africa.")])
    V = residual_at(lm, list(texts), pool="response", response_spans=list(spans))
    assert V.shape == (2, lm.dim) and torch.isfinite(V).all()
    ok(f"residual_at(response span) -> {tuple(V.shape)}")

    Vm = residual_at(lm, list(texts), pool="mean")
    assert not torch.allclose(V, Vm), "response pooling identical to full mean"
    ok("response-span pooling differs from whole-sequence mean")

    # 3. SelfIE template + injection
    tmpl, pid = build_selfie_template(lm)
    n_ph = (lm.tokenizer(tmpl, return_tensors="pt", add_special_tokens=False)
            .input_ids[0] == pid).sum().item()
    assert n_ph >= 1
    ok(f"template built, {n_ph} placeholder slot(s), token id {pid}")

    # random scalar-affine adapter: f(h) = alpha * h/||h|| + b
    torch.manual_seed(0)
    alpha, b = 7.17, torch.randn(lm.dim) * (20.87 / (lm.dim ** 0.5))
    soft = alpha * torch.nn.functional.normalize(H, dim=-1) + b
    descs = selfie_describe(lm, soft, n=2, max_new_tokens=12, temperature=0.7, seed=0)
    assert len(descs) == 3 and all(len(d) == 2 for d in descs)
    ok("selfie_describe returned 3x2 descriptions")
    for t, d in zip(titles, descs):
        print(f"      {t!r:26s} -> {d[0][:60]!r}")

    # 4. soft token actually matters: zero vs real input must differ
    zero_soft = alpha * torch.zeros(1, lm.dim) + b
    d0 = selfie_describe(lm, zero_soft, n=1, max_new_tokens=12, temperature=0.0, seed=0)
    d1 = selfie_describe(lm, soft[:1], n=1, max_new_tokens=12, temperature=0.0, seed=0)
    ok(f"bias-only greedy  : {d0[0][0][:55]!r}")
    ok(f"with activation   : {d1[0][0][:55]!r}")
    if d0[0][0] == d1[0][0]:
        print("      [WARN] identical at greedy decoding -- expected for an "
              "untrained random adapter, must NOT happen with the trained one")

    # 5. steering hook
    v = torch.randn(lm.dim); v = v / v.norm()
    base = generate_steered(lm, prompts[:1], None, 0.0, max_new_tokens=12, temperature=0.0)
    stee = generate_steered(lm, prompts[:1], v, 40.0, max_new_tokens=12, temperature=0.0)
    ok(f"unsteered : {base[0][:55]!r}")
    ok(f"steered   : {stee[0][:55]!r}")
    assert base[0] != stee[0], "steering hook had no effect at coeff=40"
    ok("steering hook changes generation")

    # 6. hook is removed cleanly
    after = generate_steered(lm, prompts[:1], None, 0.0, max_new_tokens=12, temperature=0.0)
    assert after[0] == base[0], "hook leaked past the context manager"
    ok("hook removed cleanly (generation identical to baseline)")

    # 7. coeff=0 is a true no-op
    z = generate_steered(lm, prompts[:1], v, 0.0, max_new_tokens=12, temperature=0.0)
    assert z[0] == base[0]
    ok("coeff=0 is a no-op")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    sys.exit(main())
