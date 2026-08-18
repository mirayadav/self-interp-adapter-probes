#!/usr/bin/env python3
"""
Core model plumbing shared by every phase: activation extraction, SelfIE soft-token
injection, and residual-stream steering hooks.

Deliberately model-agnostic so the whole pipeline can be smoke-tested on
Qwen2.5-0.5B-Instruct on a CPU laptop before renting a GPU. The only
Llama-specific thing in the upstream repo is the hard-coded SelfIE template with
`<|reserved_special_token_0|>`; `build_selfie_template` derives an equivalent
template from any tokenizer's chat template.

Conventions (kept identical to the upstream repo so results stay comparable):
  * residual stream at layer L  = `output_hidden_states[L+1]`
    (index 0 is the embedding output) -- see examples/contrastive_topic_vector.py
  * topic activation h          = last token of "Tell me about {title}." with
                                  add_generation_prompt=True
  * contrastive vector          = h - mean_vector[layer]
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLAMA_PLACEHOLDER = "<|reserved_special_token_0|>"


# ---------------------------------------------------------------- model loading

@dataclass
class LM:
    model: torch.nn.Module
    tokenizer: object
    device: str
    layer: int
    name: str

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def blocks(self):
        """The list of transformer blocks, for hook registration."""
        for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = self.model
            try:
                for p in path.split("."):
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        raise RuntimeError(f"cannot locate transformer blocks on {type(self.model)}")


def load_lm(name: str, layer: int, device: Optional[str] = None,
            dtype: Optional[torch.dtype] = None) -> LM:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    tok.clean_up_tokenization_spaces = False
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map=device if device == "cuda" else None)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    if layer < 0:
        layer = model.config.num_hidden_layers + layer
    return LM(model=model, tokenizer=tok, device=device, layer=layer, name=name)


# ------------------------------------------------------------ activation extract

@torch.no_grad()
def residual_at(lm: LM, texts: Sequence[str], layer: Optional[int] = None,
                pool: str = "last", batch_size: int = 16,
                response_spans: Optional[Sequence[tuple]] = None) -> torch.Tensor:
    """
    Residual-stream activations for a batch of already-formatted strings.

    pool="last"     -> last non-pad token (matches the upstream topic-vector code)
    pool="mean"     -> mean over non-pad tokens
    pool="response" -> mean over the token span given in `response_spans`
                       (char offsets), used for AxBench concept vectors where the
                       concept lives in the response rather than the prompt.
    """
    layer = lm.layer if layer is None else layer
    outs = []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i:i + batch_size])
        enc = lm.tokenizer(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024,
                           add_special_tokens=False,
                           return_offsets_mapping=(pool == "response")).to(lm.device)
        offsets = enc.pop("offset_mapping", None)
        hs = lm.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                      output_hidden_states=True).hidden_states[layer + 1]
        mask = enc.attention_mask.unsqueeze(-1).to(hs.dtype)
        if pool == "last":
            # left padding => the true last token is always index -1
            v = hs[:, -1, :]
        elif pool == "mean":
            v = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        elif pool == "response":
            sel = torch.zeros_like(mask)
            for b in range(hs.shape[0]):
                lo, hi = response_spans[i + b]
                for t, (s, e) in enumerate(offsets[b].tolist()):
                    if e > s and s >= lo and e <= hi:
                        sel[b, t, 0] = 1.0
            sel = sel * mask
            v = (hs * sel).sum(1) / sel.sum(1).clamp(min=1)
        else:
            raise ValueError(pool)
        outs.append(v.float().cpu())
    return torch.cat(outs)


def format_topic_prompt(lm: LM, title: str) -> str:
    """"Tell me about {title}." as a generation-prompted chat turn."""
    return lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Tell me about {title}."}],
        tokenize=False, add_generation_prompt=True)


def format_qa(lm: LM, instruction: str, response: str) -> tuple[str, tuple[int, int]]:
    """Full user+assistant turn; returns (text, char span of the response)."""
    text = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction},
         {"role": "assistant", "content": response}], tokenize=False)
    lo = text.rfind(response)
    return text, (lo, lo + len(response))


# ------------------------------------------------------------- SelfIE injection

def build_selfie_template(lm: LM) -> tuple[str, int]:
    """
    The explanation-seeking prompt, with a single-token placeholder marking where
    the soft token goes. Returns (template_text, placeholder_token_id).

    Mirrors the upstream template exactly for Llama; for other models it is
    reconstructed from the tokenizer's own chat template so the smoke path works.
    """
    ph = LLAMA_PLACEHOLDER
    pid = lm.tokenizer.convert_tokens_to_ids(ph)
    if pid is None or pid == lm.tokenizer.unk_token_id:
        # fall back to any single-token string not otherwise used
        for cand in ("<|fim_prefix|>", "<|box_start|>", "<|extra_0|>", "ĠÃŸ", "ø"):
            cid = lm.tokenizer.convert_tokens_to_ids(cand)
            if cid is not None and cid != lm.tokenizer.unk_token_id:
                ph, pid = cand, cid
                break
        else:
            raise RuntimeError("no usable single-token placeholder for this tokenizer")

    body = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": f'What is the meaning of "{ph}"?'}],
        tokenize=False, add_generation_prompt=True)
    template = body + f'The meaning of "{ph}" is "'
    return template, pid


@torch.no_grad()
def selfie_describe(lm: LM, soft_tokens: torch.Tensor, n: int = 6,
                    max_new_tokens: int = 40, temperature: float = 0.5,
                    template: Optional[tuple[str, int]] = None,
                    seed: Optional[int] = None, batch_size: int = 48) -> list[list[str]]:
    """
    Inject each soft token into the SelfIE prompt and sample `n` descriptions.

    soft_tokens: (B, d) already passed through the adapter.
    Returns a list of B lists of n strings.
    """
    tmpl, pid = template or build_selfie_template(lm)
    enc = lm.tokenizer(tmpl, return_tensors="pt", add_special_tokens=False).to(lm.device)
    positions = (enc.input_ids[0] == pid).nonzero().flatten().tolist()
    if not positions:
        raise RuntimeError("placeholder token not found in SelfIE template")

    embed = lm.model.get_input_embeddings()
    base = embed(enc.input_ids)                       # (1, T, d)
    if soft_tokens.ndim == 1:
        soft_tokens = soft_tokens[None]

    if seed is not None:
        torch.manual_seed(seed)

    B = soft_tokens.shape[0]
    # flatten (B, n) into rows so several soft tokens share one generate() call
    st_all = soft_tokens.repeat_interleave(n, dim=0).to(dtype=base.dtype, device=base.device)
    flat: list[str] = []
    for i in range(0, B * n, batch_size):
        st = st_all[i:i + batch_size]
        emb = base.repeat(st.shape[0], 1, 1).clone()
        for p in positions:
            emb[:, p, :] = st
        gen = lm.model.generate(
            inputs_embeds=emb,
            attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device),
            max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=lm.tokenizer.eos_token_id)
        for g in gen:
            t = lm.tokenizer.decode(g, skip_special_tokens=True).strip()
            flat.append(t.rsplit('"', 1)[0] if '"' in t else t)
    return [flat[k * n:(k + 1) * n] for k in range(B)]


# ----------------------------------------------------------------- steering hook

@contextlib.contextmanager
def steering(lm: LM, vector: Optional[torch.Tensor], coeff: float,
             layer: Optional[int] = None, positions: str = "all"):
    """
    Add `coeff * vector` to the residual stream at `layer` for the duration of the
    block. positions="all" matches CAA (all token positions, including generated
    ones); "prompt" restricts to the initial forward pass.
    """
    if vector is None or coeff == 0.0:
        yield
        return
    layer = lm.layer if layer is None else layer
    blk = lm.blocks()[layer]
    v = vector.to(lm.device)
    state = {"first": True}

    def hook(_module, _inp, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if positions == "prompt" and not state["first"]:
            return output
        state["first"] = False
        hidden = hidden + coeff * v.to(hidden.dtype)
        return (hidden,) + tuple(output[1:]) if is_tuple else hidden

    handle = blk.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate_steered(lm: LM, prompts: Sequence[str], vector: Optional[torch.Tensor],
                     coeff: float, max_new_tokens: int = 60, temperature: float = 0.7,
                     batch_size: int = 8, layer: Optional[int] = None,
                     seed: Optional[int] = None) -> list[str]:
    """Generate continuations with the residual stream steered by coeff*vector."""
    if seed is not None:
        torch.manual_seed(seed)
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i:i + batch_size])
        enc = lm.tokenizer(chunk, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(lm.device)
        with steering(lm, vector, coeff, layer=layer):
            gen = lm.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=lm.tokenizer.eos_token_id)
        for j, g in enumerate(gen):
            new = g[enc.input_ids.shape[1]:]
            outs.append(lm.tokenizer.decode(new, skip_special_tokens=True).strip())
    return outs
