#!/usr/bin/env python3
"""Emit self-contained inline SVG figures for the write-up.

All figures use CSS custom properties (var(--signal) etc) so they inherit the
page's light/dark palette rather than hard-coding colour.
"""
import argparse, json
import numpy as np
import pandas as pd

W, H = 720, 330
L, R, T, B = 62, 24, 18, 46          # margins


def _axes(xlab, ylab, xt, yt, x2px, y2px, ygrid=True):
    s = []
    if ygrid:
        for v, _ in yt:
            s.append(f'<line x1="{L}" y1="{y2px(v):.1f}" x2="{W-R}" y2="{y2px(v):.1f}" '
                     f'stroke="var(--rule)" stroke-width="1"/>')
    for v, lab in yt:
        s.append(f'<text x="{L-9}" y="{y2px(v)+4:.1f}" text-anchor="end" font-size="10.5" '
                 f'font-family="IBM Plex Mono, monospace" fill="var(--ink-faint)">{lab}</text>')
    for v, lab in xt:
        s.append(f'<text x="{x2px(v):.1f}" y="{H-B+20}" text-anchor="middle" font-size="10.5" '
                 f'font-family="IBM Plex Mono, monospace" fill="var(--ink-faint)">{lab}</text>')
    s.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-6}" text-anchor="middle" font-size="10.5" '
             f'font-family="IBM Plex Mono, monospace" fill="var(--ink-faint)">{xlab}</text>')
    s.append(f'<text x="14" y="{(T+H-B)/2:.0f}" text-anchor="middle" font-size="10.5" '
             f'font-family="IBM Plex Mono, monospace" fill="var(--ink-faint)" '
             f'transform="rotate(-90 14 {(T+H-B)/2:.0f})">{ylab}</text>')
    return "".join(s)


def fig_sweep(curves, out):
    """Hero: S(lambda) for all arms, with bootstrap CI bands."""
    c = curves[curves.metric == "S_rank"]
    lams = sorted(c.lam.unique())
    lo_y = min(0.25, c.lo.min() - .02); hi_y = max(0.85, c.hi.max() + .02)
    x2 = lambda v: L + (v - lams[0]) / (lams[-1] - lams[0]) * (W - L - R)
    y2 = lambda v: (H - B) - (v - lo_y) / (hi_y - lo_y) * (H - B - T)
    style = {"concept": ("var(--signal)", 3, "none"),
             "random": ("var(--flat)", 1.8, "none"),
             "style": ("var(--flat)", 1.8, "2 4"),
             "pure_v": ("var(--amber)", 1.8, "5 4")}
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Concept-ness of the '
         f'self-description against steering strength, with 95% bootstrap bands.">']
    yt = [(v, f"{v:.2f}") for v in np.arange(0.3, hi_y, 0.1)]
    xt = [(v, f"{v:+.1f}".replace("+0.0", "0")) for v in lams]
    s.append(_axes("steering strength λ  (units of ‖h‖)", "looks like the concept →", xt, yt, x2, y2))
    s.append(f'<line x1="{L}" y1="{y2(0.5):.1f}" x2="{W-R}" y2="{y2(0.5):.1f}" '
             f'stroke="var(--ink-faint)" stroke-dasharray="2 3" stroke-width="1"/>')
    s.append(f'<text x="{W-R+2}" y="{y2(0.5)+3:.1f}" font-size="9.5" '
             f'font-family="IBM Plex Mono, monospace" fill="var(--ink-faint)">chance</text>')
    s.append(f'<line x1="{x2(0):.1f}" y1="{T}" x2="{x2(0):.1f}" y2="{H-B}" stroke="var(--rule)"/>')
    for arm in ["pure_v", "style", "random", "concept"]:
        d = c[c.arm == arm].sort_values("lam")
        if d.empty: continue
        col, wdt, dash = style[arm]
        if arm == "concept":                       # CI band on the main arm
            up = " ".join(f"{x2(r.lam):.1f},{y2(r.hi):.1f}" for r in d.itertuples())
            dn = " ".join(f"{x2(r.lam):.1f},{y2(r.lo):.1f}" for r in reversed(list(d.itertuples())))
            s.append(f'<polygon points="{up} {dn}" fill="var(--signal)" opacity="0.13"/>')
        pts = " ".join(f"{x2(l):.1f},{y2(m):.1f}" for l, m in zip(d.lam, d["mean"]))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wdt}" '
                 f'stroke-linejoin="round" {"stroke-dasharray="+chr(34)+dash+chr(34) if dash!="none" else ""}/>')
        if arm == "concept":
            for l, m, lo, hi in zip(d.lam, d["mean"], d.lo, d.hi):
                s.append(f'<line x1="{x2(l):.1f}" y1="{y2(lo):.1f}" x2="{x2(l):.1f}" '
                         f'y2="{y2(hi):.1f}" stroke="{col}" stroke-width="1.2" opacity=".65"/>')
                s.append(f'<circle cx="{x2(l):.1f}" cy="{y2(m):.1f}" r="3" fill="{col}"/>')
    s.append("</svg>")
    open(out, "w").write("".join(s))
    return out


def fig_behavioral(beh, out):
    """B(lambda) with coherence on a second series."""
    real = beh[~beh.uid.str.startswith("__")]
    g = real.groupby("lam").agg(m=("rank_pct", "mean"), sd=("rank_pct", "std"),
                                n=("rank_pct", "size"), d2=("distinct2", "mean")).reset_index()
    g["se"] = g.sd / np.sqrt(g.n)
    lams = list(g.lam)
    x2 = lambda v: L + (v - lams[0]) / (lams[-1] - lams[0]) * (W - L - R)
    y2 = lambda v: (H - B) - (v - 0.2) / 0.85 * (H - B - T)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Behavioural concept presence and coherence versus steering strength.">']
    yt = [(v, f"{v:.1f}") for v in np.arange(0.2, 1.05, 0.2)]
    xt = [(v, f"{v:+.1f}".replace("+0.0", "0")) for v in lams]
    s.append(_axes("steering strength λ", "score", xt, yt, x2, y2))
    for col, key, dash in [("var(--signal)", "m", "none"), ("var(--flat)", "d2", "4 4")]:
        pts = " ".join(f"{x2(l):.1f},{y2(v):.1f}" for l, v in zip(g.lam, g[key]))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4" '
                 f'stroke-linejoin="round" {"stroke-dasharray="+chr(34)+dash+chr(34) if dash!="none" else ""}/>')
    for l, m, se in zip(g.lam, g.m, g.se):
        s.append(f'<line x1="{x2(l):.1f}" y1="{y2(m-1.96*se):.1f}" x2="{x2(l):.1f}" '
                 f'y2="{y2(m+1.96*se):.1f}" stroke="var(--signal)" stroke-width="1.2" opacity=".6"/>')
        s.append(f'<circle cx="{x2(l):.1f}" cy="{y2(m):.1f}" r="2.8" fill="var(--signal)"/>')
    s.append("</svg>")
    open(out, "w").write("".join(s))
    return out


def fig_sensitivity(curves, beh, out):
    """Overlay: does the self report move at the same strength as behaviour?"""
    sel = curves[(curves.metric == "S_rank") & (curves.arm == "concept")].set_index("lam")["mean"]
    lams = sorted(sel.index)
    bh = beh.reindex(lams)
    x2 = lambda v: L + (v - lams[0]) / (lams[-1] - lams[0]) * (W - L - R)
    y2 = lambda v: (H - B) - (v - 0.20) / 0.85 * (H - B - T)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Behavioural change and self reported change plotted against steering strength.">']
    yt = [(v, f"{v:.1f}") for v in np.arange(0.2, 1.05, 0.2)]
    xt = [(v, f"{v:+.1f}".replace("+0.0", "0")) for v in lams]
    s.append(_axes("steering strength", "concept present", xt, yt, x2, y2))
    for series, col, dash, wid in ((bh, "var(--flat)", "5 4", 2.4),
                                   (sel, "var(--signal)", "none", 3)):
        pts = " ".join(f"{x2(l):.1f},{y2(v):.1f}" for l, v in zip(lams, series.values))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wid}" '
                 f'stroke-linejoin="round" {"stroke-dasharray="+chr(34)+dash+chr(34) if dash!="none" else ""}/>')
        for l, v in zip(lams, series.values):
            s.append(f'<circle cx="{x2(l):.1f}" cy="{y2(v):.1f}" r="3" fill="{col}"/>')
    s.append("</svg>")
    open(out, "w").write("".join(s))
    return out


def fig_variance(dec, out):
    """Stacked bar: what explains where a description lands."""
    f = dec["factors"]
    parts = [("topic identity", f["topic"], "var(--signal)"),
             ("concept", f["concept"], "#7FB8BE"),
             ("the nudge λ", f["lam"], "var(--amber)"),
             ("sampling noise", f["residual"], "var(--flat)")]
    Wb, Hb = 720, 150
    s = [f'<svg viewBox="0 0 {Wb} {Hb}" role="img" aria-label="Share of description variation explained by each factor.">']
    x = 20; y = 34; bw = Wb - 40; bh = 40
    for lab, v, col in parts:
        w = max(1.5, v * bw)
        s.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bh}" fill="{col}"/>')
        if v > .04:
            s.append(f'<text x="{x+w/2:.1f}" y="{y+bh/2+4:.0f}" text-anchor="middle" font-size="11" '
                     f'font-family="IBM Plex Mono, monospace" fill="#fff">{v*100:.1f}%</text>')
        else:
            s.append(f'<line x1="{x+w/2:.1f}" y1="{y-4}" x2="{x+w/2:.1f}" y2="{y-14}" stroke="{col}" stroke-width="1.5"/>')
            s.append(f'<text x="{x+w/2:.1f}" y="{y-18}" text-anchor="middle" font-size="11" font-weight="600" '
                     f'font-family="IBM Plex Mono, monospace" fill="{col}">{v*100:.2f}%</text>')
        x += w
    lx = 20
    for lab, v, col in parts:
        s.append(f'<rect x="{lx}" y="{y+bh+18}" width="10" height="10" fill="{col}"/>')
        s.append(f'<text x="{lx+15}" y="{y+bh+27}" font-size="11" font-family="IBM Plex Sans, sans-serif" '
                 f'fill="var(--ink-soft)">{lab}</text>')
        lx += 26 + len(lab) * 6.4
    s.append("</svg>")
    open(out, "w").write("".join(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default="results/curves_v2.parquet")
    ap.add_argument("--behavioral", default="results/behavioral.parquet")
    ap.add_argument("--analysis", default="results/analysis_v2.json")
    ap.add_argument("--sweep", default="")
    ap.add_argument("--prefix", default="results/fig_")
    a = ap.parse_args()
    cur = pd.read_parquet(a.curves)
    beh = pd.read_parquet(a.behavioral)
    ana = json.load(open(a.analysis))
    print(fig_sweep(cur, a.prefix + "sweep.svg"))
    print(fig_behavioral(beh, a.prefix + "behavioral.svg"))
    print(fig_variance(ana["variance_decomposition"], a.prefix + "variance.svg"))
    used = set(pd.read_parquet(a.sweep).query("arm=='concept'").uid.unique()) if a.sweep else None
    if used:
        grid = sorted(cur.lam.unique())
        m = beh[beh.uid.isin(used) & beh.lam.isin(grid)].groupby("lam").rank_pct.mean()
        print(fig_sensitivity(cur, m, a.prefix + "sensitivity.svg"))


if __name__ == "__main__":
    main()
