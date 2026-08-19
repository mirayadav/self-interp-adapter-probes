#!/usr/bin/env python3
"""Assemble the blog post, inlining freshly generated figure SVGs.

Kept as a build step (rather than hand-edited HTML) so the post can be
regenerated whenever the underlying numbers change.
"""
import argparse, json, pathlib

def svg(p):
    return pathlib.Path(p).read_text()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="results/fig_")
    ap.add_argument("--out", default="results/post.html")
    ap.add_argument("--body", default="results/post_body.html")
    a = ap.parse_args()
    body = pathlib.Path(a.body).read_text()
    for name in ("sweep", "behavioral", "variance", "sensitivity"):
        body = body.replace(f"<!--FIG:{name}-->", svg(f"{a.figdir}{name}.svg"))
    pathlib.Path(a.out).write_text(pathlib.Path("results/post_head.html").read_text() + body)
    print(a.out)

if __name__ == "__main__":
    main()
