"""Re-project saved CLaMP 2 embeddings at a chosen density, without re-embedding.

`clamp2_dataset_probe.py` writes every piece's 768-d embedding to
`clamp2_embeddings.npz`, so changing how many points a figure shows needs only a fresh t-SNE over a
subset -- CPU-only, seconds to a couple of minutes, no GPU and no MIDI re-embedding.

Why a balanced version is usually the more readable one: at full density Hooktheory contributes
23,462 of 28,060 points (84%), so the eight other corpora are drawn into a small fraction of the ink
and whichever is plotted last wins every overlap. Capping each corpus at --max_per_dataset gives
each one comparable visual weight; corpora smaller than the cap are used in full and the legend
reports what each contributed.

t-SNE is not a projection that can be subsetted after the fact: neighbourhoods and cluster spacing
depend on the whole input set, so a capped figure is a genuinely different projection from the full
one, not a filtered view of it. Both are honest; the caption should say which.

Usage:
    python scripts/eval/custom_eval/replot_clamp2_tsne.py --max_per_dataset 500
    python scripts/eval/custom_eval/replot_clamp2_tsne.py --max_per_dataset 500 --out figures/balanced.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="logs/custom_eval/clamp2_probe_full/clamp2_embeddings.npz")
    ap.add_argument("--max_per_dataset", type=int, default=500,
                    help="cap per corpus (-1 = all, i.e. reproduce the full-density figure)")
    ap.add_argument("--out", default=None,
                    help="default: <npz dir>/tsne_by_dataset_max<N>.pdf")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=int, default=None, help="default: min(30, n/10)")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    embeddings, labels = data["music_embeddings"], data["dataset_labels"]
    names = sorted(set(labels.tolist()))

    rng = np.random.default_rng(args.seed)
    keep = []
    for name in names:
        idx = np.flatnonzero(labels == name)
        if args.max_per_dataset != -1 and len(idx) > args.max_per_dataset:
            idx = rng.choice(idx, args.max_per_dataset, replace=False)
        keep.append(np.sort(idx))
    keep = np.concatenate(keep)
    emb, lab = embeddings[keep], labels[keep]
    print(f"{len(keep)} of {len(labels)} points "
          f"(cap {'none' if args.max_per_dataset == -1 else args.max_per_dataset}):")
    for name in names:
        print(f"  {name:24s} {int((lab == name).sum()):6d} of {int((labels == name).sum()):6d}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    perplexity = args.perplexity or min(30, max(5, len(emb) // 10))
    print(f"\nt-SNE over {len(emb)} points (perplexity {perplexity}) ...")
    projected = TSNE(n_components=2, perplexity=perplexity, init="pca",
                     random_state=args.seed).fit_transform(emb)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    # largest corpora first so the small ones are not buried under them
    order = sorted(names, key=lambda n: -int((lab == n).sum()))
    colour = {name: cmap(i % 10) for i, name in enumerate(names)}
    for name in order:
        mask = lab == name
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   label=f"{name} ({int(mask.sum())})", color=colour[name], s=25, alpha=0.75)
    ax.set_title("t-SNE of CLaMP2 music embeddings, colored by dataset")
    handles, text = ax.get_legend_handles_labels()
    by_name = dict(zip(text, handles))
    ax.legend([by_name[t] for t in sorted(by_name)], sorted(by_name),
              loc="best", fontsize=8, markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    suffix = "all" if args.max_per_dataset == -1 else str(args.max_per_dataset)
    out = Path(args.out) if args.out else Path(args.npz).parent / f"tsne_by_dataset_max{suffix}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
