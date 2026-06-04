"""Render the cross-era latent space to a single shareable PNG (assets/cross_era_map.png).

    python -m cross_era.render
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

MARQUEE = ["Michael Jordan", "Kobe Bryant", "LeBron James", "Stephen Curry", "Nikola Jokić",
           "Victor Wembanyama", "Tim Duncan", "Shaquille O'Neal", "Allen Iverson",
           "Kevin Garnett", "Steve Nash", "Dirk Nowitzki", "James Harden",
           "Giannis Antetokounmpo", "Kevin Durant", "Chris Paul", "Russell Westbrook"]


def render():
    if not C.MAP2D.exists():
        from .report import build_2d
        build_2d()
    df = pd.read_csv(C.MAP2D)
    try:
        cat = pd.read_csv(C.ARCHETYPES)
        labels = dict(zip(cat["archetype"], cat["label"]))
    except Exception:  # noqa: BLE001
        labels = {}

    fig, ax = plt.subplots(figsize=(15, 10))
    cmap = plt.get_cmap("tab10")
    for k in sorted(df["archetype"].unique()):
        sub = df[df["archetype"] == k]
        ax.scatter(sub["pc1"], sub["pc2"], s=6, alpha=0.20, color=cmap(k % 10),
                   label=f"{k}: {labels.get(k, '')}", linewidths=0)
    for nm in MARQUEE:
        cand = df[df["name"] == nm]
        if cand.empty:
            cand = df[df["name"].str.contains(nm.split()[-1], case=False, na=False)]
        if cand.empty:
            continue
        r = cand.sort_values("g", ascending=False).iloc[0]
        ax.scatter([r["pc1"]], [r["pc2"]], s=44, color="black", zorder=5)
        ax.annotate(f"{r['name'].split()[-1]} '{str(r['season_str'])[2:4]}", (r["pc1"], r["pc2"]),
                    fontsize=9, fontweight="bold", xytext=(5, 4), textcoords="offset points", zorder=6)
    ax.set_title("30 Years of NBA Players in One Era-Normalized Space (1996–2026)\n"
                 "12,916 player-seasons · style from 17M play-by-play events, z-scored within season",
                 fontsize=13)
    ax.set_xlabel("latent dim 1"); ax.set_ylabel("latent dim 2")
    ax.legend(title="archetype (unsupervised)", loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.MAP_PNG, dpi=150, bbox_inches="tight")
    print(f"WROTE {C.MAP_PNG}")


if __name__ == "__main__":
    render()
