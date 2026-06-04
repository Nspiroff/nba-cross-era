"""Query + describe the cross-era space (runs off bundled ./data, no raw events).

    python -m cross_era.report comp "LeBron James"   # cross-era analogs for a player
    python -m cross_era.report catalog               # the archetype catalog
    python -m cross_era.report map                    # 2D coordinates for plotting
    python -m cross_era.report report                 # catalog + 2D + markdown writeup
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config as C
from .build import era_normalize

_PATTERNS = [
    ("Glass-crashing big", [("oreb_share", +1), ("reb_pg", +1), ("fg3a_share", -1)]),
    ("Rim-protecting big", [("blk_pg", +1), ("reb_pg", +1), ("fg3a_pg", -1)]),
    ("Stretch big", [("blk_pg", +1), ("fg3a_share", +1), ("reb_pg", +1)]),
    ("Floor-spacing shooter", [("fg3a_share", +1), ("fg3pct", +1), ("reb_pg", -1)]),
    ("Lead playmaker", [("ast_pg", +1), ("ast_per_fga", +1), ("usage_pg", +1)]),
    ("High-usage scorer", [("usage_pg", +1), ("pts_pg", +1), ("fta_pg", +1)]),
    ("Slashing FT-drawer", [("ft_rate", +1), ("fta_pg", +1), ("fg3a_share", -1)]),
    ("Two-way guard", [("stl_pg", +1), ("ast_pg", +1), ("fg3a_share", +1)]),
    ("3-and-D role wing", [("fg3pct", +1), ("stl_pg", +1), ("usage_pg", -1)]),
]
_TERM = {"fg3a_share": "3PT rate", "fg3a_pg": "3PT volume", "fg3pct": "3PT%", "ts": "efficiency",
         "ast_pg": "assists", "ast_per_fga": "playmaking", "reb_pg": "rebounds",
         "oreb_share": "off-rebounding", "blk_pg": "blocks", "stl_pg": "steals",
         "usage_pg": "usage", "pts_pg": "scoring", "fta_pg": "FT volume", "ft_rate": "FT rate"}


def _season_str(y):
    return f"{int(y)}-{str(int(y) + 1)[-2:]}"


def _load():
    emb = pd.read_parquet(C.EMBEDDINGS)
    ecols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    E = emb[ecols].to_numpy(float)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return emb, E, En


def comps(query, season=None, k=8, cross_era_only=True, exclude_same_player=True):
    emb, E, En = _load()
    cand = emb[emb["name"].str.contains(query, case=False, na=False)]
    if season is not None:
        cand = cand[cand["season"] == season]
    if cand.empty:
        print(f"no player matching '{query}'"); return None
    i = cand.sort_values("g", ascending=False).index[0]
    qs, qpid = int(emb.loc[i, "season"]), int(emb.loc[i, "player_id"])
    sims = En @ En[i]
    seen, res = set(), []
    for j in np.argsort(-sims):
        if j == i or (exclude_same_player and int(emb.loc[j, "player_id"]) == qpid):
            continue
        if cross_era_only and abs(int(emb.loc[j, "season"]) - qs) < 3:
            continue
        pid = int(emb.loc[j, "player_id"])
        if pid in seen:
            continue
        seen.add(pid)
        res.append({"name": emb.loc[j, "name"], "season": _season_str(emb.loc[j, "season"]),
                    "similarity": round(float(sims[j]), 3)})
        if len(res) >= k:
            break
    print(f"\nCross-era comps for {emb.loc[i,'name']} ({_season_str(qs)}):")
    for r in res:
        print(f"   {r['similarity']:.3f}  {r['name']} ({r['season']})")
    return pd.DataFrame(res)


def _label(zmean):
    best, score_thr = None, 0.4
    for name, feats in _PATTERNS:
        if not all(f in zmean for f, _ in feats):
            continue
        if not all((zmean[f] > 0) == (s > 0) for f, s in feats):
            continue
        sc = float(np.mean([s * zmean[f] for f, s in feats]))
        if sc > score_thr:
            best, score_thr = name, sc
    return best or f"High {_TERM.get(zmean.index[-1], zmean.index[-1])}, low {_TERM.get(zmean.index[0], zmean.index[0])}"


def archetype_catalog():
    emb, E, _ = _load()
    prof = pd.read_parquet(C.PROFILES)
    X, Z, used = era_normalize(prof)
    znorm = pd.DataFrame(X, columns=used); znorm["archetype"] = emb["archetype"].to_numpy()
    centroids = {c: E[emb["archetype"].to_numpy() == c].mean(0) for c in sorted(emb["archetype"].unique())}
    rows = []
    for c in sorted(emb["archetype"].unique()):
        zmean = znorm[znorm["archetype"] == c][used].mean().sort_values()
        mask = emb["archetype"].to_numpy() == c
        d = np.linalg.norm(E[mask] - centroids[c], axis=1)
        ex = emb[mask].assign(_d=d).sort_values("_d").drop_duplicates("player_id").head(6)
        rows.append({"archetype": int(c), "n": int(mask.sum()), "label": _label(zmean),
                     "exemplars": "; ".join(f"{r['name']} ({_season_str(r['season'])})"
                                            for _, r in ex.iterrows())})
    cat = pd.DataFrame(rows); cat.to_csv(C.ARCHETYPES, index=False)
    for _, r in cat.iterrows():
        print(f"[{r['archetype']}] {r['label']} (n={r['n']})\n     e.g. {r['exemplars']}")
    return cat


def build_2d():
    from sklearn.decomposition import PCA
    emb, E, _ = _load()
    prof = pd.read_parquet(C.PROFILES)
    X, Z, used = era_normalize(prof)
    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    out = emb[["player_id", "name", "season", "archetype", "g"]].copy()
    out["pc1"], out["pc2"] = xy[:, 0], xy[:, 1]
    out["season_str"] = out["season"].map(_season_str)
    out.to_csv(C.MAP2D, index=False)
    print(f"WROTE {C.MAP2D} ({len(out)} rows)")
    return out


def write_report():
    cat = archetype_catalog()
    marquee = ["Michael Jordan", "Stephen Curry", "Allen Iverson", "Kevin Garnett",
               "LeBron James", "Tim Duncan", "Nikola Jokić", "Victor Wembanyama"]
    lines = ["# Cross-Era NBA Player Representation (1996–2026)\n",
             "12,916 player-seasons in one era-normalized latent space. See README.md.\n",
             "## Cross-era analogs (unsupervised)\n"]
    for nm in marquee:
        r = comps(nm, k=4)
        if r is not None and len(r):
            lines.append(f"- **{nm}** → " + ", ".join(f"{x['name']} ({x['season']})"
                                                       for _, x in r.iterrows()))
    lines.append("\n## Archetypes\n")
    for _, r in cat.iterrows():
        lines.append(f"- **{r['label']}** ({r['n']}) — {'; '.join(r['exemplars'].split('; ')[:4])}")
    (C.DATA_DIR.parent / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWROTE {C.DATA_DIR.parent / 'REPORT.md'}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    pc = sub.add_parser("comp"); pc.add_argument("name"); pc.add_argument("--season", type=int)
    sub.add_parser("catalog"); sub.add_parser("map"); sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "comp":
        comps(args.name, season=args.season)
    elif args.cmd == "catalog":
        archetype_catalog()
    elif args.cmd == "map":
        build_2d()
    else:
        build_2d(); write_report()


if __name__ == "__main__":
    main()
