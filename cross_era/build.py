"""Build the cross-era embedding from raw play-by-play events.

Pipeline: per-(player,season) profiles from events -> merge physical bios ->
era-normalize within season -> PCA -> KMeans archetypes. Writes everything to
./data. Requires NBA_EVENTS_DIR (a folder of <year>.parquet event files).

Run: python -m cross_era.build
"""
from __future__ import annotations

import glob
import json
import os
import ssl
import time
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

SHOTS = ["MADE_2PT", "MISSED_2PT", "MADE_3PT", "MISSED_3PT"]

_SSL = ssl.create_default_context(); _SSL.check_hostname = False; _SSL.verify_mode = ssl.CERT_NONE
_HEADERS = {
    "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9", "Connection": "keep-alive",
    "Origin": "https://www.nba.com", "Referer": "https://www.nba.com/",
    "Sec-Fetch-Dest": "empty", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Site": "same-site",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"),
    "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    "sec-ch-ua-mobile": "?0", "sec-ch-ua-platform": '"Windows"',
}


def _season_profile(year: int, d: pd.DataFrame) -> pd.DataFrame:
    def by1(mask=None):
        x = d if mask is None else d[mask]
        return x.groupby("player1_id").size()

    def by_actor(col):
        if col not in d.columns:
            return pd.Series(dtype=float)
        x = pd.to_numeric(d[col], errors="coerce")
        return d[x > 0].groupby(col).size()

    pts = d.groupby("player1_id")["points"].sum()
    fga = by1(d["action"].isin(SHOTS))
    fg3a = by1(d["action"].isin(["MADE_3PT", "MISSED_3PT"]))
    fg3m = by1(d["action"] == "MADE_3PT")
    fta = by1(d["action"].isin(["FT_MAKE", "FT_MISS"]))
    ftm = by1(d["action"] == "FT_MAKE")
    reb = by1(d["action"].isin(["REB_OFF", "REB_DEF"]))
    oreb = by1(d["action"] == "REB_OFF")
    tov = by1(d["action"] == "TURNOVER")
    games = d.groupby("player1_id")["game_id"].nunique()
    ast, stl, blk = by_actor("assist_player"), by_actor("steal_player"), by_actor("block_player")

    df = pd.DataFrame({"pts": pts, "fga": fga, "fg3a": fg3a, "fg3m": fg3m, "fta": fta,
                       "ftm": ftm, "reb": reb, "oreb": oreb, "tov": tov, "g": games,
                       "ast": ast, "stl": stl, "blk": blk}).fillna(0.0)
    df = df[(df["g"] >= 20) & (df.index < 1_000_000_000)].copy()  # drop team-rebound pseudo-players
    g, fga_s = df["g"], df["fga"].clip(lower=1)
    out = pd.DataFrame(index=df.index)
    out["season"] = year
    for c in ["pts", "reb", "ast", "stl", "blk", "tov", "fga", "fta", "fg3a"]:
        out[f"{c}_pg"] = df[c] / g
    out["fg3a_share"] = df["fg3a"] / fga_s
    out["ft_rate"] = df["fta"] / fga_s
    out["ast_per_fga"] = df["ast"] / fga_s
    out["oreb_share"] = df["oreb"] / df["reb"].clip(lower=1)
    out["ts"] = df["pts"] / (2 * (df["fga"] + 0.44 * df["fta"]).clip(lower=1))
    out["fg3pct"] = df["fg3m"] / df["fg3a"].clip(lower=1)
    out["ftpct"] = df["ftm"] / df["fta"].clip(lower=1)
    out["usage_pg"] = (df["fga"] + 0.44 * df["fta"] + df["tov"]) / g
    out["g"] = g
    return out


def build_profiles() -> pd.DataFrame:
    if C.RAW_EVENTS_DIR is None:
        raise RuntimeError("Set NBA_EVENTS_DIR to a folder of <year>.parquet event files to rebuild.")
    frames = []
    for fp in sorted(glob.glob(str(C.RAW_EVENTS_DIR / "*.parquet"))):
        yr = int(os.path.basename(fp)[:4])
        frames.append(_season_profile(yr, pd.read_parquet(fp)))
    prof = pd.concat(frames).reset_index().rename(columns={"index": "player_id"})
    prof["player_id"] = prof["player_id"].astype(int)
    prof.to_parquet(C.PROFILES, index=False)
    print(f"profiles: {len(prof)} player-seasons, {prof['season'].nunique()} seasons")
    return prof


def _bio_url(season: str) -> str:
    return ("https://stats.nba.com/stats/leaguedashplayerbiostats?College=&Conference=&"
            "Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&"
            "GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&Month=0&"
            "OpponentTeamID=0&Outcome=&PORound=0&PerMode=PerGame&Period=0&PlayerExperience=&"
            "PlayerPosition=&Season=" + season + "&SeasonSegment=&SeasonType=Regular%20Season&"
            "ShotClockRange=&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight=")


def fetch_bios() -> pd.DataFrame:
    if C.BIOS.exists():
        return pd.read_parquet(C.BIOS)
    frames = []
    for yr in range(C.FIRST_SEASON, C.LAST_SEASON + 1):
        season = f"{yr}-{str(yr + 1)[-2:]}"
        try:
            req = urllib.request.Request(_bio_url(season), headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=60, context=_SSL) as r:
                j = json.loads(r.read().decode())
            rs = j["resultSets"][0]
            d = pd.DataFrame(rs["rowSet"], columns=rs["headers"]); d["season"] = yr
            frames.append(d); print(f"  bios {season}: {len(d)}")
        except Exception as e:  # noqa: BLE001
            print(f"  bios {season} failed: {e}")
        time.sleep(0.6)
    allb = pd.concat(frames, ignore_index=True)
    keep = {"PLAYER_ID": "player_id", "PLAYER_NAME": "name", "AGE": "age",
            "PLAYER_HEIGHT_INCHES": "height_in", "PLAYER_WEIGHT": "weight"}
    out = allb[["season"] + [k for k in keep if k in allb.columns]].rename(
        columns={k: v for k, v in keep.items() if k in allb.columns})
    for c in ("player_id", "age", "height_in", "weight"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["player_id"]); out["player_id"] = out["player_id"].astype(int)
    out.to_parquet(C.BIOS, index=False)
    return out


def era_normalize(prof: pd.DataFrame):
    cols = [c for c in C.FEATURE_COLS if c in prof.columns]
    Z = prof.copy()
    for c in cols:
        Z[c] = prof.groupby("season")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))
    X = np.nan_to_num(Z[cols].to_numpy(float), nan=0.0)
    return X, Z, cols


def build(n_pca=12):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    bios = fetch_bios()
    prof = build_profiles()
    prof = prof.merge(bios[[c for c in ["player_id", "season", "height_in", "weight", "age"]
                            if c in bios.columns]], on=["player_id", "season"], how="left")
    names = {int(k): v for k, v in zip(*[bios.dropna(subset=["player_id"]).sort_values("season")
             .drop_duplicates("player_id", keep="last")[c] for c in ("player_id", "name")])}
    X, Z, used = era_normalize(prof)
    emb = PCA(n_components=min(n_pca, X.shape[1]), random_state=0).fit_transform(X)
    Z = Z.reset_index(drop=True)
    Z["name"] = Z["player_id"].map(lambda p: names.get(int(p), str(p)))
    Z["archetype"] = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(X)
    out = pd.concat([Z[["player_id", "name", "season", "archetype", "g"]],
                     pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])], axis=1)
    out.to_parquet(C.EMBEDDINGS, index=False)
    print(f"WROTE {C.EMBEDDINGS} ({len(out)} player-seasons, dim {emb.shape[1]})")


if __name__ == "__main__":
    build()
