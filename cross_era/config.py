"""Paths for the standalone cross-era project.

Precomputed outputs ship in ./data so the demo (comps, map, catalog) runs with no
raw data. Rebuilding from scratch needs the play-by-play event store — point
NBA_EVENTS_DIR at a folder of <year>.parquet files (one per season, 1996-2025).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ASSETS_DIR = REPO_ROOT / "assets"

# Optional: only needed to rebuild embeddings from raw events.
RAW_EVENTS_DIR = Path(os.environ["NBA_EVENTS_DIR"]) if os.environ.get("NBA_EVENTS_DIR") else None

# Bundled / generated artifacts.
EMBEDDINGS = DATA_DIR / "cross_era_embeddings.parquet"
PROFILES = DATA_DIR / "cross_era_profiles.parquet"
BIOS = DATA_DIR / "player_bios_allyears.parquet"
MAP2D = DATA_DIR / "cross_era_map2d.csv"
ARCHETYPES = DATA_DIR / "cross_era_archetypes.csv"
MAP_PNG = ASSETS_DIR / "cross_era_map.png"

FIRST_SEASON, LAST_SEASON = 1996, 2025

# Style/production features (z-scored within season). Physical attrs from bios.
FEATURE_COLS = ["pts_pg", "reb_pg", "ast_pg", "stl_pg", "blk_pg", "tov_pg", "fga_pg",
                "fta_pg", "fg3a_pg", "fg3a_share", "ft_rate", "ast_per_fga",
                "oreb_share", "ts", "fg3pct", "ftpct", "usage_pg",
                "height_in", "weight", "age"]

DATA_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)
