"""
Project configuration for CS777 Term Project:
"Playing-style clustering from StatsBomb open event data"
Student: Veerapat Tinnachote

All tunable constants live here so scripts don't hard-code values.
"""

# --- Dataset scope -----------------------------------------------------
# (competition_id, season_id, human-readable name) from StatsBomb's
# data/competitions.json. Chosen as the "broader multi-competition set":
# men's + women's World Cup, men's + women's Euros.
#
# NOTE: Champions League is NOT included even though it's on the
# public-data list of covered competitions -- StatsBomb only publishes
# full match-by-match CL data for a handful of finals (1 match per
# season), not full CL seasons, so it can't support this analysis.
# If you want a club-competition angle, La Liga (Messi-era Barcelona,
# competition_id=11) is the one StatsBomb releases with full-season
# open data.
COMPETITIONS = [
    (43, 106, "FIFA World Cup 2022"),
    (55, 282, "UEFA Euro 2024"),
    (72, 107, "Women's World Cup 2023"),
    (53, 106, "UEFA Women's Euro 2022"),
]
# Actual match counts (pulled from competitions.json / matches/*.json on
# 2026-08-05): 64 + 51 + 64 + 31 = 210 matches, ~3,000-3,500 events per
# match => roughly 650,000-700,000 raw events feeding the pipeline.

BASE_RAW_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# --- Local paths ---------------------------------------------------------
DATA_DIR = "data"
MATCHES_DIR = f"{DATA_DIR}/matches"
EVENTS_DIR = f"{DATA_DIR}/events"
LINEUPS_DIR = f"{DATA_DIR}/lineups"

FEATURES_OUTPUT_PATH = "output/player_features"
CLUSTER_OUTPUT_PATH = "output/player_clusters"

# --- Feature engineering ---------------------------------------------------
# Minimum minutes played across the selected competitions for a player to
# be included in clustering (filters out players with too little data for
# a stable per-90 profile).
MIN_MINUTES_PLAYED = 270  # ~3 full matches worth

# --- Clustering -----------------------------------------------------------
RANDOM_SEED = 42
K_RANGE = range(3, 11)  # candidate number of clusters to try (elbow/silhouette)

# Override the silhouette-chosen k (set to None to auto-pick the max-
# silhouette k). On the full 210-match run, silhouette peaks at k=3,
# which just recovers goalkeeper / defender / attacker -- too coarse to
# answer the research question about style sub-types WITHIN those roles
# (Messi and Ronaldo land in the same cluster at k=3). k=5 has nearly
# identical silhouette (0.3160 vs 0.3267), sits in the proposal's
# expected 5-8 range, and is the interpretability-driven choice the
# k-sweep table is meant to inform. Report both numbers.
K_OVERRIDE = 5

# --- Train/test (generalization check) -------------------------------------
# Competitions held out to test whether cluster profiles generalize.
# Fit on TRAIN_COMPETITIONS, evaluate profile consistency on TEST_COMPETITIONS.
TRAIN_COMPETITIONS = [(43, 106), (55, 282)]        # men's WC 2022 + Euro 2024
TEST_COMPETITIONS = [(72, 107), (53, 106)]          # women's WC 2023 + Euro 2022

# --- Secondary analysis: pressing trend ------------------------------------
# The proposal's pre/post-2017 era split is not usable with this dataset
# scope (all four competitions are 2022-2024, so the "pre" group would be
# empty). The pipeline instead reports pressing intensity per tournament,
# ordered by year -- see evaluate.pressing_trend().
