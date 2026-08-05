# Playing-Style Clustering from StatsBomb Open Data
CS777 Term Project — Veerapat Tinnachote

Clusters soccer players into data-driven playing-style groups (rather
than official position labels) from StatsBomb's free open event data,
using PySpark for ingestion/feature engineering and Spark ML for
K-Means / Gaussian Mixture clustering. See the approved proposal for
full motivation, research question, and evaluation plan.

## Dataset

4 competition-seasons, 210 matches, ~650-700K raw events total:

| Competition | Matches |
|---|---|
| FIFA World Cup 2022 | 64 |
| UEFA Euro 2024 | 51 |
| Women's World Cup 2023 | 64 |
| UEFA Women's Euro 2022 | 31 |

Source: https://github.com/statsbomb/open-data (no account/API key needed).

**Note on scope change from the proposal:** the proposal listed
Champions League as a candidate competition. StatsBomb's open data only
publishes full match-by-match data for one CL *final* per season, not
full seasons, so it can't support player-level clustering — it's been
swapped for the two Women's tournaments instead, which do have full
open coverage and roughly double the player pool.

`data/sample_10_matches/` contains 10 real World Cup 2022 matches
(already downloaded) so you can run the pipeline end-to-end on a small
sample without waiting on the full download — useful for iterating on
the Spark code. Run `download_data.py` to get the full 210-match set for
the real project run.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Download the full dataset (run on a machine with normal internet
   access — this project uses PySpark locally per the assignment's
   "acceptable if done on your laptop" allowance):

   ```bash
   python scripts/download_data.py
   ```

   This populates `data/matches/`, `data/events/`, `data/lineups/`.

2. Run the full pipeline:

   ```bash
   spark-submit scripts/main.py
   ```

   This will:
   - load events + lineups into Spark DataFrames (`scripts/ingest.py`)
   - compute minutes played per player per match (`scripts/minutes.py`)
   - build per-90 feature vectors, split by TRAIN/TEST competitions
     (`scripts/features.py`, `config.py`)
   - sweep k for K-Means (elbow + silhouette), fit final K-Means + GMM
     (`scripts/clustering.py`)
   - evaluate: generalization check on held-out competitions, cross-tab
     vs. official positions, Messi/Ronaldo spot-check, pressing trend
     (`scripts/evaluate.py`)
   - write `output/player_features_train` / `_test` as Parquet

## Project layout

```
config.py            tunable constants (competitions, thresholds, k range)
scripts/
  download_data.py   pulls raw JSON from StatsBomb's GitHub repo
  ingest.py           Spark: raw JSON -> flat events_df / lineup_players_df
  minutes.py          per-player-per-match minutes played + primary position
  features.py         per-90 feature engineering (passing, shooting,
                       dribbling, defending, pressing, attacking-third presence)
  clustering.py        StandardScaler + KMeans + GaussianMixture, elbow/silhouette sweep
  evaluate.py           generalization check, position cross-tab, spot-checks, pressing trend
  main.py               orchestrates the full pipeline
data/                  raw StatsBomb JSON (gitignored except the sample)
output/                 Parquet feature/cluster tables (created on run)
```

## Status / what's verified so far

- Feature-engineering logic (minutes-played calc, pass completion,
  shots/goals, etc.) was validated with an offline pandas prototype
  against the 10 sample matches, cross-checked against known real-world
  results (e.g., Bukayo Saka's 4 shots / 2 goals vs. Iran matches the
  actual match record).
- The full Spark pipeline has **not yet been run end-to-end** — it
  needs a real PySpark environment with internet access to
  `raw.githubusercontent.com` (StatsBomb's data host), which this
  development sandbox doesn't have. Run it locally to generate the
  actual cluster results, elbow/silhouette numbers, and Messi/Ronaldo
  cluster assignment for the report.
- Feature definitions flagged in `features.py` comments (progressive
  pass threshold, aerial-duel-won proxy) are first-pass assumptions —
  worth double-checking against the StatsBomb Open Data Specification
  before finalizing numbers for the report.

## Academic integrity note

Per the assignment's two-line rule, cite any external code snippets you
add beyond this scaffold with a source-URL comment.
