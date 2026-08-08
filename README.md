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

Requires Python 3.10+ and Java 17 (for Spark). Developed and verified
on Python 3.14 / PySpark 4.1 / OpenJDK 17.

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
   python3 scripts/main.py      # or: spark-submit scripts/main.py
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

## Results (full 210-match run, verified)

The pipeline has been run end-to-end on the full dataset (exit 0, logs
in `results/`). 387 TRAIN players (men's tournaments) and 327 TEST
players (women's tournaments) met the 270-minute threshold.

- **Cluster count:** silhouette peaks at k=3 (0.327), but k=3 only
  recovers goalkeeper / defender / attacker. k=5 (silhouette 0.316,
  within the proposal's expected 5–8 range) is used via
  `config.K_OVERRIDE` for style-level resolution — both numbers and the
  rationale are reported.
- **k=5 clusters are interpretable style groups:** creative attackers
  (wingers/attacking mids, high key passes + dribbles), ball-playing
  center backs (highest passing volume/completion + progressive
  passes), goalkeepers, all-round fullbacks/defensive midfielders, and
  penalty-box forwards (low pass volume, high shots).
- **Headline spot-check confirmed:** Messi (creative-attacker cluster)
  and Ronaldo (penalty-box-forward cluster) land in different clusters
  despite both being listed as forwards.
- **Clusters cut across position labels** as hypothesized: e.g. the
  penalty-box cluster contains wingers alongside center forwards, and
  the creative cluster mixes wings, attacking mids, and forwards.
- **Generalization:** the TRAIN-fit scaler+KMeans applied to the
  held-out women's tournaments reproduces the same profile structure
  with closely matching per-cluster means.
- **K-Means vs GMM:** K-Means separates better (silhouette 0.327 vs
  0.209 at k=3); GMM's soft assignments surface blended-style players
  (e.g. hybrid wingers/forwards with low max cluster probability).
- **Pressing (descriptive):** both women's tournaments show more
  pressures per 90 player-minutes (12.6, 15.5) than the men's (10.5,
  11.8); men's pressing rose slightly from WC 2022 to Euro 2024.

Feature semantics (aerial duels via `aerial_won` flags, tackles via
duel type, key passes via shot/goal assist flags) were verified against
the StatsBomb Open Data Specification and cross-checked in the raw
sample JSON — see comments in `ingest.py` / `features.py`.

## Academic integrity note

Per the assignment's two-line rule, cite any external code snippets you
add beyond this scaffold with a source-URL comment.
