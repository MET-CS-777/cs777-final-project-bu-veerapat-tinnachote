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

3. Run the within-position-family companion analysis (optional, adds
   ~2-3 min; requires step 1's full dataset):

   ```bash
   python3 scripts/position_families.py
   ```

   This is a supplementary analysis alongside the global clustering
   above: it clusters goalkeepers against goalkeepers, defenders
   against defenders, midfielders against midfielders, and forwards
   against forwards -- each family gets its own StandardScaler and its
   own k-sweep -- to show that a single official position label can
   itself contain multiple distinct styles. Pools all four competitions
   (no train/test split, unlike the primary model) to keep each
   family's sample large enough to cluster meaningfully. Writes
   `results/position_family_clusters.json` and
   `output/position_families/{GK,DEF,MID,FWD}` as Parquet.

## Project layout

```
config.py            tunable constants (competitions, thresholds, k range)
scripts/
  download_data.py   pulls raw JSON from StatsBomb's GitHub repo
  ingest.py           Spark: raw JSON -> flat events_df / lineup_players_df
  minutes.py          per-player-per-match minutes played + primary position
  features.py         per-90 feature engineering (passing, shooting,
                       dribbling, defending, pressing, attacking-third presence,
                       goalkeeper shot-stopping/claims/sweeping)
  clustering.py        StandardScaler + KMeans + GaussianMixture, elbow/silhouette sweep
  evaluate.py           generalization check, position cross-tab, spot-checks, pressing trend
  main.py               orchestrates the full (global) pipeline
  position_families.py  supplementary: cluster within each traditional
                         position family (GK/DEF/MID/FWD) separately
data/                  raw StatsBomb JSON (gitignored except the sample)
output/                 Parquet feature/cluster tables (created on run)
results/                pipeline run logs + position_family_clusters.json
report/                 report.html (full write-up with charts)
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
- **k=5 clusters are interpretable style groups, named from their
  own stats (not position labels)** — cluster names describe the
  1-2 features that most separate that cluster from the other four,
  not the position that happens to correlate with it:
  High Dribbles & Key Passes, High Passing & Progression, Minimal
  On-Ball Output, High Tackles & Interceptions, High Conversion &
  Aerials.
- **Headline spot-check confirmed:** Messi (High Dribbles & Key
  Passes) and Ronaldo (High Conversion & Aerials) land in different
  clusters despite both being listed as forwards.
- **Clusters cut across position labels** as hypothesized: e.g. the
  High Conversion & Aerials cluster contains wingers alongside center
  forwards, and High Dribbles & Key Passes mixes wings, attacking
  mids, and forwards.
- **Generalization:** the TRAIN-fit scaler+KMeans applied to the
  held-out women's tournaments reproduces the same profile structure
  with closely matching per-cluster means.
- **K-Means vs GMM:** K-Means separates better (silhouette 0.327 vs
  0.209 at k=3); GMM's soft assignments surface blended-style players
  (e.g. hybrid wingers/forwards with low max cluster probability).
- **Pressing (descriptive):** both women's tournaments show more
  pressures per 90 player-minutes (12.6, 15.5) than the men's (10.5,
  11.8); men's pressing rose slightly from WC 2022 to Euro 2024.
- **Within-position-family clustering** (`position_families.py`,
  pooling all 4 competitions): clustering goalkeepers, defenders,
  midfielders, and forwards *separately* against only their own family
  — own scaler, own k-sweep. Silhouette favored k=2 for all four.
  `config.POSITION_FAMILY_K_OVERRIDE` lets k be bumped per family where
  the runner-up is close (same idea as the global model's `K_OVERRIDE`);
  the script checks any override against a minimum-cluster-size floor
  before accepting it. DEF → k=3 passed (all 3 resulting clusters ≥ 90
  players) and splits defenders into no-nonsense stoppers, overlapping
  attacking fullbacks, and deep ball-playing center backs. MID → k=4 was
  tried and **rejected**: it produced a 4-player outlier cluster well
  under the floor, so MID stays at k=2 — reported as a negative result,
  not silently dropped. Restricted to forwards only (166 players, no
  other positions in the comparison), Ronaldo lands in a low-pass/
  high-shot-conversion/high-aerial cluster and Messi in a high-pass/
  high-dribble/high-key-pass cluster — the same separation the global
  model found, this time using only players who share his position tag.

Feature semantics (aerial duels via `aerial_won` flags, tackles via
duel type, key passes via shot/goal assist flags) were verified against
the StatsBomb Open Data Specification and cross-checked in the raw
sample JSON — see comments in `ingest.py` / `features.py`.

## Academic integrity note

Per the assignment's two-line rule, cite any external code snippets you
add beyond this scaffold with a source-URL comment.
