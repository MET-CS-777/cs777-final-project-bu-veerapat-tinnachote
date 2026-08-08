"""
main.py

End-to-end pipeline entry point:
  1. Load raw events + lineups with Spark (ingest.py)
  2. Compute minutes played per player per match (minutes.py)
  3. Build per-90 feature vectors, split by TRAIN/TEST competitions (features.py)
  4. Fit K-Means + GMM on the TRAIN split, sweep k for elbow/silhouette (clustering.py)
  5. Evaluate: apply TRAIN-fit models to TEST split, cross-tab vs position
     labels, spot-check known players, pressing trend (evaluate.py)
  6. Write player_features + player_clusters out to output/

Run with:
    spark-submit scripts/main.py
or, for local dev with default local[*] master:
    python scripts/main.py
"""
import json
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPTS_DIR))  # project root, for config.py

from pyspark.sql import functions as F

import config
from ingest import build_spark_session, load_events, load_lineups
from minutes import compute_minutes_played
from features import build_player_features
from clustering import (
    assemble_and_scale,
    choose_k,
    fit_gmm,
    fit_kmeans,
    silhouette_score,
    sweep_kmeans_k,
)
from evaluate import (
    apply_fitted_pipeline,
    cluster_profile_means,
    position_cross_tab,
    pressing_trend,
    spot_check_players,
)


def load_match_competition_map(spark):
    """match_id -> (competition_id, season_id, season_year) from the
    downloaded matches/*.json files, used to split TRAIN/TEST and to tag
    features with a season year for the pressing-trend analysis."""
    rows = []
    for comp_id, season_id, name in config.COMPETITIONS:
        path = f"{config.MATCHES_DIR}/{comp_id}/{season_id}.json"
        with open(path, encoding="utf-8") as f:
            matches = json.load(f)
        for m in matches:
            season_year = int(str(m["match_date"])[:4])
            rows.append((m["match_id"], comp_id, season_id, name, season_year))

    return spark.createDataFrame(
        rows, schema=["match_id", "competition_id", "season_id", "competition_name", "season_year"]
    )


def main():
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("Loading events + lineups ...")
    events_df = load_events(spark, config.EVENTS_DIR).cache()
    lineups_df = load_lineups(spark, config.LINEUPS_DIR)

    print("Computing minutes played ...")
    minutes_df = compute_minutes_played(lineups_df, events_df).cache()

    match_comp_map = load_match_competition_map(spark).cache()

    train_ids = {c[0] * 1_000_000 + c[1] for c in config.TRAIN_COMPETITIONS}
    test_ids = {c[0] * 1_000_000 + c[1] for c in config.TEST_COMPETITIONS}
    match_comp_map = match_comp_map.withColumn(
        "comp_season_key", F.col("competition_id") * 1_000_000 + F.col("season_id")
    )
    train_match_ids = match_comp_map.filter(
        F.col("comp_season_key").isin(train_ids)
    ).select("match_id")
    test_match_ids = match_comp_map.filter(
        F.col("comp_season_key").isin(test_ids)
    ).select("match_id")

    events_train = events_df.join(train_match_ids, "match_id", "inner")
    events_test = events_df.join(test_match_ids, "match_id", "inner")
    minutes_train = minutes_df.join(train_match_ids, "match_id", "inner")
    minutes_test = minutes_df.join(test_match_ids, "match_id", "inner")

    print("\nBuilding TRAIN player features ...")
    train_features = build_player_features(
        events_train, minutes_train, config.MIN_MINUTES_PLAYED
    ).cache()
    print(f"  {train_features.count()} players (TRAIN)")

    print("Building TEST player features ...")
    test_features = build_player_features(
        events_test, minutes_test, config.MIN_MINUTES_PLAYED
    ).cache()
    print(f"  {test_features.count()} players (TEST)")

    print("\nScaling TRAIN features and sweeping k for KMeans (elbow/silhouette) ...")
    train_scaled, scaler_model = assemble_and_scale(train_features)
    sweep_results = sweep_kmeans_k(train_scaled, config.K_RANGE, config.RANDOM_SEED)
    silhouette_k = choose_k(sweep_results)
    best_k = config.K_OVERRIDE or silhouette_k
    if best_k != silhouette_k:
        print(f"Silhouette-max k={silhouette_k}, using K_OVERRIDE={best_k} "
              f"from config.py (see rationale there).")
    else:
        print(f"Selected k={best_k} by silhouette (inspect the printed table above "
              f"and set K_OVERRIDE in config.py if a different k is more interpretable).")

    print("\nFitting final KMeans and GMM on TRAIN ...")
    kmeans_model, train_kmeans_pred = fit_kmeans(train_scaled, best_k, config.RANDOM_SEED)
    gmm_model, train_gmm_pred = fit_gmm(train_scaled, best_k, config.RANDOM_SEED)

    print("\n=== TRAIN cluster profiles (KMeans) ===")
    cluster_profile_means(train_kmeans_pred, "cluster_kmeans").show(truncate=False)

    print("=== TRAIN vs official position labels (KMeans) ===")
    position_cross_tab(train_kmeans_pred, "cluster_kmeans").show(50, truncate=False)

    print("=== Spot-check: Messi / Ronaldo (TRAIN, KMeans) ===")
    spot_check_players(train_kmeans_pred, "cluster_kmeans", ["Messi", "Ronaldo"]).show(truncate=False)

    print("\n=== KMeans vs GMM comparison (TRAIN) ===")
    print(f"Silhouette  KMeans: {silhouette_score(train_kmeans_pred, 'cluster_kmeans'):.4f}  "
          f"GMM: {silhouette_score(train_gmm_pred, 'cluster_gmm'):.4f}")

    print("\nTRAIN cluster profiles (GMM):")
    cluster_profile_means(train_gmm_pred, "cluster_gmm").show(truncate=False)

    print("=== Spot-check: Messi / Ronaldo (TRAIN, GMM) ===")
    spot_check_players(train_gmm_pred, "cluster_gmm", ["Messi", "Ronaldo"]).show(truncate=False)

    # GMM soft assignments: players whose highest cluster probability is
    # low are the "blended style" players a hard KMeans label hides.
    from pyspark.ml.functions import vector_to_array
    train_gmm_pred = train_gmm_pred.withColumn(
        "gmm_max_prob", F.array_max(vector_to_array("probability"))
    )
    print("=== Most style-blended players (lowest GMM max-probability) ===")
    train_gmm_pred.orderBy("gmm_max_prob").select(
        "player_name", "position_label", "cluster_gmm", F.round("gmm_max_prob", 3).alias("gmm_max_prob")
    ).show(15, truncate=False)

    print("\n=== Generalization check: applying TRAIN-fit models to TEST competitions ===")
    test_kmeans_pred = apply_fitted_pipeline(test_features, scaler_model, kmeans_model, "cluster_kmeans")
    print("TEST cluster profiles (using TRAIN-fit KMeans model):")
    cluster_profile_means(test_kmeans_pred, "cluster_kmeans").show(truncate=False)
    print("Compare these means to the TRAIN cluster profiles above -- similar "
          "per-cluster feature means indicate the clusters generalize beyond "
          "the competitions they were fit on.")

    print("\n=== Pressing trend across tournaments (descriptive only) ===")
    pressing_trend(events_df, minutes_df, match_comp_map).show(truncate=False)

    print("\nWriting outputs ...")
    os.makedirs("output", exist_ok=True)
    gmm_cols = train_gmm_pred.select("player_id", "cluster_gmm", "gmm_max_prob")
    (
        train_kmeans_pred.join(gmm_cols, "player_id", "left")
        .drop("features_raw", "features")
        .write.mode("overwrite").parquet(f"{config.FEATURES_OUTPUT_PATH}_train")
    )
    test_kmeans_pred.drop("features_raw", "features").write.mode("overwrite").parquet(
        f"{config.FEATURES_OUTPUT_PATH}_test"
    )
    print("Done.")

    spark.stop()


if __name__ == "__main__":
    main()
