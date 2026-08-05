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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import functions as F

import config
from ingest import build_spark_session, load_events, load_lineups
from minutes import compute_minutes_played
from features import build_player_features
from clustering import assemble_and_scale, choose_k, fit_gmm, fit_kmeans, sweep_kmeans_k
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
    best_k = choose_k(sweep_results)
    print(f"Selected k={best_k} by silhouette (inspect the printed table above "
          f"and override in config.py if a different k is more interpretable).")

    print("\nFitting final KMeans and GMM on TRAIN ...")
    kmeans_model, train_kmeans_pred = fit_kmeans(train_scaled, best_k, config.RANDOM_SEED)
    gmm_model, train_gmm_pred = fit_gmm(train_scaled, best_k, config.RANDOM_SEED)

    print("\n=== TRAIN cluster profiles (KMeans) ===")
    cluster_profile_means(train_kmeans_pred, "cluster_kmeans").show(truncate=False)

    print("=== TRAIN vs official position labels (KMeans) ===")
    position_cross_tab(train_kmeans_pred, "cluster_kmeans").show(50, truncate=False)

    print("=== Spot-check: Messi / Ronaldo (TRAIN, KMeans) ===")
    spot_check_players(train_kmeans_pred, "cluster_kmeans", ["Messi", "Ronaldo"]).show(truncate=False)

    print("\n=== Generalization check: applying TRAIN-fit models to TEST competitions ===")
    test_kmeans_pred = apply_fitted_pipeline(test_features, scaler_model, kmeans_model, "cluster_kmeans")
    print("TEST cluster profiles (using TRAIN-fit KMeans model):")
    cluster_profile_means(test_kmeans_pred, "cluster_kmeans").show(truncate=False)
    print("Compare these means to the TRAIN cluster profiles above -- similar "
          "per-cluster feature means indicate the clusters generalize beyond "
          "the competitions they were fit on.")

    print("\n=== Pressing trend (descriptive only) ===")
    train_with_year = train_features.join(
        minutes_train.select("player_id", "match_id").join(match_comp_map, "match_id")
        .select("player_id", "season_year").dropDuplicates(["player_id"]),
        on="player_id", how="left",
    )
    pressing_trend(train_with_year, config.PRESSING_TREND_CUTOFF_YEAR).show()

    print("\nWriting outputs ...")
    os.makedirs("output", exist_ok=True)
    train_kmeans_pred.drop("features_raw", "features").write.mode("overwrite").parquet(
        f"{config.FEATURES_OUTPUT_PATH}_train"
    )
    test_kmeans_pred.drop("features_raw", "features").write.mode("overwrite").parquet(
        f"{config.FEATURES_OUTPUT_PATH}_test"
    )
    print("Done.")

    spark.stop()


if __name__ == "__main__":
    main()
