"""
position_families.py

Supplementary analysis alongside the primary global (all-positions)
clustering in main.py: within each traditional position family (GK,
defender, midfielder, forward), cluster players against each other only,
to show that a single official position label can itself contain
multiple distinct playing styles (e.g. is a "Center Forward" a target
man or a false nine?).

Design choices, and why:
  - Each family gets its OWN StandardScaler, fit only on that family's
    players. Fitting one global scaler and then slicing by position
    would make every goalkeeper look uniformly "extreme" relative to
    outfield players and hide the real variation *within* the family,
    which is the entire point of this analysis.
  - k is chosen per family by the same silhouette-sweep method as the
    primary model -- NOT fixed to match any particular taxonomy. A
    family might turn out to have 2 sub-styles or 5; the number and
    the resulting cluster names are read off the data, not assumed in
    advance.
  - This analysis pools ALL FOUR competitions (no train/test split).
    It is an illustrative companion to the primary model, which is the
    one held to the train/test generalization check in main.py --
    splitting further by position family here would leave too few
    players per family per split to cluster meaningfully.
  - The goalkeeper family uses GK_FEATURE_COLUMNS (save %, claims,
    sweeper actions, plus passing) instead of FEATURE_COLUMNS, since
    shot/dribble/tackle features are structurally ~0 for keepers and
    would add noise, not signal.

Run with:
    python3 scripts/position_families.py
"""
import json
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPTS_DIR))

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions as F

import config
from ingest import build_spark_session, load_events, load_lineups
from minutes import compute_minutes_played
from features import build_player_features, FEATURE_COLUMNS, GK_FEATURE_COLUMNS

RANDOM_SEED = config.RANDOM_SEED
MIN_PLAYERS_PER_CLUSTER = 12  # floor used to cap how large k can go per family


def position_family_col(position_label_col):
    """Spark-native (no Python UDF -- those need the defining module on
    every executor's import path, which local[*] multi-process mode
    doesn't guarantee) string-match into GK / DEF / MID / FWD."""
    c = position_label_col
    return (
        F.when(c == "Goalkeeper", "GK")
        .when(c.contains("Back"), "DEF")  # covers "*Back" and "*Wing Back"
        .when(c.contains("Forward") | c.contains("Wing") | c.contains("Attacking Midfield"), "FWD")
        .when(c.contains("Midfield"), "MID")
        .otherwise("OTHER")
    )


def k_range_for(n_players: int):
    max_k = max(2, min(6, n_players // MIN_PLAYERS_PER_CLUSTER))
    return range(2, max_k + 1)


def sweep_and_fit(scaled_df, k_range, seed):
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette")
    results = []
    best = None
    for k in k_range:
        km = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=seed)
        model = km.fit(scaled_df)
        preds = model.transform(scaled_df)
        sil = evaluator.evaluate(preds)
        results.append({"k": k, "silhouette": sil, "training_cost": model.summary.trainingCost})
        print(f"    k={k}  silhouette={sil:.4f}  training_cost={model.summary.trainingCost:.1f}")
        if best is None or sil > best[1]:
            best = (k, sil, model, preds)
    return results, best


def main():
    spark = build_spark_session("position-families")
    spark.sparkContext.setLogLevel("WARN")

    print("Loading events + lineups (all 4 competitions, pooled) ...")
    events_df = load_events(spark, config.EVENTS_DIR).cache()
    lineups_df = load_lineups(spark, config.LINEUPS_DIR)
    minutes_df = compute_minutes_played(lineups_df, events_df).cache()

    print("Building player features (all competitions combined) ...")
    features = build_player_features(events_df, minutes_df, config.MIN_MINUTES_PLAYED).cache()
    n_total = features.count()
    print(f"  {n_total} players total (>= {config.MIN_MINUTES_PLAYED} min)")

    features = features.withColumn("family", position_family_col(F.col("position_label")))

    report = {}
    os.makedirs("results", exist_ok=True)
    os.makedirs("output/position_families", exist_ok=True)

    for family in ["GK", "DEF", "MID", "FWD"]:
        print(f"\n{'='*70}\nFAMILY: {family}\n{'='*70}")
        fam_df = features.filter(F.col("family") == family)
        n = fam_df.count()
        print(f"  n players: {n}")
        if n < 2 * MIN_PLAYERS_PER_CLUSTER:
            print(f"  Skipping {family}: fewer than {2*MIN_PLAYERS_PER_CLUSTER} players.")
            continue

        cols = GK_FEATURE_COLUMNS if family == "GK" else FEATURE_COLUMNS
        assembler = VectorAssembler(inputCols=cols, outputCol="features_raw")
        assembled = assembler.transform(fam_df)
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
        scaler_model = scaler.fit(assembled)
        scaled = scaler_model.transform(assembled)

        krange = k_range_for(n)
        print(f"  Sweeping k in {list(krange)} (feature set: {'GK' if family=='GK' else 'standard'}, {len(cols)} features)")
        sweep_results, (best_k, best_sil, model, preds) = sweep_and_fit(scaled, krange, RANDOM_SEED)
        print(f"  Chosen k={best_k} (silhouette={best_sil:.4f}) by max silhouette")

        profile = preds.groupBy("cluster").agg(
            F.count("*").alias("n"), *[F.mean(c).alias(c) for c in cols]
        ).orderBy("cluster").collect()
        profile_rows = [row.asDict() for row in profile]
        print("  Cluster profiles:")
        for row in profile_rows:
            print(f"    cluster {row['cluster']}: n={row['n']}  " +
                  "  ".join(f"{c}={row[c]:.3f}" for c in cols))

        cross = preds.groupBy("cluster", "position_label").count().orderBy("cluster", F.desc("count")).collect()
        cross_rows = [row.asDict() for row in cross]

        top_players = (
            preds.select("player_name", "position_label", "cluster", *cols)
            .orderBy("cluster")
            .collect()
        )
        # sample up to 6 players per cluster for reporting
        by_cluster = {}
        for row in top_players:
            by_cluster.setdefault(row["cluster"], []).append(row.asDict())
        sample_players = {c: rows[:8] for c, rows in by_cluster.items()}

        # spot check named players if present (Messi/Ronaldo land in FWD)
        spot = preds.filter(
            F.col("player_name").contains("Messi") | F.col("player_name").contains("Ronaldo")
        ).select("player_name", "position_label", "cluster", *cols).collect()
        spot_rows = [row.asDict() for row in spot]
        if spot_rows:
            print("  Spot check:")
            for row in spot_rows:
                print(f"    {row['player_name']} ({row['position_label']}) -> cluster {row['cluster']}")

        report[family] = {
            "n": n,
            "features_used": cols,
            "k_sweep": sweep_results,
            "chosen_k": best_k,
            "chosen_silhouette": best_sil,
            "cluster_profiles": profile_rows,
            "position_cross_tab": cross_rows,
            "sample_players": sample_players,
            "spot_check": spot_rows,
        }

        preds.select("player_id", "player_name", "position_label", "family", "cluster", *cols).write.mode("overwrite").parquet(
            f"output/position_families/{family}"
        )

    with open("results/position_family_clusters.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nWrote results/position_family_clusters.json")

    spark.stop()


if __name__ == "__main__":
    main()
