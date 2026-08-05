"""
evaluate.py

Implements the evaluation plan from the approved proposal:
  1. Generalization check (train/test-by-competition equivalent of a
     held-out split): fit scaler+KMeans/GMM on TRAIN_COMPETITIONS, apply
     the fitted models to TEST_COMPETITIONS' player features, and check
     whether the resulting per-cluster feature-profile means are
     consistent with the train-side profiles.
  2. External sanity check: cross-tabulate cluster assignment against
     each player's primary on-field position label.
  3. Qualitative spot-check: look up specific well-known players
     (e.g. Messi, Ronaldo) and report their assigned cluster.
  4. Pressing trend: compare average pressing_p90 pre/post a cutoff
     year (descriptive only, see config.PRESSING_TREND_CUTOFF_YEAR).
"""
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from features import FEATURE_COLUMNS


def cluster_profile_means(clustered_df: DataFrame, cluster_col: str) -> DataFrame:
    return clustered_df.groupBy(cluster_col).agg(
        F.count("*").alias("n_players"),
        *[F.mean(c).alias(f"mean_{c}") for c in FEATURE_COLUMNS],
    ).orderBy(cluster_col)


def apply_fitted_pipeline(test_features_df: DataFrame, scaler_model, cluster_model,
                           cluster_col: str) -> DataFrame:
    """Apply a scaler + clustering model fit on the TRAIN competitions to
    a held-out (TEST-competition) feature set, to check generalization."""
    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features_raw")
    assembled = assembler.transform(test_features_df)
    scaled = scaler_model.transform(assembled)
    return cluster_model.transform(scaled)


def position_cross_tab(clustered_df: DataFrame, cluster_col: str):
    return (
        clustered_df.groupBy(cluster_col, "position_label")
        .count()
        .orderBy(cluster_col, F.desc("count"))
    )


def spot_check_players(clustered_df: DataFrame, cluster_col: str, name_substrings):
    """name_substrings: list of case-insensitive substrings, e.g.
    ["Messi", "Ronaldo"]."""
    cond = None
    for s in name_substrings:
        c = F.col("player_name").contains(s)
        cond = c if cond is None else (cond | c)
    return clustered_df.filter(cond).select(
        "player_name", "position_label", cluster_col, *FEATURE_COLUMNS
    )


def pressing_trend(features_by_competition: DataFrame, cutoff_year: int):
    """features_by_competition must have a `season_year` column (join in
    main.py from matches metadata) alongside pressures_p90."""
    return (
        features_by_competition
        .withColumn(
            "era", F.when(F.col("season_year") < cutoff_year, "pre").otherwise("post")
        )
        .groupBy("era")
        .agg(F.mean("pressures_p90").alias("avg_pressures_p90"), F.count("*").alias("n"))
    )
