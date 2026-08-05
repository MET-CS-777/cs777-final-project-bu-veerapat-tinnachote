"""
minutes.py

Computes minutes played per (match_id, player_id) from the lineup
position-segments, so downstream features can be normalized to a
per-90-minutes basis.

StatsBomb's lineup `positions` array records contiguous stretches of the
match a player occupied a given position, timestamped on a continuous
match-clock (mm:ss, not reset at half-time -- verified against sample
data: period 2 timestamps continue from ~45:00 onward, including
stoppage time). A player's total time on pitch is therefore just:

    (last segment's `to`, or match end if `to` is null) - (first segment's `from`)

Unused substitutes have an empty `positions` array and are excluded
(0 minutes played).

Also derives each player's primary (most common) on-field position
across their appearances, used later as the "official position" label
for the external validity check against clusters.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def _mmss_to_minutes(col):
    parts = F.split(col, ":")
    return parts.getItem(0).cast("double") + parts.getItem(1).cast("double") / 60.0


def compute_match_end_minutes(events_df: DataFrame) -> DataFrame:
    """Last observed event timestamp per match, in minutes (continuous
    clock). Used as the effective `to` time for players still on the
    pitch at full time (whose lineup `to` field is null)."""
    return events_df.groupBy("match_id").agg(
        (F.max("minute") + F.max("second") / 60.0).alias("match_end_minute")
    )


def compute_minutes_played(lineup_players_df: DataFrame, events_df: DataFrame) -> DataFrame:
    match_end = compute_match_end_minutes(events_df)

    segments = (
        lineup_players_df.filter(F.col("position_name").isNotNull())
        .withColumn("from_min", _mmss_to_minutes(F.col("from_ts")))
        .withColumn("to_min_raw", _mmss_to_minutes(F.col("to_ts")))
        .join(match_end, on="match_id", how="left")
        .withColumn(
            "to_min", F.coalesce(F.col("to_min_raw"), F.col("match_end_minute"))
        )
    )

    # Primary position = most-played position (by total segment duration)
    # for this player within this match.
    segments = segments.withColumn("segment_len", F.col("to_min") - F.col("from_min"))

    pos_totals = segments.groupBy(
        "match_id", "player_id", "position_name"
    ).agg(F.sum("segment_len").alias("pos_minutes"))

    w = Window.partitionBy("match_id", "player_id").orderBy(F.desc("pos_minutes"))
    primary_position = (
        pos_totals.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .select("match_id", "player_id", F.col("position_name").alias("primary_position"))
    )

    minutes_played = segments.groupBy(
        "match_id", "player_id", "player_name", "team_id", "team_name"
    ).agg(
        F.min("from_min").alias("first_from"),
        F.max("to_min").alias("last_to"),
    ).withColumn(
        "minutes_played", F.greatest(F.col("last_to") - F.col("first_from"), F.lit(0.0))
    ).join(primary_position, on=["match_id", "player_id"], how="left").select(
        "match_id", "player_id", "player_name", "team_id", "team_name",
        "minutes_played", "primary_position",
    )

    return minutes_played
