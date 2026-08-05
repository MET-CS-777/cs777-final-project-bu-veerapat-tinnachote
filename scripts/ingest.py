"""
ingest.py

Spark ingestion layer: reads the raw StatsBomb JSON (one array-of-events
file per match, one array-of-teams file per match for lineups) into two
flat Spark DataFrames:

  - events_df: one row per event, tagged with match_id
  - lineup_players_df: one row per (match_id, team, player, position
    segment), tagged with match_id

Each events/{match_id}.json file is a JSON array (not JSON-lines), so we
read with multiLine=True and recover match_id from the file path since
StatsBomb does not repeat it inside every event record.
"""
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


MATCH_ID_PATTERN = r"(\d+)\.json$"


def _with_match_id(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "match_id",
        F.regexp_extract(F.input_file_name(), MATCH_ID_PATTERN, 1).cast("long"),
    )


def load_events(spark: SparkSession, events_dir: str) -> DataFrame:
    """Load all events/*.json files into one flat events DataFrame.

    Only the fields the feature-engineering step actually needs are
    selected explicitly (StatsBomb events have ~30 different event-type
    sub-objects; letting Spark infer + carry every possible nested field
    is unnecessary overhead for this project).
    """
    raw = spark.read.option("multiLine", True).json(f"{events_dir}/*.json")
    raw = _with_match_id(raw)

    events = raw.select(
        "match_id",
        F.col("id").alias("event_id"),
        "period",
        "minute",
        "second",
        F.col("type.name").alias("event_type"),
        F.col("team.id").alias("team_id"),
        F.col("team.name").alias("team_name"),
        F.col("player.id").alias("player_id"),
        F.col("player.name").alias("player_name"),
        F.col("position.name").alias("event_position"),
        "location",
        "under_pressure",
        "duration",
        # Pass
        F.col("pass.recipient.id").alias("pass_recipient_id"),
        F.col("pass.length").alias("pass_length"),
        F.col("pass.end_location").alias("pass_end_location"),
        F.col("pass.outcome.name").alias("pass_outcome"),
        F.col("pass.goal_assist").alias("pass_goal_assist"),
        F.col("pass.shot_assist").alias("pass_shot_assist"),
        # Shot
        F.col("shot.outcome.name").alias("shot_outcome"),
        F.col("shot.statsbomb_xg").alias("shot_xg"),
        # Dribble
        F.col("dribble.outcome.name").alias("dribble_outcome"),
        # Duel (includes aerials, tackles)
        F.col("duel.type.name").alias("duel_type"),
        F.col("duel.outcome.name").alias("duel_outcome"),
        # Interception
        F.col("interception.outcome.name").alias("interception_outcome"),
        # Substitution (used for minutes-played fallback / sanity checks)
        F.col("substitution.replacement.id").alias("sub_replacement_id"),
    )
    return events


def load_lineups(spark: SparkSession, lineups_dir: str) -> DataFrame:
    """Load all lineups/*.json files into a flat (match, team, player,
    position-segment) DataFrame. Each player's `positions` array records
    every contiguous stretch of the match they occupied a given position,
    with from/to timestamps in continuous match-clock mm:ss (StatsBomb
    does not reset the clock at half-time boundaries)."""
    raw = spark.read.option("multiLine", True).json(f"{lineups_dir}/*.json")
    raw = _with_match_id(raw)

    teams = raw.select(
        "match_id",
        F.col("team_id"),
        F.col("team_name"),
        F.explode("lineup").alias("player"),
    )

    players = teams.select(
        "match_id",
        "team_id",
        "team_name",
        F.col("player.player_id").alias("player_id"),
        F.col("player.player_name").alias("player_name"),
        F.col("player.jersey_number").alias("jersey_number"),
        F.explode_outer("player.positions").alias("pos_segment"),
    )

    lineup_players = players.select(
        "match_id",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "jersey_number",
        F.col("pos_segment.position").alias("position_name"),
        F.col("pos_segment.from").alias("from_ts"),
        F.col("pos_segment.to").alias("to_ts"),
    )
    return lineup_players


def build_spark_session(app_name: str = "statsbomb-style-clustering") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
