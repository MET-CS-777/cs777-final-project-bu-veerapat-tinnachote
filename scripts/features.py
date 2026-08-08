"""
features.py

Aggregates raw per-event rows (events_df) plus per-match minutes played
(minutes_df) into one standardized per-90-minutes feature vector per
player, across whatever set of matches is passed in (this is what makes
the train/competition-based test split in main.py trivial: call
build_player_features() separately on the train-competition subset and
the test-competition subset).

Field semantics (duel.type for tackles, aerial_won flags for aerial
duels, pass.shot_assist / pass.goal_assist) follow the official
StatsBomb Open Data Specification and were verified against the raw
sample-match JSON (see the aerial-duel note below and in ingest.py):
https://github.com/statsbomb/open-data/blob/master/doc/StatsBomb%20Open%20Data%20Specification%20v4.0.pdf
Coordinates are already normalized so every team always attacks toward
x=120 (verified against sample match 3857274), so no half-time
direction-flip handling is needed for the progressive-pass definition
below.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# A pass is called "progressive" if it advances the ball at least this
# many pitch-length units (StatsBomb pitch is 120 units long) toward the
# opponent's goal. 10 units =~ 10 meters. Simplified stand-in for more
# elaborate progressive-pass definitions (e.g. FBref's % of remaining
# distance-to-goal reduced).
PROGRESSIVE_PASS_MIN_ADVANCE = 10.0

# x >= this value counts as the "attacking third" of the pitch (pitch
# length is 120 units, so 80 = start of final third).
ATTACKING_THIRD_X = 80.0


def _end_x(col):
    return col.getItem(0)


def _start_x(loc_col):
    return loc_col.getItem(0)


def build_player_match_stats(events_df: DataFrame) -> DataFrame:
    """One row per (match_id, player_id) with raw per-match event counts."""
    e = events_df

    e = e.withColumn("start_x", _start_x(F.col("location")))
    e = e.withColumn("end_x_pass", F.when(F.col("pass_end_location").isNotNull(), _end_x(F.col("pass_end_location"))))

    is_pass = F.col("event_type") == "Pass"
    is_pass_complete = is_pass & F.col("pass_outcome").isNull()
    is_progressive_pass = is_pass_complete & (
        (F.col("end_x_pass") - F.col("start_x")) >= PROGRESSIVE_PASS_MIN_ADVANCE
    )
    is_key_pass = is_pass & (
        (F.col("pass_shot_assist") == True) | (F.col("pass_goal_assist") == True)  # noqa: E712
    )
    is_shot = F.col("event_type") == "Shot"
    is_goal = is_shot & (F.col("shot_outcome") == "Goal")
    is_dribble_success = (F.col("event_type") == "Dribble") & (F.col("dribble_outcome") == "Complete")
    is_dribble_attempt = F.col("event_type") == "Dribble"
    is_tackle = (F.col("event_type") == "Duel") & (F.col("duel_type") == "Tackle")
    is_tackle_won = is_tackle & (F.col("duel_outcome").isin("Won", "Success", "Success In Play", "Success Out"))
    is_interception = F.col("event_type") == "Interception"
    # A won aerial duel is recorded as aerial_won=true on the winning
    # player's follow-up action (pass/shot/clearance/miscontrol), not as
    # a Duel event -- see the note in ingest.py. Cross-checked on sample
    # match 3857255: aerial_won flags exactly balance "Aerial Lost" duels.
    is_aerial_won = (
        (F.col("pass_aerial_won") == True)  # noqa: E712
        | (F.col("shot_aerial_won") == True)  # noqa: E712
        | (F.col("clearance_aerial_won") == True)  # noqa: E712
        | (F.col("miscontrol_aerial_won") == True)  # noqa: E712
    )
    is_pressure = F.col("event_type") == "Pressure"
    is_attacking_third_touch = F.col("start_x") >= ATTACKING_THIRD_X

    # Goalkeeper sub-events (StatsBomb type name has a literal space,
    # "Goal Keeper" -- see ingest.py). "Shot Saved" covers three outcome
    # variants (on target, to the post, saved off target); all three
    # still count as a save for shot-stopping purposes.
    is_gk_shot_faced = F.col("gk_type") == "Shot Faced"
    is_gk_shot_saved = F.col("gk_type").isin(
        "Shot Saved", "Shot Saved to Post", "Shot Saved Off Target"
    )
    is_gk_claim = F.col("gk_type").isin("Collected", "Punch")
    is_gk_sweeper = F.col("gk_type") == "Keeper Sweeper"

    stats = e.filter(F.col("player_id").isNotNull()).groupBy("match_id", "player_id").agg(
        F.count("*").alias("total_events"),
        F.sum(is_pass.cast("int")).alias("passes_attempted"),
        F.sum(is_pass_complete.cast("int")).alias("passes_completed"),
        F.sum(is_progressive_pass.cast("int")).alias("progressive_passes"),
        F.sum(is_key_pass.cast("int")).alias("key_passes"),
        F.sum(is_shot.cast("int")).alias("shots"),
        F.sum(is_goal.cast("int")).alias("goals"),
        F.sum(is_dribble_attempt.cast("int")).alias("dribble_attempts"),
        F.sum(is_dribble_success.cast("int")).alias("dribbles_completed"),
        F.sum(is_tackle_won.cast("int")).alias("tackles_won"),
        F.sum(is_interception.cast("int")).alias("interceptions"),
        F.sum(is_aerial_won.cast("int")).alias("aerial_duels_won"),
        F.sum(is_pressure.cast("int")).alias("pressures"),
        F.sum(is_attacking_third_touch.cast("int")).alias("attacking_third_touches"),
        F.sum(is_gk_shot_faced.cast("int")).alias("gk_shots_faced"),
        F.sum(is_gk_shot_saved.cast("int")).alias("gk_shots_saved"),
        F.sum(is_gk_claim.cast("int")).alias("gk_claims"),
        F.sum(is_gk_sweeper.cast("int")).alias("gk_sweeper_actions"),
    )
    return stats


def build_player_features(events_df: DataFrame, minutes_df: DataFrame,
                           min_minutes: float) -> DataFrame:
    """Aggregate per-match stats + per-match minutes across all matches in
    events_df/minutes_df into one per-90 feature row per player.

    Both events_df and minutes_df should already be filtered to whichever
    set of matches you want included (e.g. TRAIN_COMPETITIONS only)."""
    per_match_stats = build_player_match_stats(events_df)

    per_player_minutes = minutes_df.groupBy("player_id", "player_name").agg(
        F.sum("minutes_played").alias("total_minutes"),
        F.count("*").alias("matches_played"),
        # crude "primary position" across matches: most frequent primary_position
    )

    # majority-vote primary position across matches
    from pyspark.sql.window import Window
    pos_counts = minutes_df.groupBy("player_id", "primary_position").agg(F.count("*").alias("n"))
    w = Window.partitionBy("player_id").orderBy(F.desc("n"))
    primary_position = (
        pos_counts.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .select("player_id", F.col("primary_position").alias("position_label"))
    )

    totals = per_match_stats.groupBy("player_id").agg(
        *[F.sum(c).alias(c) for c in [
            "passes_attempted", "passes_completed", "progressive_passes", "key_passes",
            "shots", "goals", "dribble_attempts", "dribbles_completed", "tackles_won",
            "interceptions", "aerial_duels_won", "pressures", "attacking_third_touches",
            "gk_shots_faced", "gk_shots_saved", "gk_claims", "gk_sweeper_actions",
        ]]
    )

    features = (
        per_player_minutes.join(totals, on="player_id", how="inner")
        .join(primary_position, on="player_id", how="left")
        .filter(F.col("total_minutes") >= min_minutes)
    )

    p90 = lambda c: (F.col(c) / F.col("total_minutes") * 90.0).alias(f"{c}_p90")

    features = features.select(
        "player_id", "player_name", "position_label", "total_minutes", "matches_played",
        p90("passes_attempted"),
        (F.col("passes_completed") / F.when(F.col("passes_attempted") > 0, F.col("passes_attempted"))).alias("pass_completion_pct"),
        p90("progressive_passes"),
        p90("key_passes"),
        p90("shots"),
        (F.col("goals") / F.when(F.col("shots") > 0, F.col("shots"))).alias("shot_conversion_pct"),
        p90("dribbles_completed"),
        (F.col("dribbles_completed") / F.when(F.col("dribble_attempts") > 0, F.col("dribble_attempts"))).alias("dribble_success_pct"),
        p90("tackles_won"),
        p90("interceptions"),
        p90("aerial_duels_won"),
        p90("pressures"),
        p90("attacking_third_touches"),
        p90("gk_shots_faced"),
        (F.col("gk_shots_saved") / F.when(F.col("gk_shots_faced") > 0, F.col("gk_shots_faced"))).alias("gk_save_pct"),
        p90("gk_claims"),
        p90("gk_sweeper_actions"),
    ).na.fill(0.0)

    return features


# The 13 features used for the PRIMARY global (all-positions) clustering
# in main.py. Deliberately does not include the goalkeeper-specific
# columns below -- those are near-zero for every outfield player and
# would just recreate the GK/outfield split rather than add signal.
FEATURE_COLUMNS = [
    "passes_attempted_p90",
    "pass_completion_pct",
    "progressive_passes_p90",
    "key_passes_p90",
    "shots_p90",
    "shot_conversion_pct",
    "dribbles_completed_p90",
    "dribble_success_pct",
    "tackles_won_p90",
    "interceptions_p90",
    "aerial_duels_won_p90",
    "pressures_p90",
    "attacking_third_touches_p90",
]

# Goalkeeper-specific features, used only by the within-position-family
# clustering (scripts/position_families.py) for the GK family. Distinct
# from FEATURE_COLUMNS because these are meaningless (near-zero) for
# outfield players and shouldn't be part of the global model.
GK_FEATURE_COLUMNS = [
    "gk_shots_faced_p90",
    "gk_save_pct",
    "gk_claims_p90",
    "gk_sweeper_actions_p90",
    "passes_attempted_p90",
    "pass_completion_pct",
    "progressive_passes_p90",
]
