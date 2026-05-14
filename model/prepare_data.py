from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "results.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

CLEAN_MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"
TEAM_STATS_PATH = PROCESSED_DIR / "team_stats.csv"
TEAMS_JSON_PATH = PROCESSED_DIR / "teams.json"
TEAM_STATS_JSON_PATH = PROCESSED_DIR / "team_stats.json"


def get_match_result(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    return "draw"


def get_winner(row):
    if row["result"] == "home_win":
        return row["home_team"]
    elif row["result"] == "away_win":
        return row["away_team"]
    return "Draw"


def build_team_stats(matches):
    home_stats = matches[["home_team", "home_score", "away_score", "result"]].copy()
    home_stats.columns = ["team", "goals_for", "goals_against", "result"]
    home_stats["is_win"] = home_stats["result"] == "home_win"
    home_stats["is_draw"] = home_stats["result"] == "draw"
    home_stats["is_loss"] = home_stats["result"] == "away_win"

    away_stats = matches[["away_team", "away_score", "home_score", "result"]].copy()
    away_stats.columns = ["team", "goals_for", "goals_against", "result"]
    away_stats["is_win"] = away_stats["result"] == "away_win"
    away_stats["is_draw"] = away_stats["result"] == "draw"
    away_stats["is_loss"] = away_stats["result"] == "home_win"

    all_team_stats = pd.concat([home_stats, away_stats], ignore_index=True)

    team_stats = (
        all_team_stats
        .groupby("team")
        .agg(
            matches_played=("team", "count"),
            wins=("is_win", "sum"),
            draws=("is_draw", "sum"),
            losses=("is_loss", "sum"),
            goals_scored=("goals_for", "sum"),
            goals_conceded=("goals_against", "sum"),
            avg_goals_scored=("goals_for", "mean"),
            avg_goals_conceded=("goals_against", "mean"),
        )
        .reset_index()
    )

    team_stats["win_rate"] = team_stats["wins"] / team_stats["matches_played"]
    team_stats["draw_rate"] = team_stats["draws"] / team_stats["matches_played"]
    team_stats["loss_rate"] = team_stats["losses"] / team_stats["matches_played"]
    team_stats["goal_difference"] = team_stats["goals_scored"] - team_stats["goals_conceded"]

    team_stats = team_stats.sort_values(
        by=["matches_played", "win_rate"],
        ascending=[False, False]
    )

    return team_stats


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_DATA_PATH}. "
            "Download results.csv and place it in data/raw/"
        )

    matches = pd.read_csv(RAW_DATA_PATH)

    required_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]

    missing_columns = [col for col in required_columns if col not in matches.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in results.csv: {missing_columns}")

    matches = matches[required_columns].copy()

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])

    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)

    matches["result"] = matches.apply(get_match_result, axis=1)
    matches["winner"] = matches.apply(get_winner, axis=1)
    matches["total_goals"] = matches["home_score"] + matches["away_score"]
    matches["goal_difference"] = matches["home_score"] - matches["away_score"]

    matches = matches.sort_values("date").reset_index(drop=True)

    team_stats = build_team_stats(matches)

    matches.to_csv(CLEAN_MATCHES_PATH, index=False)
    team_stats.to_csv(TEAM_STATS_PATH, index=False)

    teams = sorted(set(matches["home_team"]).union(set(matches["away_team"])))
    pd.Series(teams, name="team").to_json(TEAMS_JSON_PATH, orient="values", indent=2)

    team_stats.to_json(TEAM_STATS_JSON_PATH, orient="records", indent=2)

    print("Data preparation complete.")
    print(f"Clean matches saved to: {CLEAN_MATCHES_PATH}")
    print(f"Team stats saved to: {TEAM_STATS_PATH}")
    print(f"Teams JSON saved to: {TEAMS_JSON_PATH}")
    print(f"Team stats JSON saved to: {TEAM_STATS_JSON_PATH}")
    print()
    print("Sample team stats:")
    print(team_stats.head(10))


if __name__ == "__main__":
    main()