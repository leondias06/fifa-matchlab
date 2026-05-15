from pathlib import Path
import json
import shutil
import joblib
import pandas as pd
import pycountry


ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "model" / "artifacts"
WEB_DATA_DIR = ROOT_DIR / "web" / "public" / "data"

MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"
APP_TEAMS_PATH = PROCESSED_DIR / "app_teams.json"
PREDICTIONS_PATH = PROCESSED_DIR / "model_predictions.json"
TEAM_FEATURES_PATH = PROCESSED_DIR / "latest_team_features.json"
MODEL_METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"
MODEL_PATH = ARTIFACTS_DIR / "match_outcome_model.joblib"

H2H_PATH = WEB_DATA_DIR / "head_to_head.json"
RANKINGS_PATH = WEB_DATA_DIR / "rankings.json"
MODEL_INSIGHTS_PATH = WEB_DATA_DIR / "model_insights.json"
TEAM_METADATA_PATH = WEB_DATA_DIR / "team_metadata.json"


SPECIAL_TEAM_METADATA = {
    "England": {"code": "GB-ENG", "flag": "🏴"},
    "Scotland": {"code": "GB-SCT", "flag": "🏴"},
    "Wales": {"code": "GB-WLS", "flag": "🏴"},
    "Northern Ireland": {"code": "GB-NIR", "flag": "🇬🇧"},
    "Republic of Ireland": {"code": "IE", "flag": "🇮🇪"},
    "United States": {"code": "US", "flag": "🇺🇸"},
    "China PR": {"code": "CN", "flag": "🇨🇳"},
    "Chinese Taipei": {"code": "TW", "flag": "🇹🇼"},
    "Hong Kong": {"code": "HK", "flag": "🇭🇰"},
    "Macau": {"code": "MO", "flag": "🇲🇴"},
    "Kosovo": {"code": "XK", "flag": "🇽🇰"},
    "Palestine": {"code": "PS", "flag": "🇵🇸"},
    "Faroe Islands": {"code": "FO", "flag": "🇫🇴"},
    "Czech Republic": {"code": "CZ", "flag": "🇨🇿"},
    "DR Congo": {"code": "CD", "flag": "🇨🇩"},
    "Congo": {"code": "CG", "flag": "🇨🇬"},
    "Ivory Coast": {"code": "CI", "flag": "🇨🇮"},
    "Cape Verde": {"code": "CV", "flag": "🇨🇻"},
    "South Korea": {"code": "KR", "flag": "🇰🇷"},
    "North Korea": {"code": "KP", "flag": "🇰🇵"},
    "Iran": {"code": "IR", "flag": "🇮🇷"},
    "Russia": {"code": "RU", "flag": "🇷🇺"},
    "Syria": {"code": "SY", "flag": "🇸🇾"},
    "Vietnam": {"code": "VN", "flag": "🇻🇳"},
    "Venezuela": {"code": "VE", "flag": "🇻🇪"},
    "Bolivia": {"code": "BO", "flag": "🇧🇴"},
    "Moldova": {"code": "MD", "flag": "🇲🇩"},
    "Tanzania": {"code": "TZ", "flag": "🇹🇿"},
    "Turkey": {"code": "TR", "flag": "🇹🇷"},
    "Brunei": {"code": "BN", "flag": "🇧🇳"},
    "Laos": {"code": "LA", "flag": "🇱🇦"},
    "East Timor": {"code": "TL", "flag": "🇹🇱"},
    "Kyrgyzstan": {"code": "KG", "flag": "🇰🇬"},
    "Eswatini": {"code": "SZ", "flag": "🇸🇿"},
}


def country_code_to_flag(code):
    if not code or len(code) != 2:
        return "⚽"

    return "".join(chr(127397 + ord(character.upper())) for character in code)


def get_team_metadata(team):
    if team in SPECIAL_TEAM_METADATA:
        metadata = SPECIAL_TEAM_METADATA[team]
        return {
            "team": team,
            "display_name": team,
            "code": metadata["code"],
            "flag": metadata["flag"],
        }

    try:
        country = pycountry.countries.lookup(team)
        code = country.alpha_2

        return {
            "team": team,
            "display_name": team,
            "code": code,
            "flag": country_code_to_flag(code),
        }

    except LookupError:
        return {
            "team": team,
            "display_name": team,
            "code": None,
            "flag": "⚽",
        }


def build_team_metadata(app_teams):
    return {team: get_team_metadata(team) for team in app_teams}


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def get_winner_from_row(row):
    if row["home_score"] > row["away_score"]:
        return row["home_team"]
    if row["home_score"] < row["away_score"]:
        return row["away_team"]
    return "Draw"


def build_head_to_head(matches, app_teams):
    app_team_set = set(app_teams)
    h2h = {}

    filtered = matches[
        matches["home_team"].isin(app_team_set) & matches["away_team"].isin(app_team_set)
    ].copy()

    filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
    filtered = filtered.dropna(subset=["date"])

    for team_a in app_teams:
        for team_b in app_teams:
            if team_a == team_b:
                continue

            pair_matches = filtered[
                (
                    (filtered["home_team"] == team_a)
                    & (filtered["away_team"] == team_b)
                )
                |
                (
                    (filtered["home_team"] == team_b)
                    & (filtered["away_team"] == team_a)
                )
            ].copy()

            pair_matches = pair_matches.sort_values("date", ascending=False)

            team_a_wins = 0
            team_b_wins = 0
            draws = 0

            recent_matches = []

            for _, row in pair_matches.iterrows():
                winner = get_winner_from_row(row)

                if winner == team_a:
                    team_a_wins += 1
                elif winner == team_b:
                    team_b_wins += 1
                else:
                    draws += 1

                if len(recent_matches) < 8:
                    recent_matches.append(
                        {
                            "date": str(row["date"].date()),
                            "home_team": row["home_team"],
                            "away_team": row["away_team"],
                            "home_score": int(row["home_score"]),
                            "away_score": int(row["away_score"]),
                            "tournament": row["tournament"],
                            "winner": winner,
                        }
                    )

            total_matches = int(len(pair_matches))
            total_goals = int(
                pair_matches["home_score"].sum() + pair_matches["away_score"].sum()
            ) if total_matches else 0

            key = f"{team_a}__{team_b}"

            h2h[key] = {
                "team_a": team_a,
                "team_b": team_b,
                "total_matches": total_matches,
                "team_a_wins": int(team_a_wins),
                "team_b_wins": int(team_b_wins),
                "draws": int(draws),
                "average_goals": round(total_goals / total_matches, 2) if total_matches else 0,
                "recent_matches": recent_matches,
            }

    return h2h


def build_rankings(team_features):
    df = pd.DataFrame(team_features)

    min_matches = 30
    if "team_matches_played" in df.columns:
        df = df[df["team_matches_played"] >= min_matches].copy()

    df["overall_power_score"] = (
        df["team_win_rate"] * 45
        + df["team_recent_form_points"] * 15
        + df["team_goal_difference_per_match"] * 20
        + df["team_avg_goals_scored"] * 10
        - df["team_avg_goals_conceded"] * 10
    )

    def top_records(column, ascending=False, limit=10):
        return (
            df.sort_values(column, ascending=ascending)
            .head(limit)
            .round(4)
            .to_dict(orient="records")
        )

    return {
        "overall": top_records("overall_power_score", ascending=False),
        "attack": top_records("team_avg_goals_scored", ascending=False),
        "defence": top_records("team_avg_goals_conceded", ascending=True),
        "recent_form": top_records("team_recent_form_points", ascending=False),
        "win_rate": top_records("team_win_rate", ascending=False),
        "goal_difference": top_records("team_goal_difference_per_match", ascending=False),
    }


def build_model_insights():
    metrics = load_json(MODEL_METRICS_PATH)

    insights = {
        "accuracy": metrics.get("accuracy"),
        "train_rows": metrics.get("train_rows"),
        "test_rows": metrics.get("test_rows"),
        "classes": metrics.get("classes"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "classification_report": metrics.get("classification_report"),
        "top_features": [],
    }

    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]

        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_

        rows = []

        for index, feature_name in enumerate(feature_names):
            clean_name = (
                feature_name
                .replace("numeric__", "")
                .replace("categorical__", "")
                .replace("_", " ")
            )

            importance = float(abs(coefficients[:, index]).mean())

            if feature_name.startswith("numeric__"):
                rows.append(
                    {
                        "feature": clean_name,
                        "importance": round(importance, 4),
                    }
                )

        rows = sorted(rows, key=lambda item: item["importance"], reverse=True)
        insights["top_features"] = rows[:12]

    return insights


def main():
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        MATCHES_PATH,
        APP_TEAMS_PATH,
        PREDICTIONS_PATH,
        TEAM_FEATURES_PATH,
        MODEL_METRICS_PATH,
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    app_teams = load_json(APP_TEAMS_PATH)
    predictions = load_json(PREDICTIONS_PATH)
    team_features = load_json(TEAM_FEATURES_PATH)
    matches = pd.read_csv(MATCHES_PATH)

    shutil.copy(APP_TEAMS_PATH, WEB_DATA_DIR / "app_teams.json")
    shutil.copy(PREDICTIONS_PATH, WEB_DATA_DIR / "model_predictions.json")
    shutil.copy(TEAM_FEATURES_PATH, WEB_DATA_DIR / "latest_team_features.json")
    shutil.copy(MODEL_METRICS_PATH, WEB_DATA_DIR / "model_metrics.json")

    head_to_head = build_head_to_head(matches, app_teams)
    rankings = build_rankings(team_features)
    model_insights = build_model_insights()
    team_metadata = build_team_metadata(app_teams)

    save_json(H2H_PATH, head_to_head)
    save_json(RANKINGS_PATH, rankings)
    save_json(MODEL_INSIGHTS_PATH, model_insights)
    save_json(TEAM_METADATA_PATH, team_metadata)

    print("App data export complete.")
    print(f"Copied app teams: {WEB_DATA_DIR / 'app_teams.json'}")
    print(f"Copied predictions: {WEB_DATA_DIR / 'model_predictions.json'}")
    print(f"Copied team features: {WEB_DATA_DIR / 'latest_team_features.json'}")
    print(f"Copied model metrics: {WEB_DATA_DIR / 'model_metrics.json'}")
    print(f"Created head-to-head data: {H2H_PATH}")
    print(f"Created rankings data: {RANKINGS_PATH}")
    print(f"Created model insights data: {MODEL_INSIGHTS_PATH}")

    print(f"Created team metadata: {TEAM_METADATA_PATH}")


if __name__ == "__main__":
    main()