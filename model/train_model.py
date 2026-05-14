from pathlib import Path
from collections import defaultdict, deque
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]

MATCHES_PATH = ROOT_DIR / "data" / "processed" / "matches_clean.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

ARTIFACTS_DIR = ROOT_DIR / "model" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "match_outcome_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"

LATEST_TEAM_FEATURES_PATH = PROCESSED_DIR / "latest_team_features.json"
APP_TEAMS_PATH = PROCESSED_DIR / "app_teams.json"
MODEL_PREDICTIONS_PATH = PROCESSED_DIR / "model_predictions.json"

PREDICTION_TEAM_LIMIT = 100


def empty_team_state():
    return {
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "recent_points": deque(maxlen=5),
    }


def safe_divide(a, b):
    return a / b if b else 0


def average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0


def simplify_tournament(tournament):
    tournament = str(tournament)

    if tournament == "Friendly":
        return "Friendly"

    if tournament == "FIFA World Cup":
        return "FIFA World Cup"

    lowered = tournament.lower()

    if "world cup" in lowered and "qualification" in lowered:
        return "World Cup qualification"

    if "uefa euro" in lowered:
        return "UEFA Euro"

    if "copa am" in lowered:
        return "Copa America"

    if "african cup" in lowered:
        return "African Cup of Nations"

    if "asian cup" in lowered:
        return "AFC Asian Cup"

    if "gold cup" in lowered:
        return "CONCACAF Gold Cup"

    return "Other"


def team_features(prefix, state):
    matches = state["matches"]

    goals_for = state["goals_for"]
    goals_against = state["goals_against"]

    return {
        f"{prefix}_matches_played": matches,
        f"{prefix}_win_rate": safe_divide(state["wins"], matches),
        f"{prefix}_draw_rate": safe_divide(state["draws"], matches),
        f"{prefix}_loss_rate": safe_divide(state["losses"], matches),
        f"{prefix}_avg_goals_scored": safe_divide(goals_for, matches),
        f"{prefix}_avg_goals_conceded": safe_divide(goals_against, matches),
        f"{prefix}_goal_difference_per_match": safe_divide(goals_for - goals_against, matches),
        f"{prefix}_recent_form_points": average(state["recent_points"]),
    }


def build_feature_row(match, team_states):
    home_team = match["home_team"]
    away_team = match["away_team"]

    home_state = team_states[home_team]
    away_state = team_states[away_team]

    row = {
        "date": match["date"],
        "home_team": home_team,
        "away_team": away_team,
        "match_type": simplify_tournament(match["tournament"]),
        "neutral": int(bool(match["neutral"])),
        "home_advantage": 0 if bool(match["neutral"]) else 1,
        "result": match["result"],
    }

    row.update(team_features("home", home_state))
    row.update(team_features("away", away_state))

    row["win_rate_difference"] = row["home_win_rate"] - row["away_win_rate"]
    row["recent_form_difference"] = row["home_recent_form_points"] - row["away_recent_form_points"]
    row["attack_difference"] = row["home_avg_goals_scored"] - row["away_avg_goals_scored"]
    row["defence_difference"] = row["home_avg_goals_conceded"] - row["away_avg_goals_conceded"]
    row["goal_difference_gap"] = row["home_goal_difference_per_match"] - row["away_goal_difference_per_match"]
    row["experience_difference"] = row["home_matches_played"] - row["away_matches_played"]

    return row


def update_team_state(team_states, match):
    home_team = match["home_team"]
    away_team = match["away_team"]
    home_score = int(match["home_score"])
    away_score = int(match["away_score"])

    if home_score > away_score:
        home_points, away_points = 3, 0
        home_result, away_result = "win", "loss"
    elif home_score < away_score:
        home_points, away_points = 0, 3
        home_result, away_result = "loss", "win"
    else:
        home_points, away_points = 1, 1
        home_result, away_result = "draw", "draw"

    home_state = team_states[home_team]
    away_state = team_states[away_team]

    home_state["matches"] += 1
    home_state["goals_for"] += home_score
    home_state["goals_against"] += away_score
    home_state["recent_points"].append(home_points)

    away_state["matches"] += 1
    away_state["goals_for"] += away_score
    away_state["goals_against"] += home_score
    away_state["recent_points"].append(away_points)

    if home_result == "win":
        home_state["wins"] += 1
    elif home_result == "draw":
        home_state["draws"] += 1
    else:
        home_state["losses"] += 1

    if away_result == "win":
        away_state["wins"] += 1
    elif away_result == "draw":
        away_state["draws"] += 1
    else:
        away_state["losses"] += 1


def build_training_dataset(matches):
    team_states = defaultdict(empty_team_state)
    feature_rows = []

    for _, match in matches.iterrows():
        feature_rows.append(build_feature_row(match, team_states))
        update_team_state(team_states, match)

    model_data = pd.DataFrame(feature_rows)
    return model_data, team_states


def serialise_latest_team_features(team_states):
    rows = []

    for team, state in team_states.items():
        row = {"team": team}
        row.update(team_features("team", state))
        row["goals_scored"] = state["goals_for"]
        row["goals_conceded"] = state["goals_against"]
        row["goal_difference"] = state["goals_for"] - state["goals_against"]
        rows.append(row)

    latest = pd.DataFrame(rows)
    latest = latest.sort_values(
        by=["team_matches_played", "team_win_rate"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return latest


def build_prediction_input(home_team, away_team, team_states, match_type="FIFA World Cup", neutral=True):
    fake_match = {
        "date": pd.Timestamp.today(),
        "home_team": home_team,
        "away_team": away_team,
        "tournament": match_type,
        "neutral": neutral,
        "result": "draw",
    }

    row = build_feature_row(fake_match, team_states)
    row.pop("date", None)
    row.pop("result", None)

    return row


def probability_for_match(model, feature_columns, home_team, away_team, team_states):
    row = build_prediction_input(
        home_team=home_team,
        away_team=away_team,
        team_states=team_states,
        match_type="FIFA World Cup",
        neutral=True,
    )

    input_df = pd.DataFrame([row])[feature_columns]

    probabilities = model.predict_proba(input_df)[0]
    classes = model.named_steps["classifier"].classes_
    class_to_probability = dict(zip(classes, probabilities))

    home_win_probability = float(class_to_probability.get("home_win", 0))
    draw_probability = float(class_to_probability.get("draw", 0))
    away_win_probability = float(class_to_probability.get("away_win", 0))

    probabilities_for_result = {
        "home_win": home_win_probability,
        "draw": draw_probability,
        "away_win": away_win_probability,
    }

    predicted_result = max(probabilities_for_result, key=probabilities_for_result.get)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "match_type": "FIFA World Cup",
        "neutral": True,
        "home_win_probability": round(home_win_probability, 4),
        "draw_probability": round(draw_probability, 4),
        "away_win_probability": round(away_win_probability, 4),
        "predicted_result": predicted_result,
        "confidence": round(probabilities_for_result[predicted_result], 4),
    }


def generate_pair_predictions(model, feature_columns, team_states, latest_team_features):
    app_teams = (
        latest_team_features
        .head(PREDICTION_TEAM_LIMIT)["team"]
        .sort_values()
        .tolist()
    )

    predictions = []

    for home_team in app_teams:
        for away_team in app_teams:
            if home_team == away_team:
                continue

            predictions.append(
                probability_for_match(
                    model=model,
                    feature_columns=feature_columns,
                    home_team=home_team,
                    away_team=away_team,
                    team_states=team_states,
                )
            )

    return app_teams, predictions


def main():
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MATCHES_PATH}. Run model/prepare_data.py first."
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(MATCHES_PATH)
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    model_data, team_states = build_training_dataset(matches)

    categorical_features = ["home_team", "away_team", "match_type"]
    numeric_features = [
        "neutral",
        "home_advantage",
        "home_matches_played",
        "home_win_rate",
        "home_draw_rate",
        "home_loss_rate",
        "home_avg_goals_scored",
        "home_avg_goals_conceded",
        "home_goal_difference_per_match",
        "home_recent_form_points",
        "away_matches_played",
        "away_win_rate",
        "away_draw_rate",
        "away_loss_rate",
        "away_avg_goals_scored",
        "away_avg_goals_conceded",
        "away_goal_difference_per_match",
        "away_recent_form_points",
        "win_rate_difference",
        "recent_form_difference",
        "attack_difference",
        "defence_difference",
        "goal_difference_gap",
        "experience_difference",
    ]

    feature_columns = categorical_features + numeric_features

    X = model_data[feature_columns]
    y = model_data["result"]

    split_index = int(len(model_data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": feature_columns,
        "classes": list(model.named_steps["classifier"].classes_),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
        },
        MODEL_PATH,
    )

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    latest_team_features = serialise_latest_team_features(team_states)
    latest_team_features.to_json(LATEST_TEAM_FEATURES_PATH, orient="records", indent=2)

    app_teams, pair_predictions = generate_pair_predictions(
        model=model,
        feature_columns=feature_columns,
        team_states=team_states,
        latest_team_features=latest_team_features,
    )

    with open(APP_TEAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(app_teams, f, indent=2)

    with open(MODEL_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(pair_predictions, f, indent=2)

    print("Model training complete.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Latest team features saved to: {LATEST_TEAM_FEATURES_PATH}")
    print(f"App teams saved to: {APP_TEAMS_PATH}")
    print(f"Pair predictions saved to: {MODEL_PREDICTIONS_PATH}")
    print()
    print("Sample predictions:")
    print(pd.DataFrame(pair_predictions).head(10))


if __name__ == "__main__":
    main()