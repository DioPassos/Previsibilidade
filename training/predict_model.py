import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests
from xgboost import XGBClassifier

from feature_engineering.build_dataset import FEATURE_COLUMNS, SLO_TARGET
from training.train_model import METADATA_PATH, MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "latest_predictions.json"
ALERT_STATE_PATH = PROJECT_ROOT / "data" / "processed" / "alert_state.json"


def predict(
    dataset_path: Path,
    model_path: Path,
    metadata_path: Path,
    predictions_path: Path,
    webhook_url: str = "",
    alert_state_path: Path = ALERT_STATE_PATH,
) -> list[dict]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model does not exist: {model_path}")
    dataset = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    metadata = json.loads(metadata_path.read_text())
    expected_features = metadata.get("feature_columns")
    if expected_features != FEATURE_COLUMNS:
        raise ValueError("Model feature schema does not match the current dataset builder")
    missing = set(FEATURE_COLUMNS).difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing model features: {', '.join(sorted(missing))}")

    candidates = dataset.dropna(subset=FEATURE_COLUMNS).copy()
    if candidates.empty:
        raise ValueError("No complete feature rows are available for prediction")
    candidates = candidates.sort_values(["timestamp", "service"])
    latest = candidates.groupby("service", sort=False, as_index=False).tail(1)
    max_age_minutes = int(os.getenv("MAX_FEATURE_AGE_MINUTES", "15"))
    latest_timestamp = latest["timestamp"].max()
    age_minutes = (
        pd.Timestamp.now(tz="UTC") - latest_timestamp
    ).total_seconds() / 60
    if age_minutes > max_age_minutes:
        raise ValueError(
            f"Latest feature data is {age_minutes:.1f} minutes old "
            f"(maximum allowed: {max_age_minutes})"
        )

    model = XGBClassifier()
    model.load_model(model_path)
    threshold = float(os.getenv("PREDICTION_THRESHOLD", metadata["threshold"]))
    horizon_minutes = int(metadata["horizon_minutes"])
    healthy_rows = latest["slo_current"] >= SLO_TARGET
    probabilities = [None] * len(latest)
    if healthy_rows.any():
        healthy_probabilities = model.predict_proba(
            latest.loc[healthy_rows, FEATURE_COLUMNS]
        )[:, 1]
        healthy_positions = [
            index for index, is_healthy in enumerate(healthy_rows.to_numpy()) if is_healthy
        ]
        for position, probability in zip(healthy_positions, healthy_probabilities):
            probabilities[position] = float(probability)

    predictions = []
    for (_, row), probability in zip(latest.iterrows(), probabilities):
        already_breached = float(row["slo_current"]) < SLO_TARGET
        prediction = {
            "service": str(row["service"]),
            "timestamp": row["timestamp"].isoformat(),
            "horizon_minutes": horizon_minutes,
            "probability_of_future_slo_breach": probability,
            "threshold": threshold,
            "status": (
                "already_breached"
                if already_breached
                else "risk" if probability is not None and probability >= threshold else "healthy"
            ),
            "slo_current": float(row["slo_current"]),
        }
        predictions.append(prediction)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "predictions": predictions,
            },
            indent=2,
        )
        + "\n"
    )

    previous_states = {}
    if alert_state_path.exists():
        previous_states = json.loads(alert_state_path.read_text())
    next_states = {}
    if webhook_url:
        if urlsplit(webhook_url).scheme != "https":
            raise ValueError("ALERT_WEBHOOK_URL must use HTTPS")
        for prediction in predictions:
            service = prediction["service"]
            is_new_risk = (
                prediction["status"] == "risk"
                and previous_states.get(service) != "risk"
            )
            if is_new_risk:
                response = requests.post(
                    webhook_url,
                    json={
                        "text": (
                            f"SLO risk for {prediction['service']}: "
                            f"{prediction['probability_of_future_slo_breach']:.1%} "
                            f"probability of breach within {horizon_minutes} minutes."
                        ),
                        "prediction": prediction,
                    },
                    timeout=15,
                )
                response.raise_for_status()
            next_states[service] = prediction["status"]
        alert_state_path.parent.mkdir(parents=True, exist_ok=True)
        alert_state_path.write_text(json.dumps(next_states, indent=2) + "\n")
    return predictions


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset_path = Path(os.getenv("DATASET_PATH", DATASET_PATH))
    model_path = Path(os.getenv("MODEL_PATH", MODEL_PATH))
    metadata_path = Path(os.getenv("MODEL_METADATA_PATH", METADATA_PATH))
    predictions_path = Path(os.getenv("PREDICTIONS_PATH", PREDICTIONS_PATH))
    alert_state_path = Path(os.getenv("ALERT_STATE_PATH", ALERT_STATE_PATH))
    predictions = predict(
        dataset_path,
        model_path,
        metadata_path,
        predictions_path,
        webhook_url=os.getenv("ALERT_WEBHOOK_URL", "").strip(),
        alert_state_path=alert_state_path,
    )
    for prediction in predictions:
        logging.info(
            "%s: %s risk=%s",
            prediction["service"],
            prediction["status"],
            (
                f"{prediction['probability_of_future_slo_breach']:.3f}"
                if prediction["probability_of_future_slo_breach"] is not None
                else "n/a"
            ),
        )
    logging.info("Predictions saved to %s", predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
