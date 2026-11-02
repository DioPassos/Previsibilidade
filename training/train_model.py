import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from feature_engineering.build_dataset import (
    FEATURE_COLUMNS,
    HORIZON_MINUTES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "trained"
MODEL_PATH = MODEL_DIR / "slo_xgboost.json"
METADATA_PATH = MODEL_DIR / "slo_xgboost.metadata.json"
TEST_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "test_scores.csv"
PURGE_MINUTES = 120
ALERT_EPISODE_GAP_MINUTES = 30


def _partition_by_time(dataset: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    timestamps = pd.Index(dataset["timestamp"].drop_duplicates().sort_values())
    if len(timestamps) < 10:
        raise ValueError("Not enough distinct timestamps for temporal train/validation/test splits")
    validation_start = timestamps[int(len(timestamps) * 0.6)]
    test_start = timestamps[int(len(timestamps) * 0.8)]
    purge = pd.Timedelta(minutes=PURGE_MINUTES)

    train_mask = dataset["timestamp"] < validation_start - purge
    validation_mask = (
        (dataset["timestamp"] >= validation_start)
        & (dataset["timestamp"] < test_start - purge)
    )
    test_mask = dataset["timestamp"] >= test_start
    periods = {
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "purge_minutes": PURGE_MINUTES,
    }
    return train_mask, validation_mask, test_mask, periods


def _confusion_metrics(
    labels: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix_tn_fp_fn_tp": [int(value) for value in matrix],
    }


def _episode_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    evaluation = frame[["timestamp", "service", "y", "score"]].copy()
    evaluation["y"] = evaluation["y"].astype(int)
    evaluation["alert"] = evaluation["score"] >= threshold
    positive_episodes = []
    alert_episodes = []

    for service, group in evaluation.groupby("service", sort=True):
        group = group.sort_values("timestamp")
        for column, output, max_gap in (
            ("y", positive_episodes, pd.Timedelta(minutes=5)),
            ("alert", alert_episodes, pd.Timedelta(minutes=ALERT_EPISODE_GAP_MINUTES)),
        ):
            active = group[group[column].astype(bool)]
            if active.empty:
                continue
            start = previous = active.iloc[0]["timestamp"]
            for timestamp in active["timestamp"].iloc[1:]:
                if timestamp - previous > max_gap:
                    output.append(
                        {"service": service, "start": start, "end": previous}
                    )
                    start = timestamp
                previous = timestamp
            output.append({"service": service, "start": start, "end": previous})

    detected = 0
    for episode in positive_episodes:
        if any(
            alert["service"] == episode["service"]
            and alert["start"] <= episode["end"]
            and alert["end"] >= episode["start"]
            for alert in alert_episodes
        ):
            detected += 1
    false_alerts = sum(
        not any(
            episode["service"] == alert["service"]
            and episode["start"] <= alert["end"]
            and episode["end"] >= alert["start"]
            for episode in positive_episodes
        )
        for alert in alert_episodes
    )
    span_days = max(
        (evaluation["timestamp"].max() - evaluation["timestamp"].min()).total_seconds()
        / 86400,
        1 / 24,
    )
    service_count = max(evaluation["service"].nunique(), 1)
    return {
        "positive_label_episodes": len(positive_episodes),
        "detected_positive_label_episodes": detected,
        "missed_positive_label_episodes": len(positive_episodes) - detected,
        "alert_episodes": len(alert_episodes),
        "false_alert_episodes": false_alerts,
        "false_alert_episodes_per_service_day": float(
            false_alerts / (span_days * service_count)
        ),
        "episode_definition": (
            "Adjacent positive labels per service with gaps no greater than 5m; "
            "alert rows per service merged when gaps are no greater than 30m."
        ),
    }


def train(
    dataset_path: Path,
    model_path: Path,
    metadata_path: Path,
    test_scores_path: Path = TEST_SCORES_PATH,
) -> dict:
    dataset = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    required = {"timestamp", "service", "y", *FEATURE_COLUMNS}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    trainable = dataset.dropna(subset=[*FEATURE_COLUMNS, "y"]).copy()
    trainable["y"] = trainable["y"].astype(int)
    if trainable.empty:
        raise ValueError("No trainable rows: build a complete historical dataset first")
    if trainable["y"].nunique() < 2:
        raise ValueError(
            "Training data contains only one class; collect more history/events before training"
        )

    trainable = trainable.sort_values(["timestamp", "service"]).reset_index(drop=True)
    train_rows, validation_rows, test_rows, split_metadata = _partition_by_time(trainable)
    partitions = {
        "train": trainable.loc[train_rows],
        "validation": trainable.loc[validation_rows],
        "test": trainable.loc[test_rows],
    }
    for name in ("train", "test"):
        partition = partitions[name]
        if partition.empty or partition["y"].nunique() < 2:
            raise ValueError(
                f"Chronological {name} partition needs both label classes; "
                "collect more representative history"
            )
    if partitions["validation"].empty:
        raise ValueError("Chronological validation partition is empty; collect more history")

    train_frame = partitions["train"]
    validation_frame = partitions["validation"]
    test_frame = partitions["test"]
    positive_count = int(train_frame["y"].sum())
    negative_count = int(len(train_frame) - positive_count)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=int(os.getenv("XGB_N_ESTIMATORS", "300")),
        max_depth=int(os.getenv("XGB_MAX_DEPTH", "4")),
        learning_rate=float(os.getenv("XGB_LEARNING_RATE", "0.05")),
        subsample=0.85,
        colsample_bytree=0.9,
        scale_pos_weight=negative_count / positive_count,
        random_state=42,
        n_jobs=int(os.getenv("XGB_N_JOBS", "2")),
    )
    model.fit(train_frame[FEATURE_COLUMNS], train_frame["y"])

    validation_scores = model.predict_proba(validation_frame[FEATURE_COLUMNS])[:, 1]
    if validation_frame["y"].nunique() == 2:
        validation_precision, validation_recall, validation_thresholds = precision_recall_curve(
            validation_frame["y"],
            validation_scores,
        )
        if len(validation_thresholds):
            validation_f1 = (
                2
                * validation_precision[:-1]
                * validation_recall[:-1]
                / np.maximum(validation_precision[:-1] + validation_recall[:-1], 1e-12)
            )
            selected_threshold = float(validation_thresholds[int(np.argmax(validation_f1))])
            selected_validation_f1 = float(np.max(validation_f1))
            threshold_method = "maximum F1 on validation partition"
        else:
            selected_threshold = 0.5
            selected_validation_f1 = None
            threshold_method = "default 0.5; validation threshold unavailable"
    else:
        selected_threshold = 0.5
        selected_validation_f1 = None
        threshold_method = (
            "default 0.5; validation partition has one label class, "
            "so threshold tuning is not statistically supported"
        )
    configured_threshold = float(os.getenv("PREDICTION_THRESHOLD", selected_threshold))

    test_scores = model.predict_proba(test_frame[FEATURE_COLUMNS])[:, 1]
    scored_test = test_frame[["timestamp", "service", "y"]].copy()
    scored_test["score"] = test_scores
    threshold_metrics = {
        str(threshold): _confusion_metrics(test_frame["y"], test_scores, threshold)
        for threshold in sorted({0.15, 0.20, 0.50, selected_threshold, configured_threshold})
    }
    selected_test_metrics = _confusion_metrics(
        test_frame["y"],
        test_scores,
        configured_threshold,
    )
    selected_test_metrics.update(_episode_metrics(scored_test, configured_threshold))

    test_positive_count = int(test_frame["y"].sum())
    test_metrics = {
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "train_positive_labels": positive_count,
        "validation_positive_labels": int(validation_frame["y"].sum()),
        "test_positive_labels": test_positive_count,
        "test_positive_rate": float(test_frame["y"].mean()),
        "threshold": configured_threshold,
        **selected_test_metrics,
        "pr_auc": float(average_precision_score(test_frame["y"], test_scores)),
        "roc_auc": float(roc_auc_score(test_frame["y"], test_scores)),
        "threshold_comparison": threshold_metrics,
    }
    test_precision, test_recall, test_thresholds = precision_recall_curve(
        test_frame["y"], test_scores
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    test_scores_path.parent.mkdir(parents=True, exist_ok=True)
    scored_test.to_csv(test_scores_path, index=False)
    try:
        test_scores_location = str(test_scores_path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        test_scores_location = str(test_scores_path)
    metadata = {
        "model_type": "xgboost.XGBClassifier",
        "feature_columns": FEATURE_COLUMNS,
        "label": (
            "1h proxy SLO breaches target within next 30m, "
            "given current proxy SLO is healthy"
        ),
        "slo_type": "prototype_proxy",
        "slo_target": 0.9995,
        "interval_minutes": 5,
        "horizon_minutes": HORIZON_MINUTES,
        "rolling_window_minutes": 60,
        "purge_minutes": PURGE_MINUTES,
        "threshold": configured_threshold,
        "split_method": "chronological 60% train / 20% validation / 20% test by timestamp",
        "threshold_selection": {
            "method": threshold_method,
            "validation_threshold": selected_threshold,
            "validation_f1": selected_validation_f1,
            "validation_rows": int(len(validation_frame)),
            "validation_positive_labels": int(validation_frame["y"].sum()),
            "validation_label_classes": int(validation_frame["y"].nunique()),
            "prediction_threshold": configured_threshold,
        },
        "train_period_start": train_frame["timestamp"].min().isoformat(),
        "train_period_end": train_frame["timestamp"].max().isoformat(),
        "validation_period_start": validation_frame["timestamp"].min().isoformat(),
        "validation_period_end": validation_frame["timestamp"].max().isoformat(),
        "test_period_start": test_frame["timestamp"].min().isoformat(),
        "test_period_end": test_frame["timestamp"].max().isoformat(),
        "split_boundaries": split_metadata,
        "test_precision_recall_curve": {
            "precision": [float(value) for value in test_precision],
            "recall": [float(value) for value in test_recall],
            "thresholds": [float(value) for value in test_thresholds],
        },
        "feature_importance_gain": {
            name: float(value)
            for name, value in zip(FEATURE_COLUMNS, model.feature_importances_)
        },
        "test_scores_path": test_scores_location,
        "metrics": test_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset_path = Path(os.getenv("DATASET_PATH", DATASET_PATH))
    model_path = Path(os.getenv("MODEL_PATH", MODEL_PATH))
    metadata_path = Path(os.getenv("MODEL_METADATA_PATH", METADATA_PATH))
    scores_path = Path(os.getenv("TEST_SCORES_PATH", TEST_SCORES_PATH))
    metadata = train(dataset_path, model_path, metadata_path, scores_path)
    metrics = metadata["metrics"]
    logging.info("Trained model saved to %s", model_path)
    logging.info(
        "Temporal test: precision=%.3f recall=%.3f PR-AUC=%.3f threshold=%.3f",
        metrics["precision"],
        metrics["recall"],
        metrics["pr_auc"],
        metrics["threshold"],
    )
    logging.info("Individual test scores saved to %s", scores_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
