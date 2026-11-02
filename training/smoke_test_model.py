import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


FEATURE_NAMES = (
    "error_rate_5xx_pct",
    "error_rate_delta_30m",
    "p99_latency_ms",
    "cpu_usage_pct",
    "memory_usage_pct",
    "restarts_60m",
)


def make_synthetic_data(sample_count: int = 12_000, seed: int = 7):
    rng = np.random.default_rng(seed)
    latent_risk = np.zeros(sample_count, dtype=np.float32)

    for index in range(1, sample_count):
        latent_risk[index] = np.clip(
            0.985 * latent_risk[index - 1] + rng.normal(0, 0.025),
            0,
            1,
        )

    incident_count = max(30, sample_count // 250)
    incident_starts = rng.choice(
        np.arange(60, sample_count - 60),
        size=incident_count,
        replace=False,
    )
    for start in incident_starts:
        duration = int(rng.integers(15, 55))
        severity = float(rng.uniform(0.55, 1.0))
        end = min(start + duration, sample_count)
        ramp = np.linspace(0.2, severity, end - start, dtype=np.float32)
        latent_risk[start:end] = np.maximum(latent_risk[start:end], ramp)

    error_rate = np.clip(
        0.005 + latent_risk * 0.16 + rng.normal(0, 0.008, sample_count),
        0,
        None,
    )
    p99_latency = np.clip(
        45 + latent_risk * 650 + rng.normal(0, 35, sample_count),
        1,
        None,
    )
    cpu_usage = np.clip(25 + latent_risk * 70 + rng.normal(0, 8, sample_count), 0, 100)
    memory_usage = np.clip(
        30 + latent_risk * 62 + rng.normal(0, 7, sample_count),
        0,
        100,
    )
    restarts = rng.poisson(0.02 + latent_risk * 0.8).astype(np.float32)
    error_rate_delta = np.zeros(sample_count, dtype=np.float32)
    error_rate_delta[6:] = error_rate[6:] - error_rate[:-6]

    features = np.column_stack(
        (
            error_rate,
            error_rate_delta,
            p99_latency,
            cpu_usage,
            memory_usage,
            restarts,
        )
    ).astype(np.float32)

    horizon = 12
    labels = np.zeros(sample_count, dtype=np.uint8)
    for index in range(sample_count - horizon):
        labels[index] = int(np.max(latent_risk[index + 1 : index + horizon + 1]) >= 0.55)

    valid_count = sample_count - horizon
    return features[:valid_count], labels[:valid_count]


def main() -> int:
    features, labels = make_synthetic_data()
    split_index = int(len(labels) * 0.8)
    x_train, x_test = features[:split_index], features[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]

    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    if positive_count == 0 or y_test.sum() == 0 or y_test.sum() == len(y_test):
        raise RuntimeError("Synthetic temporal split does not contain both classes")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.9,
        scale_pos_weight=negative_count / positive_count,
        random_state=7,
        n_jobs=2,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.uint8)
    metrics = {
        "dataset": "synthetic temporal smoke test; not Dynatrace or Pix production data",
        "samples": int(len(labels)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_positive_labels": positive_count,
        "test_positive_labels": int(y_test.sum()),
        "test_prevalence_pct": float(y_test.mean() * 100),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel().tolist(),
    }

    with tempfile.TemporaryDirectory(prefix="xgboost-smoke-test-") as temp_dir:
        model_path = Path(temp_dir) / "synthetic-smoke-test.json"
        model.save_model(model_path)
        reloaded_model = XGBClassifier()
        reloaded_model.load_model(model_path)
        reloaded_probabilities = reloaded_model.predict_proba(x_test)[:, 1]
        if not np.allclose(probabilities, reloaded_probabilities):
            raise RuntimeError("Reloaded model predictions differ from trained model")

    metrics["model_save_reload"] = "passed"
    metrics["feature_importance"] = {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, model.feature_importances_)
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
