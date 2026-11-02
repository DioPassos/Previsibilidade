import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from feature_engineering.build_dataset import FEATURE_COLUMNS
from training.predict_model import predict
from training.train_model import PURGE_MINUTES, train


class TrainingPipelineTests(unittest.TestCase):
    def test_train_save_and_predict_round_trip(self):
        rng = np.random.default_rng(4)
        row_count = 600
        labels = (np.arange(row_count) // 50 % 2).astype(int)
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    pd.Timestamp.now(tz="UTC").floor("5min")
                    - pd.Timedelta(minutes=(row_count - 1) * 5),
                    periods=row_count,
                    freq="5min",
                ),
                "service": "test-service",
                "y": labels,
                "slo_current": 0.9996,
            }
        )
        for column in FEATURE_COLUMNS:
            if column not in frame:
                frame[column] = labels.astype(float) + rng.normal(0, 0.05, row_count)
        frame["slo_error_budget_remaining_pct"] = 80 - labels * 30

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.csv"
            model_path = root / "model.json"
            metadata_path = root / "metadata.json"
            predictions_path = root / "predictions.json"
            test_scores_path = root / "test_scores.csv"
            frame.to_csv(dataset_path, index=False)
            with patch.dict(os.environ, {"XGB_N_ESTIMATORS": "10"}):
                metadata = train(
                    dataset_path,
                    model_path,
                    metadata_path,
                    test_scores_path,
                )
            results = predict(
                dataset_path,
                model_path,
                metadata_path,
                predictions_path,
            )

            self.assertTrue(model_path.is_file())
            self.assertEqual(metadata["metrics"]["test_rows"] > 0, True)
            self.assertEqual(metadata["horizon_minutes"], 30)
            self.assertEqual(metadata["slo_type"], "prototype_proxy")
            self.assertLess(
                pd.Timestamp(metadata["train_period_end"])
                + pd.Timedelta(minutes=PURGE_MINUTES),
                pd.Timestamp(metadata["validation_period_start"]),
            )
            self.assertLess(
                pd.Timestamp(metadata["validation_period_end"])
                + pd.Timedelta(minutes=PURGE_MINUTES),
                pd.Timestamp(metadata["test_period_start"]),
            )
            self.assertEqual(metadata["purge_minutes"], 120)
            self.assertEqual(metadata["threshold_selection"]["method"], "maximum F1 on validation partition")
            self.assertIn("0.15", metadata["metrics"]["threshold_comparison"])
            self.assertIn("0.2", metadata["metrics"]["threshold_comparison"])
            self.assertIn("0.5", metadata["metrics"]["threshold_comparison"])
            self.assertTrue(metadata["feature_importance_gain"])
            self.assertTrue(test_scores_path.is_file())
            self.assertEqual(metadata["test_scores_path"], str(test_scores_path))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["service"], "test-service")
            self.assertEqual(results[0]["horizon_minutes"], 30)
            self.assertTrue(predictions_path.is_file())
if __name__ == "__main__":
    unittest.main()
