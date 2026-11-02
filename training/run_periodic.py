import logging
import os
import time

from collector.collect_pix_data import main as collect
from feature_engineering.build_dataset import main as build
from training.predict_model import main as predict


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    interval_minutes = int(os.getenv("RUN_INTERVAL_MINUTES", "15"))
    if interval_minutes < 1:
        raise ValueError("RUN_INTERVAL_MINUTES must be at least 1")

    while True:
        os.environ["DT_LOOKBACK"] = os.getenv("DT_INCREMENTAL_LOOKBACK", "1h")
        if collect() != 0:
            raise RuntimeError("Dynatrace collection failed; stopping periodic runner")
        if build() != 0:
            raise RuntimeError("Dataset generation failed; stopping periodic runner")
        if predict() != 0:
            raise RuntimeError("Prediction failed; stopping periodic runner")
        logging.info("Next collection cycle in %d minutes", interval_minutes)
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    raise SystemExit(main())
