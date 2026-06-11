import csv
import os
from datetime import datetime, UTC

LOG_PATH = "/app/prediction_logs.csv"



def log_prediction(text_length: int, label: str):
    exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow([
                "timestamp",
                "text_length",
                "label"
            ])

        writer.writerow([
            datetime.now(UTC).isoformat(),
            text_length,
            label
        ])