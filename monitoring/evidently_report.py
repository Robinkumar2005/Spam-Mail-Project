import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

df = pd.read_csv("prediction_logs.csv")

reference_data = df.iloc[: len(df) // 2]
current_data = df.iloc[len(df) // 2 :]

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

report.run(
    reference_data=reference_data,
    current_data=current_data
)

report.save_html("spam_drift_report.html")

print("✅ Spam drift report generated")