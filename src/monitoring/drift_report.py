import pandas as pd
import numpy as np
from pathlib import Path
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

def generate_drift_report():
    df = pd.read_parquet(PROCESSED / "train_features.parquet")

    reference = df.iloc[:50000].drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    production = df.iloc[50000:60000].drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore").copy()

    # Simulate drift
    production["EXT_SCORE_MEAN"] = production["EXT_SCORE_MEAN"] * 0.85
    production["DEBT_INCOME_RATIO"] = production["DEBT_INCOME_RATIO"] * 1.2

    # Use numeric columns only to avoid issues
    num_cols = reference.select_dtypes(include=[np.number]).columns.tolist()[:30]
    reference = reference[num_cols]
    production = production[num_cols]

    definition = DataDefinition()
    ref_dataset = Dataset.from_pandas(reference, data_definition=definition)
    cur_dataset = Dataset.from_pandas(production, data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_dataset, current_data=cur_dataset)
    result.save_html(str(REPORTS / "drift_report.html"))
    print("Drift report saved to reports/drift_report.html")
    print("Open it in your browser to view the full report")

if __name__ == "__main__":
    generate_drift_report()