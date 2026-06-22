import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from xgboost import XGBClassifier

PROCESSED = Path("data/processed")

def train():
    df = pd.read_parquet(PROCESSED / "train_features.parquet")

    TARGET = "TARGET"
    X = df.drop(columns=[TARGET, "SK_ID_CURR"], errors="ignore")
    y = df[TARGET]

    # Save feature names for the API to use
    feature_names = X.columns.tolist()
    with open(PROCESSED / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print(f"Saved {len(feature_names)} feature names")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = round(neg / pos, 2)

    mlflow.set_experiment("credit-risk-v1")

    with mlflow.start_run(run_name="xgb-baseline-v2"):
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "scale_pos_weight": spw,
            "random_state": 42,
            "n_jobs": -1
        }
        mlflow.log_params(params)

        model = XGBClassifier(**params, eval_metric="auc", early_stopping_rounds=30)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        ks  = ks_2samp(probs[y_val == 1], probs[y_val == 0]).statistic

        mlflow.log_metrics({"roc_auc": round(auc, 4), "ks_stat": round(ks, 4)})
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="CreditRiskXGB"
        )

        print(f"\nROC-AUC : {auc:.4f}")
        print(f"KS Stat : {ks:.4f}")

if __name__ == "__main__":
    train()