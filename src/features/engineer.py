import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path("data/processed")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Credit domain ratios
    df["DEBT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
    df["EXT_SCORE_MEAN"] = df[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].mean(axis=1)

    # DAYS_EMPLOYED artifact — 365243 means unemployed, not employed 1000 years
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["DAYS_EMPLOYED_PCT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    # Encode categoricals
    cat_cols = df.select_dtypes("object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols)

    # Fill remaining NaNs with median
    df = df.fillna(df.median(numeric_only=True))

    return df

def save_features(df: pd.DataFrame, filename: str = "train_features.parquet") -> None:
    PROCESSED.mkdir(exist_ok=True)
    df.to_parquet(PROCESSED / filename, index=False)
    print(f"Saved {len(df):,} rows to data/processed/{filename}")

if __name__ == "__main__":
    from src.ingestion.load_data import load_application
    df = load_application()
    df_eng = engineer_features(df)
    print(f"Features after engineering: {df_eng.shape[1]}")
    save_features(df_eng)