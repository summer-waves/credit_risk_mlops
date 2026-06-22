import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

def load_application() -> pd.DataFrame:
    df = pd.read_csv(RAW / "application_train.csv")
    print(f"Rows: {len(df):,}  |  Columns: {df.shape[1]}  |  Default rate: {df['TARGET'].mean():.2%}")
    return df

def basic_eda(df: pd.DataFrame) -> None:
    print("\n--- Dtypes ---")
    print(df.dtypes.value_counts())
    print("\n--- Top 20 columns with missing values ---")
    print(df.isnull().sum().sort_values(ascending=False).head(20))
    
if __name__ == "__main__":
    df = load_application()
    basic_eda(df)
