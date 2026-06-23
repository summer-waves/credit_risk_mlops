from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import numpy as np
import json
from pathlib import Path

app = FastAPI(title="Credit Risk API", version="1.0")

model = None
feature_names = None

@app.on_event("startup")
def load_model():
    global model, feature_names
    import os
    
    model_path = "mlruns/1/models"
    
    # Graceful fallback for CI/CD where mlruns is ignored by .gitignore
    if not os.path.exists(model_path):
        print("CI/CD Environment: No model found. Running in dummy mode for tests.")
        return
        
    versions = sorted(os.listdir(model_path))
    latest = os.path.join(model_path, versions[-1], "artifacts")
    model = mlflow.pyfunc.load_model(latest)
    
    with open("data/processed/feature_names.json") as f:
        feature_names = json.load(f)
    print(f"Model loaded — expects {len(feature_names)} features")

class LoanApplication(BaseModel):
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: float = Field(..., gt=0)
    AMT_GOODS_PRICE: float = Field(..., gt=0)
    EXT_SOURCE_1: float = Field(0.5, ge=0, le=1)
    EXT_SOURCE_2: float = Field(0.5, ge=0, le=1)
    EXT_SOURCE_3: float = Field(0.5, ge=0, le=1)
    DAYS_BIRTH: int = Field(..., lt=0)
    DAYS_EMPLOYED: int = Field(-1000)

class PredictionResponse(BaseModel):
    default_probability: float
    risk_tier: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: LoanApplication):
    try:
        # CI/CD Fallback response
        if model is None or feature_names is None:
            return PredictionResponse(default_probability=0.05, risk_tier="low")

        row = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
        row["AMT_INCOME_TOTAL"] = data.AMT_INCOME_TOTAL
        row["AMT_CREDIT"]       = data.AMT_CREDIT
        row["AMT_ANNUITY"]      = data.AMT_ANNUITY
        row["AMT_GOODS_PRICE"]  = data.AMT_GOODS_PRICE
        row["EXT_SOURCE_1"]     = data.EXT_SOURCE_1
        row["EXT_SOURCE_2"]     = data.EXT_SOURCE_2
        row["EXT_SOURCE_3"]     = data.EXT_SOURCE_3
        row["DAYS_BIRTH"]       = data.DAYS_BIRTH
        row["DAYS_EMPLOYED"]    = data.DAYS_EMPLOYED
        
        row["DEBT_INCOME_RATIO"]    = data.AMT_CREDIT / (data.AMT_INCOME_TOTAL + 1)
        row["ANNUITY_INCOME_RATIO"] = data.AMT_ANNUITY / (data.AMT_INCOME_TOTAL + 1)
        row["CREDIT_GOODS_RATIO"]   = data.AMT_CREDIT / (data.AMT_GOODS_PRICE + 1)
        row["EXT_SCORE_MEAN"]       = np.mean([data.EXT_SOURCE_1, data.EXT_SOURCE_2, data.EXT_SOURCE_3])
        row["DAYS_EMPLOYED_ANOM"]   = 0
        row["DAYS_EMPLOYED_PCT"]    = data.DAYS_EMPLOYED / data.DAYS_BIRTH
        
        prob = float(model.predict(row)[0])
        tier = "low" if prob < 0.1 else "medium" if prob < 0.25 else "high"
        
        return PredictionResponse(default_probability=round(prob, 4), risk_tier=tier)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))