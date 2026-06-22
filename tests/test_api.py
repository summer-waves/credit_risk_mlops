from fastapi.testclient import TestClient
from api.main import app

def test_health():
    # Using the 'with' block triggers the @app.on_event("startup") 
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200

def test_predict():
    payload = {
        "AMT_INCOME_TOTAL": 50000, "AMT_CREDIT": 200000,
        "AMT_ANNUITY": 15000, "AMT_GOODS_PRICE": 180000,
        "EXT_SOURCE_1": 0.6, "EXT_SOURCE_2": 0.55,
        "EXT_SOURCE_3": 0.5, "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -1500
    }
    
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        
        # If it fails again, this will print the exact Python error to the console
        if r.status_code != 200:
            print(r.json())
            
        assert r.status_code == 200
        assert 0 <= r.json()["default_probability"] <= 1