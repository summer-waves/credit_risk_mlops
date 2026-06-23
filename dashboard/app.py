import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow

st.set_page_config(page_title="Credit Risk MLOps", layout="wide")
st.title("Credit Risk MLOps — Monitoring Dashboard")

# Attempt to load runs from MLflow
client = mlflow.tracking.MlflowClient()
try:
    runs = client.search_runs(experiment_ids=["1"], order_by=["metrics.roc_auc DESC"], max_results=10)
except Exception:
    runs = []

# Graceful Fallback for Cloud Deployment
if not runs:
    st.warning("⚠️ Running in Cloud Mode: Live MLflow tracking database not found. Displaying static portfolio snapshot.")
    metrics_df = pd.DataFrame([
        {"run": "xgb-baseline-v2", "roc_auc": 0.7600, "ks_stat": 0.3863},
        {"run": "xgb-baseline-v1", "roc_auc": 0.7512, "ks_stat": 0.3721}
    ])
    total_runs = "2 (Simulated)"
else:
    metrics_df = pd.DataFrame([
        {"run": r.info.run_name, 
         "roc_auc": r.data.metrics.get("roc_auc", 0), 
         "ks_stat": r.data.metrics.get("ks_stat", 0)}
        for r in runs
    ])
    total_runs = len(metrics_df)

col1, col2, col3 = st.columns(3)
col1.metric("Best ROC-AUC", f"{metrics_df['roc_auc'].max():.4f}")
col2.metric("Best KS Stat", f"{metrics_df['ks_stat'].max():.4f}")
col3.metric("Total Runs", total_runs)

st.subheader("Experiment history")
fig = px.line(metrics_df, x="run", y="roc_auc", markers=True)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Drift report")
try:
    with open("reports/drift_report.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=600, scrolling=True)
except FileNotFoundError:
    st.info("Run src/monitoring/drift_report.py first")
