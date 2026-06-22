import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow

st.set_page_config(page_title="Credit Risk MLOps", layout="wide")
st.title("Credit Risk MLOps — Monitoring Dashboard")

client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=["1"], order_by=["metrics.roc_auc DESC"], max_results=10)

metrics_df = pd.DataFrame([
    {"run": r.info.run_name,
     "roc_auc": r.data.metrics.get("roc_auc", 0),
     "ks_stat": r.data.metrics.get("ks_stat", 0)}
    for r in runs
])

col1, col2, col3 = st.columns(3)
col1.metric("Best ROC-AUC", f"{metrics_df['roc_auc'].max():.4f}")
col2.metric("Best KS Stat", f"{metrics_df['ks_stat'].max():.4f}")
col3.metric("Total Runs", len(metrics_df))

st.subheader("Experiment history")
fig = px.line(metrics_df, x="run", y="roc_auc", markers=True)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Drift report")
try:
    with open("reports/drift_report.html", "r") as f:
        st.components.v1.html(f.read(), height=600, scrolling=True)
except FileNotFoundError:
    st.info("Run src/monitoring/drift_report.py first")