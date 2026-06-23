# Credit Risk MLOps Platform

An end-to-end, production-grade Machine Learning Operations (MLOps) pipeline designed to predict loan default risk using the Home Credit dataset. 

This project demonstrates a complete lifecycle architecture, moving from raw data processing and model training to containerized API deployment, experiment tracking, automated CI/CD testing, and live data drift monitoring.

## 🏗️ Architecture & Tech Stack
* **Modeling:** XGBoost, Scikit-Learn, Pandas
* **Experiment Tracking & Registry:** MLflow (Local SQLite Database)
* **Model Serving:** FastAPI, Uvicorn
* **Containerization:** Docker, Docker Compose
* **Production Monitoring:** Evidently AI, Streamlit
* **CI/CD:** GitHub Actions, Pytest

---

## 🚀 Quick Start (Local Deployment)

# Build and spin up the containerized architecture
docker compose up --build
FastAPI Swagger Docs: http://localhost:8000/docs

Streamlit Monitoring Dashboard: http://localhost:8501

📊 Pipeline Phases & Project Showcase
1. Data Engineering & Model Training
Processed over 300,000 loan applications, engineering features such as debt-to-income ratios and external scoring means to build a robust XGBoost classifier.

2. MLflow Experiment Tracking & Model Registry
Logged parameters, ROC-AUC, and KS Statistics for every training run. The best-performing models are versioned and stored in the MLflow Model Registry for deployment.

MLflow Run Detail (ROC-AUC + KS logged)

MLflow Model Registry (CreditRiskXGB)

3. FastAPI Model Serving
Developed a REST API using FastAPI and Pydantic to serve the latest XGBoost model from the registry. The API accepts loan application payloads and returns a default probability and risk tier.

FastAPI Swagger /predict Endpoint

4. Containerization
Packaged the API and its dependencies into a lightweight Docker container, ensuring consistent environments across development and production.

Docker Terminal Output (Uvicorn running)

5. Data Drift Monitoring
Integrated Evidently AI to simulate and detect data drift in production environments. The drift reports are visualized dynamically via a Streamlit web application.

Streamlit Monitoring Dashboard

Evidently Data Drift Report

6. Continuous Integration (CI/CD)
Implemented a GitHub Actions workflow to automatically run Pytest suites on every push and pull request, ensuring API stability and graceful degradation if underlying artifacts are missing.

GitHub Actions Green CI Checkmark
