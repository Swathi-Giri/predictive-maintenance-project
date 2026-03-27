# 🔧 Predictive Maintenance System for Turbofan Engines

> An end-to-end machine learning pipeline that predicts engine failure using NASA C-MAPSS sensor data — from exploratory analysis to a live deployed dashboard with real-time predictions.

🔴 **[Live Demo →](YOUR_DEPLOYED_URL_HERE)** &nbsp;|&nbsp; 📊 **[API Docs →](YOUR_API_URL/docs)**

---

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)

---

## 📌 Business Impact

| Metric | Value |
|--------|-------|
| Failure detection window | **48+ cycles** in advance |
| Prediction accuracy (±15 cycles) | **XX%** of cases |
| Late predictions (missed failures) | Under **X%** |
| Potential downtime reduction | Significant — early warnings enable scheduled maintenance |

> *"This system detects engine degradation 48 cycles before failure, giving maintenance teams a multi-day window to schedule repairs instead of dealing with unexpected breakdowns."*

---

## 🏗️ System Architecture

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────┐    ┌──────────────────┐
│  NASA C-MAPSS │───▶│  Feature Engine   │───▶│   ML Model   │───▶│  FastAPI Server  │
│  Sensor Data  │    │  130+ features    │    │  XGBoost /   │    │  REST API        │
│  21 sensors   │    │  Rolling stats    │    │  LightGBM /  │    │  /predict        │
│  per cycle    │    │  Trends & EMA     │    │  Random Forest│    │  /health         │
└──────────────┘    └───────────────────┘    └──────────────┘    └────────┬─────────┘
                                                                          │
                                                                          ▼
                                                                 ┌──────────────────┐
                                                                 │ Streamlit Dashboard│
                                                                 │ • Health Gauge     │
                                                                 │ • Sensor Trends    │
                                                                 │ • RUL Tracking     │
                                                                 │ • Live Simulation  │
                                                                 └──────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11 |
| **ML Models** | Random Forest, XGBoost, LightGBM |
| **Feature Engineering** | Rolling statistics, EMA, trend indicators (130+ features) |
| **API** | FastAPI + Uvicorn |
| **Dashboard** | Streamlit + Plotly (interactive charts) |
| **Containerization** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Testing** | Pytest |
| **Data** | NASA C-MAPSS Turbofan Engine Degradation Dataset |

---

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/predictive-maintenance.git
cd predictive-maintenance

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the NASA dataset
python download_data.py

# Train the model
python src/train.py

# Start the live dashboard
cd dashboard && streamlit run app.py
# Opens at http://localhost:8501
```

### Option 2: Run with Docker

```bash
docker-compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## 📊 Model Performance

| Model | MAE (cycles) | RMSE | R² | Within ±15 | Late Pred. |
|-------|:------------:|:----:|:--:|:----------:|:----------:|
| Random Forest | 20.05 | XX.X | 0.XX | XX% | X.X% |
| XGBoost | XX.X | XX.X | 0.XX | XX% | X.X% |
| **LightGBM** | **XX.X** | **XX.X** | **0.XX** | **XX%** | **X.X%** |

> **MAE** = Mean Absolute Error (lower is better)
> **Within ±15** = % of predictions within 15 cycles of actual failure
> **Late Pred.** = % of predictions that overestimate RUL (dangerous — could miss failure)

---

## 📁 Project Structure

```
predictive-maintenance/
├── src/
│   ├── data_loader.py         # Load & preprocess NASA C-MAPSS data
│   ├── features.py            # 130+ engineered features
│   ├── train.py               # Train & compare 3 ML models
│   └── predict.py             # Prediction & health classification
├── api/
│   └── main.py                # FastAPI REST API
├── dashboard/
│   └── app.py                 # Streamlit live monitoring dashboard
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── tests/
│   └── test_predict.py        # Unit tests (pytest)
├── models/                    # Trained models (generated)
├── data/raw/                  # NASA dataset (downloaded)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/workflows/ci.yml   # GitHub Actions CI/CD
└── README.md
```

---

## 🔍 Key Features

### Feature Engineering (what makes this project stand out)

Raw sensor data → **130+ engineered features** including:
- **Rolling statistics** (window = 5, 10, 20 cycles) — captures trend direction
- **Exponential Moving Averages** — weights recent readings more heavily
- **Cycle-over-cycle differences** — captures rate of degradation
- **Normalized time features** — provides temporal context

### Dashboard Highlights
- **Real-time health gauge** with color-coded zones (Healthy → Danger)
- **Interactive sensor trend charts** — select any combination of 14 sensors
- **Actual vs Predicted RUL tracking** — see model accuracy over engine life
- **Live simulation mode** — watch an engine degrade in real-time

### API
- Full REST API with automatic Swagger documentation at `/docs`
- Health check endpoint for monitoring
- Returns predicted RUL + health status + recommended action

---

## 📈 Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)
- 100 turbofan engines run to failure
- 21 sensor channels per operating cycle
- 3 operational settings
- Single fault mode: High-Pressure Compressor degradation

Source: [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## 🎓 What I Learned

- Designing **end-to-end ML pipelines** from raw data to production deployment
- **Feature engineering** for time-series sensor data (rolling stats, EMA, trend indicators)
- Evaluating models with **business-relevant metrics**, not just accuracy
- Building **REST APIs** with FastAPI and interactive dashboards with Streamlit
- **Docker containerization** for reproducible deployment
- Setting up **CI/CD pipelines** with GitHub Actions

---

## 📬 Contact

**Your Name** — Applied Data Science Student at [Your University]

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/YOUR_LINKEDIN)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:your.email@example.com)

---

*Built as a portfolio project demonstrating applied data science skills for predictive maintenance in manufacturing and automotive industries.*
