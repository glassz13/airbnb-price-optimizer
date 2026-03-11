# Airbnb Price Advisor — NYC

> Data-driven nightly price recommendations for Airbnb hosts using an XGBoost + Neural Network ensemble, trained on 14,000+ NYC listings.

Live App: [airbnb-price-advisor.streamlit.app](#) ← paste your URL here

---

## Overview

Airbnb hosts often underprice or overprice their listings without knowing where they stand in their neighbourhood market. This app takes your listing details and recommends an optimal nightly rate — showing how it compares to your neighbourhood median and which factors are driving the price.

This project extends a previous Airbnb NYC data analytics project, adding a full ML/DL training pipeline and an interactive price recommendation interface.

---

## Demo

![App Demo](assets/demo.gif)

**Input Form**
![Inputs](assets/app_inputs.png)

**Price Results**
![Results](assets/app_results.png)

**Feature Importance**
![Features](assets/app_features.png)

---

## Project Structure
```
airbnb-price-optimizer/
│
├── train.py                  # ML/DL training pipeline
├── app.py                    # Streamlit app
├── airbnb_data.csv           # Dataset
├── model_artifacts.pkl       # Saved model + encoders + scaler
├── nn_model.pt               # Saved neural network weights
├── requirements.txt
└── assets/
    ├── demo.gif
    ├── app_inputs.png
    ├── app_results.png
    └── app_features.png
```

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | Kaggle — Airbnb NYC Open Data |
| Raw rows | 20,786 listings |
| After cleaning | 14,277 listings |
| Price range | $41 – $703 / night |
| City | New York City |

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `geo_distance` | Haversine distance from Times Square |
| `occupancy_rate` | 1 - availability / 365 |
| `host_quality` | Composite of superhost status, rating, listing count |
| `log_reviews` | Log-scaled review count |
| `rating_x_reviews` | Review score × log reviews interaction |
| `accommodates_sqrd` | Guests squared — captures non-linear demand |
| `neigh_price_tier` | Weighted blend of neighbourhood mean and median |

---

## Workflow
```
Load & Clean → Feature Engineering → Encode Categoricals
    → Train/Test Split → XGBoost (5-Fold CV)
    → Neural Network (PyTorch) → Weighted Ensemble
    → Save Artifacts → Streamlit App
```

---

## Models

### XGBoost
- 5-fold cross validation
- 600 estimators, learning rate 0.04, max depth 6
- Histogram-based tree method for speed

### Neural Network (PyTorch)
- Entity embeddings for categorical features (neighbourhood, room type, property type, host type)
- Architecture: Linear(128) → BatchNorm → ReLU → Dropout → Linear(64) → ReLU → Linear(1)
- HuberLoss with AdamW optimizer
- ReduceLROnPlateau scheduler

### Ensemble
- Weighted average: XGBoost × 0.70 + NN × 0.30
- Weight selected by minimizing MAE on test set

---

## Results

| Model | MAE | R² |
|-------|-----|----|
| XGBoost (5-fold CV) | $41.84 ± $1.10 | — |
| XGBoost (test) | $41.61 | 0.7316 |
| Neural Network (test) | $44.74 | 0.6719 |
| **Ensemble (test)** | **$41.01** | **0.7330** |

---

## Top Predictors

| Feature | Importance |
|---------|------------|
| Guest capacity (sq) | 0.155 |
| Area mean price | 0.130 |
| Guests | 0.118 |
| Room type | 0.093 |
| Bathrooms | 0.086 |

Guest capacity and neighbourhood pricing dominate — consistent with real-world Airbnb market dynamics.

---

## App Features

- Recommended nightly price based on your listing specs
- Comparison against neighbourhood median price
- Market position indicator (above / below / well positioned)
- Feature importance chart showing what drives the price
- Neighbourhood price distribution with your price marked

---

## Tech Stack

- Python 3.x
- pandas, numpy, scikit-learn
- xgboost
- PyTorch
- Streamlit
- Plotly

---

## How to Run
```bash
pip install -r requirements.txt

# Train the model first
python train.py

# Launch the app
streamlit run app.py
```

---

## requirements.txt
```
pandas
numpy
scikit-learn
xgboost
torch
streamlit
plotly
```
