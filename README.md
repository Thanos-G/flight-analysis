# Flight Delay Prediction — Big Data & Data Mining

An end-to-end machine learning pipeline for predicting flight delays using 1.9 million records from the US Airline On-Time Performance dataset.

---

## Overview

This project applies Big Data and Data Mining techniques to predict whether a flight will be delayed by more than 30 minutes. It covers the full data science workflow: from data preprocessing and exploratory analysis to association rule mining and classification model evaluation.

---

## Project Structure

```
flight_analysis/
│
├── data/
│   ├── DelayedFlights.csv          # Raw dataset
│   ├── flight_clean.csv            # Cleaned dataset
│   └── flight_ml_ready.csv         # Feature-engineered dataset
│
├── notebooks/
│   ├── 01_preprocessing.ipynb      # Data cleaning & target variable
│   ├── 02_eda.ipynb                # Exploratory Data Analysis
│   ├── 03_feature_engineering.ipynb # Feature engineering
│   ├── 04_association_rules.ipynb  # FP-Growth association rules
│   ├── 05_classification.ipynb     # Naive Bayes, LR, Random Forest
│   ├── 06_neural_network.ipynb     # MLP Neural Network
│   └── 07_evaluation.ipynb         # Model comparison & evaluation
│
├── images/                         # Exported visualizations
│
└── README.md
```

---

## Dataset

- **Source:** [Kaggle — Airline Delay Causes](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses)
- **Records:** 1,928,371 (after cleaning)
- **Features:** 30 original + engineered features
- **Target:** `DELAYED` — binary (1 if ArrDelay > 30 min, else 0)
- **Class distribution:** 43% Delayed / 57% On-Time

---

## Installation

```bash
pip install pandas scikit-learn matplotlib seaborn mlxtend
```

---

## Usage

Run the notebooks in order:

```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

Each notebook saves its output to the `data/` folder for use in the next step.

---

## Results

| Model | Accuracy | F1 (macro) | ROC-AUC |
|---|---|---|---|
| Naive Bayes | 78.5% | 0.767 | 0.823 |
| Logistic Regression | 77.1% | 0.758 | 0.837 |
| Random Forest | 81.1% | 0.805 | 0.886 |
| **Neural Network** | **82.9%** | **0.821** | **0.901** |

Best model: **Neural Network** (MLP, 3 hidden layers: 128-64-32, ReLU, Adam)

---

## Key Findings

- **Seasonality:** Double peak pattern — winter (Jan-Feb) and summer (Jun-Jul) show the highest delay rates
- **Propagation effect:** LateAircraftDelay is the leading cause — delays accumulate throughout the day
- **Carrier performance:** Southwest Airlines (WN) consistently outperforms other carriers
- **Top predictive features:** NASDelay, CarrierDelay, CRSDepTime (departure time)

---

## Tech Stack

- **Language:** Python 3.x
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn, mlxtend
- **Environment:** Jupyter Notebook (VSCodium)

---

## License

This project is for educational purposes.
