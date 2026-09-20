# Financial Transaction Anomaly Detection System

An end-to-end machine learning pipeline for detecting fraudulent financial transactions using a hybrid combination of unsupervised outlier detection and supervised classification.

## Overview

Financial fraud detection faces a severe class imbalance problem, where fraudulent activity often makes up less than 0.1% of all transactions. Standard classifiers tend to predict all transactions as legitimate unless explicitly designed for rare events.

This repository implements a practical anomaly detection system that identifies high-risk transactions while keeping false positives low.

## Features

* Preprocessing & Feature Engineering: Cleans transaction logs and builds features like rolling transaction velocity and deviation from typical spending.
* Class Imbalance Mitigation: Applies SMOTE and undersampling techniques to balance training targets.
* Hybrid Machine Learning Models:
  * Unsupervised (Isolation Forest, Autoencoders): Flags novel anomalies without needing labeled training data.
  * Supervised (XGBoost, Random Forest): Accurately catches known fraud signatures.
* Evaluation Metrics: Uses Precision, Recall, F1-Score, and PR-AUC instead of misleading overall accuracy metrics.
* Prediction API: Serves real-time risk scores through a FastAPI application.

## Tech Stack

* Python 3.10+
* Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost, PyTorch
* FastAPI, Uvicorn
* Matplotlib, Seaborn

## Project Structure

Financial-Transaction-Anomaly-Detection-System/
├── data/                  # Raw and processed datasets
├── notebooks/             # Data exploration & modeling experiments
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
├── src/                   # Pipeline source code
│   ├── data_pipeline.py   # Preprocessing script
│   ├── models.py          # Model definition and training
│   └── utils.py           # Metrics and evaluation helpers
├── app/                   # Web service
│   └── main.py            # API endpoints
├── requirements.txt
└── README.md


## Quick Start

### 1. Clone the repository
```bash
git clone [https://github.com/Bhaskar344433/Financial-Transaction-Anomaly-Detection-System.git](https://github.com/Bhaskar344433/Financial-Transaction-Anomaly-Detection-System.git)
cd Financial-Transaction-Anomaly-Detection-System

```

### 2. Set up virtual environment

```bash
python -m venv venv
source venv/bin/activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

## How to Run

### Train models

```bash
python src/models.py

```

### Start the API service

```bash
uvicorn app.main:app --reload

```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

## Dataset

Add your CSV file (such as the Kaggle Credit Card Fraud dataset) into the `data/` folder before running the pipeline.

## Author

Bhaskar344433
