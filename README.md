# Churn Prediction with MLflow, PyCaret & Optuna

An end-to-end machine learning experimentation pipeline for customer churn prediction, featuring automated model comparison, feature engineering, hyperparameter tuning, and comprehensive experiment tracking.

## 🎯 Overview

This project demonstrates a complete ML workflow:

1. **Exploratory Data Analysis (EDA)** - Understand data distributions and quality
2. **Baseline Models** - Compare Logistic Regression, Random Forest, and Gradient Boosting
3. **Feature Engineering** - Create domain-specific derived features
4. **Hyperparameter Tuning** - Optimize with Optuna's Bayesian optimization
5. **Model Registry** - Track and version models with MLflow

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| AutoML | PyCaret 3.x |
| Experiment Tracking | MLflow |
| Hyperparameter Tuning | Optuna |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |

## 📋 Prerequisites

- Python 3.9 - 3.11 (PyCaret does NOT support Python 3.12+)
- pyenv (recommended for Python version management)
- pip3
- libomp (for LightGBM on macOS: `brew install libomp`)

## 🚀 Quick Start

### 1. Set up Python environment

```bash
# Install Python 3.11 if needed (using pyenv)
pyenv install 3.11.11

# Create virtual environment with Python 3.11
~/.pyenv/versions/3.11.11/bin/python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 3. Run the experiment

```bash
python3 churn_experiment.py
```

### 4. View results in MLflow UI

```bash
mlflow ui --backend-store-uri mlflow_runs
```

Then open http://localhost:5000 in your browser.

## 📂 Project Structure

```
lab-churn-prediction-with-mlflow/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── spec.md                   # Project specification
├── generate_dataset.py       # Synthetic data generator
├── churn_experiment.py       # Main experiment pipeline
├── artifacts/                # Generated plots and models
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── final_model.pkl
├── mlflow_runs/              # MLflow tracking data
└── telco_churn.csv           # Generated dataset
```

## 📊 Dataset

The pipeline generates a synthetic Telco churn dataset with:

- **~7,000 customers**
- **20 features** (demographics, services, account info)
- **~26% churn rate** (realistic class imbalance)
- **Mixed data types** (numerical + categorical)

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| tenure | Numeric | Months with company |
| MonthlyCharges | Numeric | Monthly payment amount |
| TotalCharges | Numeric | Total amount paid |
| Contract | Categorical | Month-to-month, One year, Two year |
| InternetService | Categorical | DSL, Fiber optic, No |
| PaymentMethod | Categorical | Electronic check, Bank transfer, etc. |

## 🔬 Experiment Pipeline

### Step 1: EDA Analysis
- Dataset statistics
- Missing value detection
- Target distribution visualization
- Correlation heatmap

### Step 2: Baseline Models
Trains and compares three models:
- Logistic Regression
- Random Forest
- Gradient Boosting Classifier

### Step 3: Feature Engineering
Creates 6 derived features:
- `TenureBucket` - Customer lifecycle stage
- `ChargesPerTenure` - Value extraction rate
- `HighValueCustomer` - Premium customer flag
- `NumServices` - Total active services
- `ContractSecurityScore` - Retention risk score
- `PaymentRisk` - High-risk payment indicator

### Step 4: Hyperparameter Tuning
- Uses Optuna with TPE (Tree-structured Parzen Estimator)
- 20 optimization trials
- Optimizes for AUC metric

### Step 5: Final Model
- Trains on full dataset
- Generates confusion matrix
- Creates feature importance plot
- Registers in MLflow Model Registry

## 📈 MLflow Tracking

All experiments are tracked in MLflow with:

| Category | Items Logged |
|----------|--------------|
| **Parameters** | Model type, feature set, hyperparameters |
| **Metrics** | Accuracy, Precision, Recall, AUC, F1 |
| **Artifacts** | Plots, trained models |
| **Tags** | Run names, experiment context |

### Viewing Experiments

```bash
# Start MLflow UI
mlflow ui --backend-store-uri mlflow_runs

# Open in browser
open http://localhost:5000
```

## 🎯 Expected Results

Typical performance progression:

| Stage | AUC | Notes |
|-------|-----|-------|
| Baseline LR | ~0.78 | Simple, interpretable |
| Baseline GBC | ~0.82 | Best baseline |
| + Feature Engineering | ~0.84 | +2% improvement |
| + Optuna Tuning | ~0.85 | Final optimized |

## 🔧 Configuration

Key settings in `churn_experiment.py`:

```python
RANDOM_SEED = 42                           # Reproducibility
MLFLOW_EXPERIMENT_NAME = "churn_prediction_pycaret"
MLFLOW_TRACKING_URI = "mlflow_runs"        # Local directory
DATASET_PATH = "telco_churn.csv"
TARGET_COLUMN = "Churn"
```

## 📝 Customization

### Using Your Own Data

1. Replace `telco_churn.csv` with your data
2. Update `TARGET_COLUMN` if different
3. Modify `create_engineered_features()` for domain-specific features

### Adding Models

Edit `train_baseline_models()` to include additional PyCaret models:

```python
baseline_models = ['lr', 'rf', 'gbc', 'xgboost', 'lightgbm']
```

### Tuning Configuration

Adjust Optuna settings in `tune_model_with_optuna()`:

```python
tuned_model = tune_model(
    base_model,
    optimize='AUC',           # Metric to optimize
    n_iter=50,                # More trials = better results
    early_stopping=True
)
```

## 🐛 Troubleshooting

### PyCaret Installation Issues

```bash
# If pycaret fails, try:
pip3 install --upgrade pip
pip3 install pycaret[full] --no-cache-dir
```

### MLflow UI Not Loading

```bash
# Check if port 5000 is in use
lsof -i :5000

# Use alternative port
mlflow ui --backend-store-uri mlflow_runs --port 5001
```

### Memory Issues

For large datasets, reduce the sample size:

```python
df = df.sample(n=5000, random_state=42)
```

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- [PyCaret](https://pycaret.org/) - Low-code ML library
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Optuna](https://optuna.org/) - Hyperparameter optimization framework

