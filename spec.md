## Churn Experimentation with PyCaret + Optuna + MLflow

> **Role & Expectation**
> You are a senior Machine Learning engineer building a **reproducible churn prediction experimentation pipeline**.
> Your output must be **clean, modular, runnable Python code** with clear separation of concerns and inline comments.

---

### 🎯 Objective

Build an **end-to-end churn prediction workflow** that:

1. Performs **Exploratory Data Analysis (EDA)**
2. Creates and evaluates **baseline models**
3. Uses **PyCaret for rapid experimentation**
4. Performs **hyperparameter tuning using Optuna**
5. Tracks **all experiments, runs, metrics, and artifacts using MLflow**

---

### 📂 Dataset To be Created

- Dataset: `telco_churn.csv`
- Target column: `Churn` (Yes / No)
- Mix of numerical and categorical features
- Handle missing values gracefully

---

### 🧱 Mandatory Technical Requirements

- Python 3.9+
- Libraries:

  - pandas
  - numpy
  - matplotlib / seaborn
  - pycaret
  - mlflow
  - optuna

- Use **MLflow Tracking** locally
- Use **PyCaret Classification module**
- Use **Optuna as the hyperparameter tuner**
- Ensure **reproducibility** (random seeds)

---

### 🧪 Step-by-Step Tasks (MUST FOLLOW ORDER)

#### 1️⃣ Exploratory Data Analysis (EDA)

- Load the dataset
- Show:

  - Dataset shape
  - Data types
  - Missing value summary
  - Target class distribution

- Create and save:

  - Churn distribution plot
  - Correlation heatmap (numerical features)

- Log EDA plots as **MLflow artifacts**

---

#### 2️⃣ MLflow Experiment Setup

- Create an MLflow experiment named:
  `churn_prediction_pycaret`
- Use **meaningful run names**
- Log:

  - Dataset shape
  - Number of features
  - Target imbalance ratio as MLflow parameters

---

#### 3️⃣ Baseline Model Creation (PyCaret)

- Initialize PyCaret classification setup with:

  - Train/test split
  - Automatic preprocessing
  - Feature encoding

- Compare baseline models using:

  - Logistic Regression
  - Random Forest
  - Gradient Boosting

- Log:

  - Accuracy
  - Precision
  - Recall
  - AUC

- Track **each model comparison as a separate MLflow run**

---

#### 4️⃣ Feature Engineering Iteration

- Create **new derived features** (examples):

  - Tenure buckets
  - Charges per tenure

- Re-train baseline models on new features
- Track results as **new MLflow runs**
- Compare baseline vs engineered features

---

#### 5️⃣ Hyperparameter Tuning using Optuna

- Select the **best baseline model**
- Perform hyperparameter tuning using:

  - `tune_model()`
  - Optuna as the search backend

- Log:

  - Best parameters
  - Improved metrics
  - Tuning trials

- Store tuning results in MLflow

---

#### 6️⃣ Final Model & Artifacts

- Finalize the tuned model
- Log:

  - Final model artifact
  - Confusion matrix
  - Feature importance plot

- Register the model in MLflow Model Registry as:
  `ChurnPredictionModel`

---

### 📌 Output Expectations

- Modular code (functions where appropriate)
- Clear comments explaining _why_, not just _what_
- No notebook-only magic (write script-friendly code)
- All MLflow logging must be explicit and visible in UI
- No skipped steps
- Use python3, pip3 and pyenv

---

### 🚫 Do NOT

- Skip MLflow logging
- Hardcode paths
- Use placeholder metrics
- Mix multiple concerns in a single function

---

### ✅ Final Deliverable

Produce a **single, end-to-end Python script** that:

- Can be run locally
- Creates a complete MLflow experiment history
- Demonstrates baseline → feature engineering → tuning progression clearly
