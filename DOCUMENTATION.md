# Churn Prediction Pipeline - Complete Documentation

> 📚 A beginner-friendly guide to understanding the ML experimentation pipeline

---

## Table of Contents

1. [What is This Project?](#what-is-this-project)
2. [The Big Picture](#the-big-picture)
3. [Configuration Setup](#configuration-setup)
4. [Data Loading](#data-loading)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Data Preprocessing](#data-preprocessing)
7. [Feature Engineering](#feature-engineering)
8. [Baseline Model Training](#baseline-model-training)
9. [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Final Model & Artifacts](#final-model--artifacts)
11. [The Main Pipeline](#the-main-pipeline)

---

## What is This Project?

### 🎯 The Problem

Imagine you run a telecom company with millions of customers. Every month, some customers leave (this is called **"churn"**). Acquiring new customers is 5-10x more expensive than keeping existing ones!

**The Question:** Can we predict which customers are likely to leave, so we can offer them special deals to stay?

### 🔧 The Solution

This project builds a machine learning model that:
1. Analyzes customer data (payment history, services used, contract type)
2. Learns patterns of customers who left vs. those who stayed
3. Predicts which current customers might leave soon

**Real-World Analogy:** Think of it like a weather forecast. Just as meteorologists use temperature, pressure, and humidity patterns to predict rain, we use customer behavior patterns to predict churn.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE ML PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📊 EDA          🧹 Preprocessing      🔧 Feature Engineering    │
│  "Understand     "Clean the           "Create new                │
│   the data"       data"                insights"                 │
│      │               │                     │                     │
│      ▼               ▼                     ▼                     │
│  ┌───────┐      ┌─────────┐          ┌──────────┐               │
│  │ Plots │      │ Handle  │          │ Tenure   │               │
│  │ Stats │      │ Missing │          │ Buckets  │               │
│  └───────┘      └─────────┘          └──────────┘               │
│                                                                  │
│  🤖 Baseline     ⚡ Tuning            📦 Final Model            │
│  "Test basic     "Optimize           "Save the                  │
│   models"         the best"           winner"                   │
│      │               │                     │                     │
│      ▼               ▼                     ▼                     │
│  ┌───────┐      ┌─────────┐          ┌──────────┐               │
│  │  LR   │      │ Optuna  │          │ MLflow   │               │
│  │  RF   │      │ 20      │          │ Registry │               │
│  │  GBC  │      │ trials  │          └──────────┘               │
│  └───────┘      └─────────┘                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Setup

### 📍 Location in Code: Lines 30-67

```python
RANDOM_SEED = 42
MLFLOW_EXPERIMENT_NAME = "churn_prediction_pycaret"
MLFLOW_TRACKING_URI = "mlflow_runs"
DATASET_PATH = "telco_churn.csv"
TARGET_COLUMN = "Churn"
ARTIFACTS_DIR = Path("artifacts")
```

### ❓ WHY do we need this?

Think of configuration as the **settings page** of your phone. Before using any app, you configure preferences. Similarly, before running ML experiments:

| Setting | Purpose | Real-World Analogy |
|---------|---------|-------------------|
| `RANDOM_SEED = 42` | Makes experiments reproducible | Like using the same recipe every time you bake a cake |
| `MLFLOW_EXPERIMENT_NAME` | Groups related experiments | Like organizing photos into albums |
| `MLFLOW_TRACKING_URI` | Where to save experiment data | Like choosing which folder to save files |
| `DATASET_PATH` | Location of our data | Like the address of a restaurant |
| `TARGET_COLUMN` | What we're trying to predict | Like the answer key in an exam |
| `ARTIFACTS_DIR` | Where to save plots/models | Like a "Downloads" folder |

### 🔨 WHAT does `setup_environment()` do?

```python
def setup_environment() -> None:
    np.random.seed(RANDOM_SEED)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
```

1. **Sets random seed** - Ensures randomness is "controlled"
2. **Creates folders** - Makes sure we have a place to save outputs
3. **Configures MLflow** - Sets up our experiment tracker

### 🎲 HOW does random seed work?

**Analogy:** Imagine shuffling a deck of cards. Normally, each shuffle is different. But what if you had a "magic shuffle number"? Using the same number always produces the same shuffle order.

```
Without seed:
Run 1: [A, 5, K, 2, 7] → Model accuracy: 78%
Run 2: [3, J, 9, Q, 4] → Model accuracy: 82%  ← Different!

With seed=42:
Run 1: [A, 5, K, 2, 7] → Model accuracy: 78%
Run 2: [A, 5, K, 2, 7] → Model accuracy: 78%  ← Same!
```

---

## Data Loading

### 📍 Location in Code: Lines 70-83

```python
def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset from {filepath}")
    return df
```

### ❓ WHY is this a separate function?

**Single Responsibility Principle:** Each function should do ONE thing well.

**Analogy:** In a restaurant, there's a separate person who takes orders, another who cooks, and another who serves. The "load_dataset" function is like the waiter who just brings ingredients from the pantry.

### 🔨 WHAT does it do?

| Step | Code | What Happens |
|------|------|--------------|
| 1 | `pd.read_csv(filepath)` | Opens CSV file and creates a table (DataFrame) |
| 2 | `print(...)` | Confirms the file was loaded successfully |
| 3 | `return df` | Passes the data to the next step |

---

## Exploratory Data Analysis (EDA)

### 📍 Location in Code: Lines 86-256

### ❓ WHY do we need EDA?

**Analogy:** Before a doctor prescribes medicine, they first examine you - check your temperature, blood pressure, ask about symptoms. EDA is the **medical checkup for your data**.

Without EDA, you might:
- Feed garbage data to the model (garbage in → garbage out)
- Miss important patterns
- Make wrong assumptions

### 🔨 WHAT does `perform_eda()` do?

```python
def perform_eda(df: pd.DataFrame) -> Dict[str, Any]:
```

This function is like a **data detective** that answers:

| Question | Code | Answer Example |
|----------|------|----------------|
| How big is our data? | `df.shape` | 7043 rows × 21 columns |
| What types of data do we have? | `df.dtypes.value_counts()` | 17 text columns, 4 number columns |
| Is data missing anywhere? | `df.isnull().sum()` | TotalCharges: 253 missing |
| How imbalanced is churn? | `df[TARGET].value_counts()` | No: 67%, Yes: 33% |

### 📊 HOW does it calculate imbalance ratio?

```python
minority_class = target_dist.min()    # 2342 (Yes - churned)
majority_class = target_dist.max()    # 4701 (No - stayed)
imbalance_ratio = majority_class / minority_class  # 4701/2342 = 2.01
```

**Analogy:** In a class of 30 students, if 20 are right-handed and 10 are left-handed, the ratio is 2:1. Our churn data has a 2:1 imbalance (more "stayed" than "left").

### 🎨 WHAT does `create_eda_visualizations()` do?

Creates two important plots:

#### 1. Churn Distribution Bar Chart
```
     ┌────────────┐
4701 │████████████│ No (Stayed)
     ├────────────┤
2342 │██████      │ Yes (Left)
     └────────────┘
```

**Why this matters:** Shows us the class imbalance visually. If 99% stayed and only 1% left, predicting "No" every time would give 99% accuracy but be useless!

#### 2. Correlation Heatmap

**Analogy:** A correlation heatmap is like a **relationship chart** at a family reunion. It shows who's connected to whom.

```
              Tenure  MonthlyCharges  TotalCharges  Churn
Tenure          1.0          0.2          0.8      -0.4
MonthlyCharges  0.2          1.0          0.6       0.2
TotalCharges    0.8          0.6          1.0      -0.2
Churn          -0.4          0.2         -0.2       1.0
```

- **+1.0** = Perfect positive relationship (when one goes up, other goes up)
- **-1.0** = Perfect negative relationship (when one goes up, other goes down)
- **0.0** = No relationship

**Insight:** Tenure has -0.4 correlation with Churn → Longer-term customers churn less!

### 📝 HOW does `log_eda_to_mlflow()` work?

```python
def log_eda_to_mlflow(eda_metrics, plot_paths):
    with mlflow.start_run(run_name="01_EDA_Analysis"):
        mlflow.log_params({...})      # Save settings
        mlflow.log_metrics({...})     # Save numbers
        mlflow.log_artifact(path)     # Save files (plots)
```

**Analogy:** Think of MLflow as a **lab notebook** for scientists:
- **Parameters:** The ingredients used (dataset size, features)
- **Metrics:** The measurements taken (churn rate, missing values)
- **Artifacts:** The photos/charts (visualizations)

---

## Data Preprocessing

### 📍 Location in Code: Lines 258-292

### ❓ WHY do we need preprocessing?

**Analogy:** Before cooking, you wash vegetables, peel onions, and measure ingredients. Raw data is "dirty" - it has missing values, wrong formats, and irrelevant columns.

### 🔨 WHAT does `preprocess_data()` do?

```python
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Don't modify original
    
    # 1. Remove ID column (not useful for prediction)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # 2. Handle missing TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 3. Fill other missing values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])  # Most common value
            else:
                df[col] = df[col].fillna(df[col].median())   # Middle value
```

### 🧹 HOW does it handle missing values?

| Data Type | Strategy | Why? | Example |
|-----------|----------|------|---------|
| Text (categorical) | Mode (most frequent) | Safe default | Gender missing → "Male" (most common) |
| Numbers | Median (middle value) | Resistant to outliers | Age missing → 35 (median age) |
| TotalCharges | Zero | New customers have no charges yet | New customer → $0 |

**Analogy - Median vs Mean:**

```
Salaries: [$30K, $40K, $50K, $60K, $1M]

Mean (average) = $236K  ← Skewed by the millionaire!
Median (middle) = $50K  ← More representative
```

---

## Feature Engineering

### 📍 Location in Code: Lines 294-365

### ❓ WHY do we need feature engineering?

**Analogy:** Raw flour, eggs, and sugar aren't a cake. You need to combine them in the right way. Feature engineering **transforms raw data into insights** that models can use better.

### 🔨 WHAT features are created?

```python
def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
```

| Feature | What It Does | Why It Helps |
|---------|--------------|--------------|
| `TenureBucket` | Groups tenure into categories | "0-6 months" is different from "4+ years" |
| `ChargesPerTenure` | Total charges / tenure | Reveals payment rate over time |
| `HighValueCustomer` | Flag if monthly charges > 75th percentile | Premium customers may behave differently |
| `NumServices` | Count of services used | More services = more engaged = less likely to leave |
| `ContractSecurityScore` | Points for contract + support features | Composite risk score |
| `PaymentRisk` | Flag for electronic check users | This payment method correlates with churn |

### 🎯 HOW does TenureBucket work?

```python
df['TenureBucket'] = pd.cut(
    df['tenure'],
    bins=[-1, 6, 12, 24, 48, 72],
    labels=['0-6_months', '6-12_months', '1-2_years', '2-4_years', '4+_years']
)
```

**Analogy:** Instead of asking "How old are you?" (exact number), we ask "Are you a teenager, adult, or senior?" (category).

```
Before (raw):
tenure = [1, 5, 8, 15, 36, 60]

After (bucketed):
TenureBucket = ['0-6_months', '0-6_months', '6-12_months', '1-2_years', '2-4_years', '4+_years']
```

**Why categories?** The model can learn: "0-6 month customers churn 3x more than 4+ year customers" - a pattern that's hard to see with raw numbers.

### 🎮 HOW does NumServices work?

```python
def count_services(row):
    count = 0
    for col in service_cols:
        if row[col] == 'Yes':
            count += 1
    return count
```

**Analogy:** Counting how many apps you use on a streaming service. If you use Netflix for movies, TV shows, and documentaries, you're more "sticky" than someone who only watches one category.

```
Customer A: Phone=Yes, Internet=Yes, Security=Yes → NumServices=3
Customer B: Phone=Yes, Internet=No → NumServices=1

Customer A is more engaged and less likely to churn!
```

---

## Baseline Model Training

### 📍 Location in Code: Lines 368-503

### ❓ WHY do we train multiple baseline models?

**Analogy:** When hiring for a job, you don't interview just one candidate. You compare several to find the best fit.

### 🤖 WHAT models do we train?

| Model | Full Name | How It Works | Analogy |
|-------|-----------|--------------|---------|
| `lr` | Logistic Regression | Draws a line to separate classes | Like drawing a boundary on a map |
| `rf` | Random Forest | Many decision trees vote together | Like asking 100 experts and taking majority vote |
| `gbc` | Gradient Boosting | Trees learn from each other's mistakes | Like students studying together, each fixing others' errors |

### 🔧 HOW does the training work?

```python
def train_baseline_models(df: pd.DataFrame) -> Dict[str, Any]:
    # 1. Initialize PyCaret (sets up preprocessing)
    setup(
        data=df,
        target=TARGET_COLUMN,
        session_id=RANDOM_SEED,
        train_size=0.8,  # 80% for training, 20% for testing
        normalize=True,  # Scale features to same range
    )
    
    # 2. Train each model
    for model_id in ['lr', 'rf', 'gbc']:
        model = create_model(model_id)  # Train
        metrics_df = pull()              # Get results
        
        # 3. Log to MLflow
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "f1_score": f1
        })
```

### 📊 Understanding the Metrics

| Metric | What It Measures | Analogy |
|--------|------------------|---------|
| **Accuracy** | % of correct predictions overall | % of exam questions you got right |
| **Precision** | Of those predicted "Yes", how many were actually "Yes"? | Of all fire alarms, how many were real fires? |
| **Recall** | Of all actual "Yes", how many did we catch? | Of all real fires, how many did the alarm detect? |
| **AUC** | Overall model quality (0 to 1) | Grade from F (0.5) to A+ (1.0) |
| **F1 Score** | Balance of precision and recall | Harmonic mean - balances both |

**Precision vs Recall Example:**

```
Actual churners: [A, B, C, D, E] (5 customers)

Model predictions: [A, B, F, G] (predicted 4 will churn)

Precision = 2/4 = 50% (A, B were correct out of 4 predictions)
Recall = 2/5 = 40% (caught 2 out of 5 actual churners)
```

### ⚖️ The Precision-Recall Tradeoff

**Analogy:** A spam filter:
- **High Precision, Low Recall:** Almost never marks good emails as spam, but lets some spam through
- **High Recall, Low Precision:** Catches all spam, but also marks some good emails as spam

For churn prediction, we usually prefer **higher recall** - we'd rather contact a customer who wasn't going to leave than miss one who was!

---

## Hyperparameter Tuning

### 📍 Location in Code: Lines 583-700

### ❓ WHY do we need tuning?

**Analogy:** A recipe says "bake at 350°F for 30 minutes." But what if 375°F for 25 minutes is better for YOUR oven? Tuning finds the best "settings" for our model.

### ⚙️ WHAT are hyperparameters?

These are **settings we choose before training** (unlike parameters that the model learns):

| Hyperparameter | What It Controls | Analogy |
|----------------|------------------|---------|
| `n_estimators` | Number of trees in the forest | Number of experts in a committee |
| `max_depth` | How deep each tree can grow | How many questions each expert can ask |
| `learning_rate` | How fast the model adapts | Study intensity (slow = thorough, fast = surface-level) |
| `min_samples_split` | Minimum data points to split | Minimum votes needed for a decision |

### 🔍 HOW does Optuna work?

```python
tuned_model = tune_model(
    base_model,
    optimize='AUC',
    search_library='optuna',
    search_algorithm='tpe',  # Tree-structured Parzen Estimator
    n_iter=20,  # 20 trials
)
```

**Analogy - Finding the best restaurant in a city:**

| Search Method | How It Works | Efficiency |
|---------------|--------------|------------|
| **Grid Search** | Try every restaurant one by one | Slow, thorough |
| **Random Search** | Pick 20 random restaurants | Faster, might miss best |
| **Optuna (TPE)** | Start random, then focus on promising neighborhoods | Smart, efficient |

**TPE (Tree-structured Parzen Estimator) Process:**

```
Trial 1: n_estimators=100, max_depth=5 → AUC=0.62
Trial 2: n_estimators=200, max_depth=10 → AUC=0.65 (Better! Try similar)
Trial 3: n_estimators=180, max_depth=12 → AUC=0.67 (Even better!)
...
Trial 20: n_estimators=165, max_depth=11 → AUC=0.68 (Best found!)
```

Optuna **learns from previous trials** to make smarter guesses!

---

## Final Model & Artifacts

### 📍 Location in Code: Lines 702-843

### ❓ WHY do we finalize and save the model?

**Analogy:** After a cooking competition, the winning recipe is written down, photographed, and saved for future use. We do the same with our best model.

### 🔨 WHAT does `create_final_model_artifacts()` do?

```python
def create_final_model_artifacts(model, df, tuning_results):
```

| Step | What Happens | Why |
|------|--------------|-----|
| 1. `finalize_model()` | Train on 100% of data | More data = better model |
| 2. `save_model()` | Save to `.pkl` file | For later use without retraining |
| 3. Confusion Matrix | Plot predictions vs reality | Visual error analysis |
| 4. Feature Importance | Rank which features matter | Explain decisions |
| 5. MLflow Registry | Version and deploy model | Production-ready |

### 📊 Understanding the Confusion Matrix

```
                    Predicted
                 No      Yes
Actual    No    TN       FP
         Yes    FN       TP
```

| Cell | Name | Meaning | Analogy |
|------|------|---------|---------|
| TN | True Negative | Correctly predicted "won't churn" | Doctor correctly says "you're healthy" |
| FP | False Positive | Wrongly predicted "will churn" | False fire alarm |
| FN | False Negative | Missed a churner | Missed a real fire 🔥 |
| TP | True Positive | Correctly predicted "will churn" | Alarm caught real fire ✓ |

**For churn, FN (False Negatives) are the most costly** - we failed to identify a leaving customer!

### 🏆 HOW does Feature Importance work?

The model tells us which features were most useful for predictions:

```
Feature Importance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Contract_Month-to-month  ████████████ 0.25
tenure                   ██████████   0.20
MonthlyCharges           ████████     0.15
NumServices              ██████       0.12
TechSupport_No           █████        0.10
...
```

**Insight:** Month-to-month contracts are the #1 predictor of churn! This is actionable - offer discounts for yearly contracts.

### 📦 Model Registry with MLflow

```python
mlflow.sklearn.log_model(
    final_model,
    artifact_path="model",
    registered_model_name="ChurnPredictionModel"
)
```

**Analogy:** Like version control for code (Git), but for ML models:

```
ChurnPredictionModel
├── Version 1 (Jan 2024) - AUC: 0.62 - Archived
├── Version 2 (Feb 2024) - AUC: 0.65 - Staging
└── Version 3 (Mar 2024) - AUC: 0.68 - Production ✓
```

---

## The Main Pipeline

### 📍 Location in Code: Lines 882-954

### 🎭 The Complete Story

```python
def main():
    # Act 1: Setup
    setup_environment()
    df_raw = load_dataset(DATASET_PATH)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Act 2: Understanding (EDA)
    eda_metrics = perform_eda(df_raw)
    plot_paths = create_eda_visualizations(df_raw)
    log_eda_to_mlflow(eda_metrics, plot_paths)
    
    # Act 3: Cleaning
    df_clean = preprocess_data(df_raw)
    
    # Act 4: First Attempt (Baseline)
    baseline_results = train_baseline_models(df_clean)
    
    # Act 5: Improvement (Feature Engineering)
    engineered_results, df_engineered = train_models_with_engineered_features(df_clean)
    
    # Act 6: Optimization (Tuning)
    tuned_model, tuning_results = tune_model_with_optuna(df_engineered, 'gbc')
    
    # Act 7: Finale (Save & Register)
    create_final_model_artifacts(tuned_model, df_engineered, tuning_results)
    print_experiment_summary()
```

### 🎬 Movie Analogy: The Hero's Journey

| Pipeline Step | Movie Equivalent |
|---------------|-----------------|
| Load Data | Hero receives the quest |
| EDA | Hero studies the map |
| Preprocessing | Hero packs supplies |
| Baseline Models | Hero tries basic weapons |
| Feature Engineering | Hero forges a magic sword |
| Hyperparameter Tuning | Hero trains with the sword |
| Final Model | Hero defeats the dragon |
| MLflow Registry | Hero's tale is recorded in history |

---

## Summary: Key Takeaways

### 🎯 What We Built

A complete ML pipeline that:
1. **Understands** the data through EDA
2. **Cleans** it through preprocessing
3. **Enriches** it through feature engineering
4. **Experiments** with multiple models
5. **Optimizes** the best model
6. **Tracks** everything in MLflow

### 💡 Key Concepts Learned

| Concept | One-Liner |
|---------|-----------|
| Random Seed | "Same input, same output" for reproducibility |
| EDA | Medical checkup for your data |
| Preprocessing | Cleaning vegetables before cooking |
| Feature Engineering | Turning flour, eggs, sugar into a cake |
| Baseline Models | Interviewing multiple candidates |
| Hyperparameter Tuning | Finding the perfect oven temperature |
| MLflow | A lab notebook for scientists |
| Model Registry | Git for ML models |

### 🚀 What's Next?

With the trained model, you can:
1. **Predict** churn probability for new customers
2. **Target** high-risk customers with retention offers
3. **Monitor** model performance over time
4. **Retrain** when performance degrades

---

## Appendix: Quick Reference

### Running the Pipeline

```bash
# Activate environment
source venv/bin/activate

# Run the experiment
python3 churn_experiment.py

# View results
mlflow ui --backend-store-uri mlflow_runs
```

### Key Files

| File | Purpose |
|------|---------|
| `churn_experiment.py` | Main pipeline code |
| `generate_dataset.py` | Creates synthetic data |
| `telco_churn.csv` | The dataset |
| `artifacts/` | Saved plots and models |
| `mlflow_runs/` | Experiment tracking data |

### Metrics Cheat Sheet

```
Accuracy = (TP + TN) / Total
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

*📝 Documentation created for educational purposes. Happy learning!*

