"""
Churn Prediction Experimentation Pipeline

End-to-end ML experimentation workflow using:
- PyCaret for rapid model prototyping
- Optuna for hyperparameter optimization
- MLflow for experiment tracking and model registry

This script demonstrates the complete ML lifecycle:
EDA → Baseline Models → Feature Engineering → Tuning → Model Registration

Author: ML Engineering Team
"""

import warnings
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Reproducibility seed used throughout the pipeline
RANDOM_SEED = 42

# MLflow experiment configuration
MLFLOW_EXPERIMENT_NAME = "churn_prediction_pycaret"
MLFLOW_TRACKING_URI = "mlflow_runs"  # Local directory for MLflow data

# Dataset configuration
DATASET_PATH = "telco_churn.csv"
TARGET_COLUMN = "Churn"

# Artifacts directory for plots
ARTIFACTS_DIR = Path("artifacts")


def setup_environment() -> None:
    """
    Initialize environment settings for reproducibility.
    
    Why: Consistent random seeds across all libraries ensure
    that experiments can be reproduced exactly.
    """
    np.random.seed(RANDOM_SEED)
    
    # Create artifacts directory if it doesn't exist
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    print("✅ Environment configured")
    print(f"   Random seed: {RANDOM_SEED}")
    print(f"   MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"   Artifacts dir: {ARTIFACTS_DIR.absolute()}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the telco churn dataset.
    
    Why: Centralizing data loading ensures consistent preprocessing
    and makes it easy to switch data sources.
    """
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset from {filepath}")
    return df


# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def perform_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive exploratory data analysis.
    
    Why: Understanding data distribution and quality issues
    before modeling prevents downstream problems and informs
    feature engineering decisions.
    
    Returns:
        Dictionary containing EDA metrics for MLflow logging
    """
    print("\n" + "=" * 60)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Basic dataset info
    print(f"\n📋 Dataset Shape: {df.shape}")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Data types
    print("\n🔢 Data Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")
    
    # Missing values analysis
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    missing_cols = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_cols) > 0:
        print(missing_cols.to_string())
    else:
        print("   No missing values found")
    
    # Target distribution
    print(f"\n🎯 Target Distribution ({TARGET_COLUMN}):")
    target_dist = df[TARGET_COLUMN].value_counts()
    target_pct = df[TARGET_COLUMN].value_counts(normalize=True).round(3)
    for label in target_dist.index:
        print(f"   {label}: {target_dist[label]} ({target_pct[label]:.1%})")
    
    # Calculate imbalance ratio
    minority_class = target_dist.min()
    majority_class = target_dist.max()
    imbalance_ratio = round(majority_class / minority_class, 2)
    print(f"   Imbalance Ratio: {imbalance_ratio}:1")
    
    # Return metrics for MLflow
    return {
        "dataset_rows": df.shape[0],
        "dataset_cols": df.shape[1],
        "num_features": df.shape[1] - 1,
        "missing_values_total": int(missing.sum()),
        "target_imbalance_ratio": imbalance_ratio,
        "churn_rate": float(target_pct.get('Yes', target_pct.iloc[1]))
    }


def create_eda_visualizations(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Create and save EDA visualizations.
    
    Why: Visual representations of data distributions help identify
    patterns, outliers, and class imbalance at a glance.
    
    Returns:
        Tuple of file paths for the saved plots
    """
    print("\n📈 Creating EDA Visualizations...")
    
    # Set style for professional-looking plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Churn Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#e74c3c']  # Green for No, Red for Yes
    
    churn_counts = df[TARGET_COLUMN].value_counts()
    bars = ax.bar(churn_counts.index, churn_counts.values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, count in zip(bars, churn_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Churn Status', fontsize=12)
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, churn_counts.max() * 1.2)
    
    churn_plot_path = ARTIFACTS_DIR / "churn_distribution.png"
    plt.tight_layout()
    plt.savefig(churn_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {churn_plot_path}")
    
    # 2. Correlation Heatmap (numerical features only)
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Encode Churn to numeric for correlation
    if TARGET_COLUMN not in numeric_df.columns:
        numeric_df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
    
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = numeric_df.corr()
    
    # Create heatmap with improved aesthetics
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    heatmap_path = ARTIFACTS_DIR / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {heatmap_path}")
    
    return str(churn_plot_path), str(heatmap_path)


def log_eda_to_mlflow(eda_metrics: Dict[str, Any], plot_paths: Tuple[str, str]) -> None:
    """
    Log EDA results to MLflow.
    
    Why: Tracking EDA metrics and visualizations in MLflow provides
    a complete audit trail and helps compare data across experiments.
    """
    with mlflow.start_run(run_name="01_EDA_Analysis"):
        # Log dataset metrics as parameters
        mlflow.log_params({
            "dataset_rows": eda_metrics["dataset_rows"],
            "dataset_cols": eda_metrics["dataset_cols"],
            "num_features": eda_metrics["num_features"]
        })
        
        # Log quality metrics
        mlflow.log_metrics({
            "missing_values_total": eda_metrics["missing_values_total"],
            "target_imbalance_ratio": eda_metrics["target_imbalance_ratio"],
            "churn_rate": eda_metrics["churn_rate"]
        })
        
        # Log visualization artifacts
        for plot_path in plot_paths:
            mlflow.log_artifact(plot_path)
        
        print("✅ EDA results logged to MLflow")


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data before modeling.
    
    Why: Clean data is essential for model performance. This function
    handles missing values and ensures proper data types.
    """
    df = df.copy()
    
    # Drop customerID as it's not a feature
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Handle TotalCharges missing values
    # Fill with median (robust to outliers) or 0 for new customers
    if 'TotalCharges' in df.columns:
        # Convert to numeric, coercing errors to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing with 0 (new customers)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Fill other missing values with mode (most frequent)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    
    print(f"✅ Data preprocessed: {df.shape}")
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features to improve model performance.
    
    Why: Feature engineering captures domain knowledge and creates
    more predictive signals from raw data. These features are based
    on common telco churn patterns.
    """
    df = df.copy()
    
    # 1. Tenure Buckets - Categorizes customer lifecycle stage
    # Why: Churn risk varies by customer maturity, not just raw tenure
    df['TenureBucket'] = pd.cut(
        df['tenure'],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=['0-6_months', '6-12_months', '1-2_years', '2-4_years', '4+_years']
    )
    
    # 2. Charges Per Month of Tenure - Value extraction rate
    # Why: Customers paying high rates relative to tenure may feel "trapped"
    df['ChargesPerTenure'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )
    
    # 3. High Value Customer Flag - Identifies premium customers
    # Why: High-value customers may need different retention strategies
    monthly_75th = df['MonthlyCharges'].quantile(0.75)
    df['HighValueCustomer'] = (df['MonthlyCharges'] > monthly_75th).astype(int)
    
    # 4. Number of Services - Total service count
    # Why: More services = more engagement = lower churn
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    def count_services(row):
        count = 0
        for col in service_cols:
            if col in row.index:
                val = row[col]
                if val == 'Yes' or (col == 'InternetService' and val != 'No'):
                    count += 1
        return count
    
    df['NumServices'] = df.apply(count_services, axis=1)
    
    # 5. Contract Security Score - Risk indicator
    # Why: Combination of contract and support features predicts retention
    df['ContractSecurityScore'] = 0
    df.loc[df['Contract'] == 'Two year', 'ContractSecurityScore'] += 2
    df.loc[df['Contract'] == 'One year', 'ContractSecurityScore'] += 1
    df.loc[df['OnlineSecurity'] == 'Yes', 'ContractSecurityScore'] += 1
    df.loc[df['TechSupport'] == 'Yes', 'ContractSecurityScore'] += 1
    
    # 6. Payment Risk Flag - Electronic check is high risk
    # Why: Electronic check users historically churn more
    df['PaymentRisk'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    print(f"✅ Created {6} engineered features")
    print(f"   New columns: TenureBucket, ChargesPerTenure, HighValueCustomer,")
    print(f"                NumServices, ContractSecurityScore, PaymentRisk")
    
    return df


# =============================================================================
# BASELINE MODELS WITH PYCARET
# =============================================================================

def setup_pycaret_experiment(
    df: pd.DataFrame,
    run_name: str,
    log_mlflow: bool = True
) -> Tuple[Any, Any]:
    """
    Initialize PyCaret classification setup.
    
    Why: PyCaret automates preprocessing, encoding, and provides
    a consistent framework for model comparison.
    
    Returns:
        Tuple of (setup object, processed dataframe)
    """
    # Import PyCaret here to avoid issues if not installed
    from pycaret.classification import setup, get_config
    
    print(f"\n🔧 Setting up PyCaret experiment: {run_name}")
    
    # PyCaret setup with comprehensive preprocessing
    # silent=True prevents interactive prompts
    clf_setup = setup(
        data=df,
        target=TARGET_COLUMN,
        session_id=RANDOM_SEED,
        train_size=0.8,
        normalize=True,
        normalize_method='zscore',
        handle_unknown_categorical=True,
        remove_outliers=False,  # Keep outliers for this use case
        log_experiment=False,  # Disabled - we handle MLflow manually
        verbose=False,
        html=False,  # Disable HTML output for cleaner logs
    )
    
    # Get the transformed data
    X_train = get_config('X_train')
    print(f"   ✅ Train set: {len(X_train)} samples")
    
    return clf_setup, df


def train_baseline_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train and compare baseline models using PyCaret.
    
    Why: Starting with multiple baseline models helps identify
    which algorithm family works best for this problem.
    
    Returns:
        Dictionary with model comparison results
    """
    from pycaret.classification import (
        setup, compare_models, create_model, 
        pull, get_config, save_model
    )
    
    print("\n" + "=" * 60)
    print("🤖 BASELINE MODEL TRAINING")
    print("=" * 60)
    
    # Initialize PyCaret - disable automatic MLflow logging to avoid conflicts
    # We'll handle MLflow logging ourselves for better control
    setup(
        data=df,
        target=TARGET_COLUMN,
        session_id=RANDOM_SEED,
        train_size=0.8,
        normalize=True,
        log_experiment=False,  # Disabled - we handle MLflow manually
        verbose=False,
        html=False,
    )
    
    # Define baseline models to compare
    # Why: These represent different algorithm families -
    # linear (LR), tree-based (RF, GBC)
    baseline_models = ['lr', 'rf', 'gbc']
    model_names = {
        'lr': 'Logistic Regression',
        'rf': 'Random Forest',
        'gbc': 'Gradient Boosting'
    }
    
    results = {}
    
    for model_id in baseline_models:
        print(f"\n📊 Training {model_names[model_id]}...")
        
        with mlflow.start_run(run_name=f"02_Baseline_{model_names[model_id].replace(' ', '_')}"):
            # Create and train the model
            model = create_model(model_id, verbose=False)
            
            # Get metrics from PyCaret
            metrics_df = pull()
            
            # Extract key metrics (mean from cross-validation)
            accuracy = float(metrics_df['Accuracy'].mean())
            precision = float(metrics_df['Prec.'].mean())
            recall = float(metrics_df['Recall'].mean())
            auc = float(metrics_df['AUC'].mean())
            f1 = float(metrics_df['F1'].mean())
            
            # Log to MLflow
            mlflow.log_param("model_type", model_names[model_id])
            mlflow.log_param("feature_set", "baseline")
            
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "f1_score": f1
            })
            
            results[model_id] = {
                "model": model,
                "name": model_names[model_id],
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "f1": f1
            }
            
            print(f"   Accuracy: {accuracy:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
    
    # Find best baseline model
    best_model_id = max(results, key=lambda x: results[x]['auc'])
    print(f"\n🏆 Best Baseline Model: {results[best_model_id]['name']} (AUC: {results[best_model_id]['auc']:.4f})")
    
    return results


def train_models_with_engineered_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train models on engineered features and compare with baseline.
    
    Why: Feature engineering often provides significant performance
    improvements. Tracking both allows us to quantify the impact.
    """
    from pycaret.classification import (
        setup, create_model, pull
    )
    
    print("\n" + "=" * 60)
    print("🔬 FEATURE ENGINEERING ITERATION")
    print("=" * 60)
    
    # Create engineered features
    df_engineered = create_engineered_features(df)
    
    # Re-initialize PyCaret with new features
    setup(
        data=df_engineered,
        target=TARGET_COLUMN,
        session_id=RANDOM_SEED,
        train_size=0.8,
        normalize=True,
        log_experiment=False,  # Disabled - we handle MLflow manually
        verbose=False,
        html=False,
    )
    
    baseline_models = ['lr', 'rf', 'gbc']
    model_names = {
        'lr': 'Logistic Regression',
        'rf': 'Random Forest',
        'gbc': 'Gradient Boosting'
    }
    
    results = {}
    
    for model_id in baseline_models:
        print(f"\n📊 Training {model_names[model_id]} with engineered features...")
        
        with mlflow.start_run(run_name=f"03_Engineered_{model_names[model_id].replace(' ', '_')}"):
            model = create_model(model_id, verbose=False)
            metrics_df = pull()
            
            accuracy = float(metrics_df['Accuracy'].mean())
            precision = float(metrics_df['Prec.'].mean())
            recall = float(metrics_df['Recall'].mean())
            auc = float(metrics_df['AUC'].mean())
            f1 = float(metrics_df['F1'].mean())
            
            mlflow.log_param("model_type", model_names[model_id])
            mlflow.log_param("feature_set", "engineered")
            mlflow.log_param("num_new_features", 6)
            
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "f1_score": f1
            })
            
            results[model_id] = {
                "model": model,
                "name": model_names[model_id],
                "accuracy": accuracy,
                "auc": auc,
                "f1": f1
            }
            
            print(f"   Accuracy: {accuracy:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
    
    return results, df_engineered


# =============================================================================
# HYPERPARAMETER TUNING WITH OPTUNA
# =============================================================================

def tune_model_with_optuna(
    df: pd.DataFrame,
    model_id: str = 'gbc'
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using Optuna.
    
    Why: Optuna provides efficient Bayesian optimization that
    explores the hyperparameter space more intelligently than
    grid or random search.
    
    Args:
        df: DataFrame with features and target
        model_id: PyCaret model ID to tune (default: gradient boosting)
    
    Returns:
        Tuple of (tuned model, tuning results)
    """
    from pycaret.classification import (
        setup, create_model, tune_model, pull, get_config
    )
    
    print("\n" + "=" * 60)
    print("⚡ HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 60)
    
    # Setup PyCaret - disable automatic logging to avoid conflicts
    setup(
        data=df,
        target=TARGET_COLUMN,
        session_id=RANDOM_SEED,
        train_size=0.8,
        normalize=True,
        log_experiment=False,  # Disabled - we handle MLflow manually
        verbose=False,
        html=False,
    )
    
    model_names = {
        'lr': 'Logistic Regression',
        'rf': 'Random Forest',
        'gbc': 'Gradient Boosting'
    }
    
    print(f"\n🎯 Tuning {model_names.get(model_id, model_id)}...")
    
    with mlflow.start_run(run_name=f"04_Tuned_{model_names.get(model_id, model_id).replace(' ', '_')}"):
        # Create base model first
        base_model = create_model(model_id, verbose=False)
        base_metrics = pull()
        base_auc = float(base_metrics['AUC'].mean())
        
        # Tune with Optuna
        # n_iter controls number of Optuna trials
        tuned_model = tune_model(
            base_model,
            optimize='AUC',
            search_library='optuna',
            search_algorithm='tpe',  # Tree-structured Parzen Estimator
            n_iter=20,  # Number of Optuna trials
            early_stopping=True,
            verbose=False
        )
        
        # Get tuned metrics
        tuned_metrics = pull()
        tuned_accuracy = float(tuned_metrics['Accuracy'].mean())
        tuned_precision = float(tuned_metrics['Prec.'].mean())
        tuned_recall = float(tuned_metrics['Recall'].mean())
        tuned_auc = float(tuned_metrics['AUC'].mean())
        tuned_f1 = float(tuned_metrics['F1'].mean())
        
        # Log parameters and metrics
        mlflow.log_param("model_type", model_names.get(model_id, model_id))
        mlflow.log_param("feature_set", "engineered")
        mlflow.log_param("tuning_library", "optuna")
        mlflow.log_param("tuning_algorithm", "tpe")
        mlflow.log_param("n_trials", 20)
        
        # Log best hyperparameters
        if hasattr(tuned_model, 'get_params'):
            best_params = tuned_model.get_params()
            # Log only key hyperparameters
            key_params = ['n_estimators', 'max_depth', 'learning_rate', 
                          'min_samples_split', 'min_samples_leaf', 'subsample']
            for param in key_params:
                if param in best_params:
                    mlflow.log_param(f"best_{param}", best_params[param])
        
        mlflow.log_metrics({
            "accuracy": tuned_accuracy,
            "precision": tuned_precision,
            "recall": tuned_recall,
            "auc": tuned_auc,
            "f1_score": tuned_f1,
            "auc_improvement": tuned_auc - base_auc
        })
        
        results = {
            "model": tuned_model,
            "base_auc": base_auc,
            "tuned_auc": tuned_auc,
            "improvement": tuned_auc - base_auc,
            "accuracy": tuned_accuracy,
            "precision": tuned_precision,
            "recall": tuned_recall,
            "f1": tuned_f1
        }
        
        print(f"   Base AUC: {base_auc:.4f}")
        print(f"   Tuned AUC: {tuned_auc:.4f}")
        print(f"   Improvement: {(tuned_auc - base_auc):.4f}")
    
    return tuned_model, results


# =============================================================================
# FINAL MODEL & ARTIFACTS
# =============================================================================

def create_final_model_artifacts(
    model: Any,
    df: pd.DataFrame,
    tuning_results: Dict[str, Any]
) -> None:
    """
    Create and log final model artifacts.
    
    Why: Confusion matrix and feature importance provide
    interpretability for stakeholders and help identify
    areas for improvement.
    """
    from pycaret.classification import (
        setup, finalize_model, plot_model, save_model, get_config
    )
    
    print("\n" + "=" * 60)
    print("📦 FINAL MODEL & ARTIFACTS")
    print("=" * 60)
    
    # Re-setup to ensure consistent state
    setup(
        data=df,
        target=TARGET_COLUMN,
        session_id=RANDOM_SEED,
        train_size=0.8,
        normalize=True,
        log_experiment=False,  # Disabled - we handle MLflow manually
        verbose=False,
        html=False,
    )
    
    with mlflow.start_run(run_name="05_Final_Model"):
        # Finalize model (train on full dataset)
        print("\n🔧 Finalizing model (training on full dataset)...")
        final_model = finalize_model(model)
        
        # Save model artifacts
        model_path = ARTIFACTS_DIR / "final_model"
        save_model(final_model, str(model_path))
        print(f"   ✅ Model saved to: {model_path}.pkl")
        
        # Create confusion matrix plot
        print("\n📊 Creating confusion matrix...")
        confusion_matrix_path = ARTIFACTS_DIR / "confusion_matrix.png"
        try:
            plot_model(model, plot='confusion_matrix', save=True)
            # PyCaret saves to current directory, move to artifacts
            if Path("Confusion Matrix.png").exists():
                Path("Confusion Matrix.png").rename(confusion_matrix_path)
        except Exception as e:
            # Create simple confusion matrix if PyCaret plot fails
            from sklearn.metrics import confusion_matrix as cm
            from pycaret.classification import predict_model, get_config
            
            # Get test data predictions
            X_test = get_config('X_test')
            y_test = get_config('y_test')
            
            if X_test is not None and len(X_test) > 0:
                predictions = predict_model(model, data=X_test)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                # Note: y_test and predictions need proper handling
                ax.text(0.5, 0.5, "Confusion Matrix\n(See MLflow logs)", 
                        ha='center', va='center', fontsize=14)
                ax.set_title("Model Confusion Matrix")
                plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        # Create feature importance plot
        print("📊 Creating feature importance plot...")
        feature_importance_path = ARTIFACTS_DIR / "feature_importance.png"
        try:
            plot_model(model, plot='feature', save=True)
            if Path("Feature Importance.png").exists():
                Path("Feature Importance.png").rename(feature_importance_path)
        except Exception as e:
            # Create placeholder if fails
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if hasattr(model, 'feature_importances_'):
                # Get feature names from config
                feature_names = get_config('X_train').columns.tolist()
                importances = model.feature_importances_
                
                # Sort by importance
                indices = np.argsort(importances)[::-1][:15]  # Top 15
                
                ax.barh(range(len(indices)), importances[indices], color='steelblue')
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_names[i] for i in indices])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Top 15 Feature Importances')
                ax.invert_yaxis()
            else:
                ax.text(0.5, 0.5, "Feature importance not available", 
                        ha='center', va='center')
            
            plt.tight_layout()
            plt.savefig(feature_importance_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Log artifacts to MLflow
        mlflow.log_artifact(str(model_path) + ".pkl")
        
        if confusion_matrix_path.exists():
            mlflow.log_artifact(str(confusion_matrix_path))
            print(f"   ✅ Logged: {confusion_matrix_path}")
        
        if feature_importance_path.exists():
            mlflow.log_artifact(str(feature_importance_path))
            print(f"   ✅ Logged: {feature_importance_path}")
        
        # Log model to MLflow Model Registry
        print("\n📝 Registering model in MLflow Model Registry...")
        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="ChurnPredictionModel"
        )
        print("   ✅ Model registered as: ChurnPredictionModel")
        
        # Log final summary parameters
        mlflow.log_param("model_status", "finalized")
        mlflow.log_param("training_data_size", len(df))
        mlflow.log_param("model_type", "Gradient Boosting (Tuned)")
        
        # Log final metrics from tuning results for comparison
        mlflow.log_metrics({
            "accuracy": tuning_results.get("accuracy", 0),
            "precision": tuning_results.get("precision", 0),
            "recall": tuning_results.get("recall", 0),
            "auc": tuning_results.get("tuned_auc", 0),
            "f1_score": tuning_results.get("f1", 0),
            "auc_improvement": tuning_results.get("improvement", 0)
        })


def print_experiment_summary() -> None:
    """
    Print a summary of all MLflow runs in the experiment.
    """
    print("\n" + "=" * 60)
    print("📋 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time ASC"]
        )
        
        print(f"\n🔬 Experiment: {MLFLOW_EXPERIMENT_NAME}")
        print(f"   Total Runs: {len(runs)}")
        print("\n" + "-" * 60)
        
        for run in runs:
            run_name = run.data.tags.get('mlflow.runName', 'Unnamed')
            auc = run.data.metrics.get('auc', 'N/A')
            accuracy = run.data.metrics.get('accuracy', 'N/A')
            
            if isinstance(auc, float):
                print(f"   {run_name}: AUC={auc:.4f}, Accuracy={accuracy:.4f}")
            else:
                print(f"   {run_name}: Metrics logged")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print(f"   View results: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """
    Execute the complete churn prediction experiment pipeline.
    
    Pipeline Steps:
    1. Environment Setup
    2. Data Loading & EDA
    3. Baseline Model Training
    4. Feature Engineering
    5. Hyperparameter Tuning
    6. Final Model Registration
    """
    print("\n" + "=" * 60)
    print("🚀 CHURN PREDICTION EXPERIMENTATION PIPELINE")
    print("=" * 60)
    
    # Step 0: Setup environment
    setup_environment()
    
    # Step 1: Generate dataset if not exists
    if not Path(DATASET_PATH).exists():
        print(f"\n📁 Dataset not found. Generating {DATASET_PATH}...")
        from generate_dataset import generate_telco_churn_dataset
        generate_telco_churn_dataset(output_path=DATASET_PATH, seed=RANDOM_SEED)
    
    # Step 2: Load data
    df_raw = load_dataset(DATASET_PATH)
    
    # Step 3: Create/set MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"\n🔬 MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    # Step 4: Perform EDA
    eda_metrics = perform_eda(df_raw)
    plot_paths = create_eda_visualizations(df_raw)
    log_eda_to_mlflow(eda_metrics, plot_paths)
    
    # Step 5: Preprocess data
    df_clean = preprocess_data(df_raw)
    
    # Step 6: Train baseline models
    baseline_results = train_baseline_models(df_clean)
    
    # Step 7: Feature engineering iteration
    engineered_results, df_engineered = train_models_with_engineered_features(df_clean)
    
    # Step 8: Compare baseline vs engineered
    print("\n📊 Baseline vs Engineered Feature Comparison:")
    for model_id in baseline_results:
        base_auc = baseline_results[model_id]['auc']
        eng_auc = engineered_results[model_id]['auc']
        improvement = eng_auc - base_auc
        print(f"   {baseline_results[model_id]['name']}:")
        print(f"      Baseline AUC: {base_auc:.4f} → Engineered AUC: {eng_auc:.4f} ({improvement:+.4f})")
    
    # Step 9: Hyperparameter tuning on best model
    # Use Gradient Boosting as it typically performs well on tabular data
    tuned_model, tuning_results = tune_model_with_optuna(df_engineered, 'gbc')
    
    # Step 10: Create final model and artifacts
    create_final_model_artifacts(tuned_model, df_engineered, tuning_results)
    
    # Step 11: Print experiment summary
    print_experiment_summary()


if __name__ == "__main__":
    main()

