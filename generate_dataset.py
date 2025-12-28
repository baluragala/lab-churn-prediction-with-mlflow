"""
Synthetic Telco Churn Dataset Generator

Creates a realistic telco customer churn dataset with:
- Numerical features (tenure, monthly charges, etc.)
- Categorical features (contract type, payment method, etc.)
- Controlled class imbalance (~26% churn rate, similar to real-world)
- Some missing values for realistic preprocessing

Author: ML Engineering Team
"""

import numpy as np
import pandas as pd
from pathlib import Path


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def generate_customer_demographics(n_samples: int) -> dict:
    """
    Generate customer demographic features.
    
    Why: Demographics influence churn behavior - senior citizens 
    and customers without dependents tend to churn more.
    """
    return {
        'customerID': [f'CUST_{i:06d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
    }


def generate_service_features(n_samples: int) -> dict:
    """
    Generate service-related features.
    
    Why: Service types strongly correlate with churn - 
    fiber optic users and those without tech support churn more.
    """
    # Phone service
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    
    # Multiple lines depends on phone service
    multiple_lines = np.where(
        phone_service == 'No',
        'No phone service',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.42, 0.58])
    )
    
    # Internet service type
    internet_service = np.random.choice(
        ['DSL', 'Fiber optic', 'No'], 
        n_samples, 
        p=[0.34, 0.44, 0.22]
    )
    
    # Additional services depend on internet
    def get_internet_dependent_service(internet_svc, yes_prob=0.40):
        return np.where(
            internet_svc == 'No',
            'No internet service',
            np.random.choice(['Yes', 'No'], n_samples, p=[yes_prob, 1 - yes_prob])
        )
    
    return {
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': get_internet_dependent_service(internet_service, 0.29),
        'OnlineBackup': get_internet_dependent_service(internet_service, 0.34),
        'DeviceProtection': get_internet_dependent_service(internet_service, 0.34),
        'TechSupport': get_internet_dependent_service(internet_service, 0.29),
        'StreamingTV': get_internet_dependent_service(internet_service, 0.38),
        'StreamingMovies': get_internet_dependent_service(internet_service, 0.39),
    }


def generate_account_features(n_samples: int) -> dict:
    """
    Generate account and billing features.
    
    Why: Contract type and payment method are among the strongest 
    churn predictors - month-to-month contracts churn significantly more.
    """
    # Tenure in months (0-72 months typical range)
    tenure = np.random.exponential(scale=20, size=n_samples).astype(int)
    tenure = np.clip(tenure, 0, 72)
    
    # Contract type - shorter contracts = higher churn
    contract = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        n_samples,
        p=[0.55, 0.21, 0.24]
    )
    
    # Paperless billing
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
    
    # Payment method
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n_samples,
        p=[0.34, 0.23, 0.22, 0.21]
    )
    
    # Monthly charges (based on services, roughly $20-120)
    base_charge = np.random.uniform(18, 25, n_samples)
    service_multiplier = np.random.uniform(1.0, 4.5, n_samples)
    monthly_charges = np.round(base_charge * service_multiplier, 2)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    
    # Total charges = monthly * tenure (with some variance)
    total_charges = np.round(monthly_charges * (tenure + 1) * np.random.uniform(0.95, 1.05, n_samples), 2)
    
    return {
        'tenure': tenure,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }


def generate_churn_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Generate churn labels based on feature correlations.
    
    Why: Churn is not random - it's influenced by tenure, contract type,
    monthly charges, and service quality. This function creates realistic
    churn probabilities based on these factors.
    """
    n_samples = len(df)
    base_churn_prob = np.full(n_samples, 0.15)
    
    # Short tenure increases churn probability
    base_churn_prob += np.where(df['tenure'] < 6, 0.20, 0)
    base_churn_prob += np.where((df['tenure'] >= 6) & (df['tenure'] < 12), 0.10, 0)
    
    # Month-to-month contracts churn more
    base_churn_prob += np.where(df['Contract'] == 'Month-to-month', 0.15, 0)
    base_churn_prob -= np.where(df['Contract'] == 'Two year', 0.10, 0)
    
    # High monthly charges increase churn
    base_churn_prob += np.where(df['MonthlyCharges'] > 80, 0.10, 0)
    
    # Electronic check payment correlates with churn
    base_churn_prob += np.where(df['PaymentMethod'] == 'Electronic check', 0.08, 0)
    
    # Fiber optic without tech support churns more
    fiber_no_support = (df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'No')
    base_churn_prob += np.where(fiber_no_support, 0.12, 0)
    
    # Senior citizens churn slightly more
    base_churn_prob += np.where(df['SeniorCitizen'] == 1, 0.05, 0)
    
    # Customers with dependents/partners churn less
    base_churn_prob -= np.where(df['Partner'] == 'Yes', 0.05, 0)
    base_churn_prob -= np.where(df['Dependents'] == 'Yes', 0.05, 0)
    
    # Clip probabilities to valid range
    base_churn_prob = np.clip(base_churn_prob, 0.05, 0.85)
    
    # Generate binary churn labels
    churn = np.random.binomial(1, base_churn_prob)
    return np.where(churn == 1, 'Yes', 'No')


def introduce_missing_values(df: pd.DataFrame, missing_rate: float = 0.02) -> pd.DataFrame:
    """
    Introduce realistic missing values.
    
    Why: Real-world data always has missing values. TotalCharges often has 
    blanks for new customers with tenure=0.
    """
    df = df.copy()
    
    # TotalCharges blank for some new customers (this is realistic)
    new_customer_mask = df['tenure'] == 0
    blank_indices = df[new_customer_mask].sample(frac=0.8, random_state=42).index
    df.loc[blank_indices, 'TotalCharges'] = np.nan
    
    # Random missing in a few columns
    for col in ['OnlineSecurity', 'TechSupport', 'StreamingTV']:
        if col in df.columns:
            missing_idx = df.sample(frac=missing_rate, random_state=42).index
            df.loc[missing_idx[:int(len(missing_idx) * 0.3)], col] = np.nan
    
    return df


def generate_telco_churn_dataset(
    n_samples: int = 7043,
    output_path: str = 'telco_churn.csv',
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate complete synthetic telco churn dataset.
    
    Args:
        n_samples: Number of customer records to generate
        output_path: Path to save the CSV file
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with all features and churn labels
    """
    set_random_seed(seed)
    
    # Generate all feature groups
    demographics = generate_customer_demographics(n_samples)
    services = generate_service_features(n_samples)
    account = generate_account_features(n_samples)
    
    # Combine into DataFrame
    df = pd.DataFrame({**demographics, **services, **account})
    
    # Generate correlated churn labels
    df['Churn'] = generate_churn_labels(df)
    
    # Introduce realistic missing values
    df = introduce_missing_values(df)
    
    # Save to CSV
    output_file = Path(output_path)
    df.to_csv(output_file, index=False)
    print(f"✅ Generated dataset with {n_samples} samples")
    print(f"📁 Saved to: {output_file.absolute()}")
    print(f"\n📊 Churn Distribution:")
    print(df['Churn'].value_counts(normalize=True).round(3))
    
    return df


if __name__ == '__main__':
    # Generate the dataset when script is run directly
    df = generate_telco_churn_dataset()
    print(f"\n📋 Dataset Shape: {df.shape}")
    print(f"🔢 Features: {list(df.columns)}")

