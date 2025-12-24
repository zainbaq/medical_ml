"""
Pytest fixtures for cardiovascular disease prediction tests
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "training"))
sys.path.insert(0, str(project_root / "backend"))


@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return {
        "age_years": 50.0,
        "gender": 1,
        "height": 170.0,
        "weight": 75.0,
        "ap_hi": 130,
        "ap_lo": 85,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }


@pytest.fixture
def sample_patient_data_v2():
    """Sample patient data with extended features for v2 API"""
    return {
        "age_years": 50.0,
        "gender": 1,
        "height": 170.0,
        "weight": 75.0,
        "ap_hi": 130,
        "ap_lo": 85,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        # Extended features
        "hdl_cholesterol": 55.0,
        "ldl_cholesterol": 120.0,
        "total_cholesterol": 200.0,
        "triglycerides": 150.0
    }


@pytest.fixture
def high_risk_patient():
    """High risk patient data"""
    return {
        "age_years": 65.0,
        "gender": 2,
        "height": 165.0,
        "weight": 100.0,
        "ap_hi": 170,
        "ap_lo": 105,
        "cholesterol": 3,
        "gluc": 3,
        "smoke": 1,
        "alco": 1,
        "active": 0
    }


@pytest.fixture
def low_risk_patient():
    """Low risk patient data"""
    return {
        "age_years": 30.0,
        "gender": 1,
        "height": 175.0,
        "weight": 70.0,
        "ap_hi": 115,
        "ap_lo": 75,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }


@pytest.fixture
def sample_training_data():
    """Sample training data for feature engineering tests"""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame({
        'id': range(n_samples),
        'age': np.random.randint(10000, 25000, n_samples),  # days
        'gender': np.random.choice([1, 2], n_samples),
        'height': np.random.normal(170, 10, n_samples),
        'weight': np.random.normal(75, 15, n_samples),
        'ap_hi': np.random.normal(130, 20, n_samples).astype(int),
        'ap_lo': np.random.normal(85, 15, n_samples).astype(int),
        'cholesterol': np.random.choice([1, 2, 3], n_samples),
        'gluc': np.random.choice([1, 2, 3], n_samples),
        'smoke': np.random.choice([0, 1], n_samples),
        'alco': np.random.choice([0, 1], n_samples),
        'active': np.random.choice([0, 1], n_samples),
        'cardio': np.random.choice([0, 1], n_samples)
    })

    # Ensure valid blood pressure (systolic > diastolic)
    data['ap_hi'] = np.maximum(data['ap_hi'], data['ap_lo'] + 10)

    # Preprocess: convert age to years and compute BMI
    data['age_years'] = (data['age'] / 365.25).round(1)
    data['bmi'] = (data['weight'] / ((data['height'] / 100) ** 2)).round(2)

    return data


@pytest.fixture
def sample_uci_data():
    """Sample UCI Heart Disease formatted data"""
    return pd.DataFrame({
        'age': [63, 67, 67, 37, 41],
        'sex': [1, 1, 1, 1, 0],
        'cp': [1, 4, 4, 3, 2],
        'trestbps': [145, 160, 120, 130, 130],
        'chol': [233, 286, 229, 250, 204],
        'fbs': [1, 0, 0, 0, 0],
        'restecg': [2, 2, 2, 0, 2],
        'thalach': [150, 108, 129, 187, 172],
        'exang': [0, 1, 1, 0, 0],
        'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
        'slope': [3, 2, 2, 3, 1],
        'ca': [0, 3, 2, 0, 0],
        'thal': [6, 3, 7, 3, 3],
        'target': [0, 2, 1, 0, 0]  # UCI uses 'target' column
    })


@pytest.fixture
def performance_targets():
    """Performance targets for model validation"""
    return {
        'roc_auc': 0.82,
        'accuracy': 0.78,
        'sensitivity': 0.75,
        'specificity': 0.75,
        'brier_score': 0.18
    }


@pytest.fixture
def subgroup_thresholds():
    """Subgroup fairness thresholds"""
    return {
        'roc_auc': 0.05,
        'accuracy': 0.05,
        'recall': 0.10,
        'precision': 0.10
    }


@pytest.fixture
def api_base_url():
    """Base URL for API tests"""
    return "http://localhost:8003"
