"""
Training configuration for cardiovascular disease prediction model
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "cardio_train.csv"

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Model configurations
MODELS_CONFIG = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'max_iter': [1000]
    }
}

# Feature engineering
OUTLIER_THRESHOLDS = {
    'ap_hi': (80, 200),      # Systolic BP range
    'ap_lo': (60, 130),      # Diastolic BP range
    'height': (140, 210),    # Height in cm
    'weight': (40, 200),     # Weight in kg
}

# Feature engineering options
USE_IMPROVED_FEATURES = True  # Set to True to use improved feature engineering
INCLUDE_EXPERIMENTAL_FEATURES = False  # Set to True to include interaction/polynomial features
REMOVE_WEIGHT = True  # Remove weight (redundant with BMI)

# Original feature names
FEATURE_COLUMNS_ORIGINAL = [
    'age_years', 'gender', 'height', 'weight', 'bmi',
    'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active'
]

# Improved feature names (high priority features)
FEATURE_COLUMNS_IMPROVED = [
    'age_years', 'gender', 'height', 'bmi',
    'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active',
    # New engineered features
    'pulse_pressure',
    'mean_arterial_pressure',
    'hypertension_stage',
    'bmi_category',
    'age_group',
    'health_risk_composite',
    'lifestyle_risk_score'
]

# Experimental features (optional)
FEATURE_COLUMNS_EXPERIMENTAL = FEATURE_COLUMNS_IMPROVED + [
    'age_bmi_interaction',
    'bp_bmi_interaction',
    'age_squared',
    'bmi_squared'
]

# Default feature columns (dynamically set based on config)
if USE_IMPROVED_FEATURES:
    if INCLUDE_EXPERIMENTAL_FEATURES:
        FEATURE_COLUMNS = FEATURE_COLUMNS_EXPERIMENTAL
    else:
        FEATURE_COLUMNS = FEATURE_COLUMNS_IMPROVED
else:
    FEATURE_COLUMNS = FEATURE_COLUMNS_ORIGINAL

TARGET_COLUMN = 'cardio'
