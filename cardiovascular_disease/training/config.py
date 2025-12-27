"""
Training configuration for cardiovascular disease prediction model

v2 Updates:
- XGBoost + Optuna configuration
- Multi-dataset training support
- Performance targets
- Extended feature tiers
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "cardio_train.csv"
RAW_DATA_DIR = DATA_DIR / "raw"

# ============================================================================
# DATASET CONFIGURATION (v2)
# ============================================================================

DATASETS = {
    'kaggle': {
        'enabled': True,
        'weight': 1.0,  # Sample weight multiplier
        'path': DATA_DIR / 'cardio_train.csv',
        'description': 'Original Kaggle cardiovascular dataset (70K records)'
    },
    'uci': {
        'enabled': False,  # EXCLUDED: Lacks height/weight/lifestyle data required for training
        'weight': 1.5,
        'path': DATA_DIR / 'uci_heart.csv',
        'description': 'UCI Heart Disease dataset - EXCLUDED (missing anthropometric features)'
    },
    'nhanes': {
        'enabled': True,
        'weight': 1.2,  # Higher weight for lab values
        'path': DATA_DIR / 'nhanes_cvd.csv',
        'description': 'NHANES 2017-2020 cardiovascular data'
    },
    'framingham': {
        'enabled': True,
        'weight': 1.3,  # Higher weight for validated outcomes
        'path': DATA_DIR / 'framingham.csv',
        'description': 'Framingham Heart Study teaching dataset (~4K records)'
    }
}

# Unified training data (after harmonization)
UNIFIED_TRAINING_FILE = DATA_DIR / 'unified_training.csv'

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
REMOVE_HEIGHT = True  # Remove height (use BMI only - enables Framingham dataset inclusion)

# Original feature names
FEATURE_COLUMNS_ORIGINAL = [
    'age_years', 'gender', 'height', 'weight', 'bmi',
    'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active'
]

# Improved feature names (high priority features)
# Note: height/weight removed - using BMI only (enables multi-dataset training)
FEATURE_COLUMNS_IMPROVED = [
    'age_years', 'gender', 'bmi',  # BMI only (no height/weight)
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

# ============================================================================
# XGBOOST + OPTUNA CONFIGURATION (v2)
# ============================================================================

XGBOOST_CONFIG = {
    'n_trials': 100,              # Number of Optuna trials
    'timeout': 3600,              # Max optimization time (1 hour)
    'early_stopping_rounds': 50,  # Early stopping patience
    'cv_folds': 5,                # Cross-validation folds
    'enable_shap': True,          # Compute SHAP values
}

# XGBoost hyperparameter search space
XGBOOST_PARAM_SPACE = {
    'max_depth': {'type': 'int', 'low': 3, 'high': 12},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
    'n_estimators': {'type': 'int', 'low': 100, 'high': 1000, 'step': 50},
    'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
    'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
    'reg_alpha': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
    'reg_lambda': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
    'gamma': {'type': 'float', 'low': 0, 'high': 0.5}
}

# ============================================================================
# PERFORMANCE TARGETS (v2)
# ============================================================================

PERFORMANCE_TARGETS = {
    'roc_auc': 0.82,      # Target: > 0.82 (currently 0.803)
    'accuracy': 0.78,     # Target: > 78% (currently 73.73%)
    'sensitivity': 0.75,  # Target: > 75%
    'specificity': 0.75,  # Target: > 75%
    'brier_score': 0.18   # Target: < 0.18
}

# Subgroup validation thresholds
SUBGROUP_DISPARITY_THRESHOLDS = {
    'roc_auc': 0.05,     # Max 5% AUC drop for any subgroup
    'accuracy': 0.05,    # Max 5% accuracy drop
    'recall': 0.10,      # Max 10% sensitivity drop
    'precision': 0.10    # Max 10% precision drop
}

# ============================================================================
# FEATURE TIER CONFIGURATION (v2)
# ============================================================================

PREDICTION_TIERS = {
    'basic': {
        'min_features': 11,
        'model_suffix': '_basic',
        'description': 'Standard prediction from user-provided data'
    },
    'extended': {
        'min_features': 17,
        'model_suffix': '_extended',
        'description': 'Enhanced prediction with lab values'
    },
    'clinical': {
        'min_features': 22,
        'model_suffix': '_clinical',
        'description': 'Full clinical assessment'
    }
}

# Extended features (v2) - these require additional datasets
EXTENDED_FEATURES = [
    # Metabolic (from NHANES/Framingham)
    'total_hdl_ratio',
    'metabolic_syndrome_score',
    'diabetes_risk_indicator',
    # Extended BP
    'bp_index',
    'hypertension_crisis',
    'bp_control_status',
    # Modifiable risk
    'modifiable_risk_count',
    'modifiable_risk_score',
    'improvement_potential',
    # Demographics
    'cardiovascular_age_group',
    'gender_adjusted_age_risk'
]

# Clinical features (from UCI)
CLINICAL_FEATURES = [
    'hr_reserve_pct',
    'exercise_capacity_score',
    'ecg_abnormality',
    'ischemia_indicator'
]

# ============================================================================
# MODEL TRAINING OPTIONS (v2)
# ============================================================================

# Use XGBoost instead of GradientBoosting
USE_XGBOOST = True

# Use Optuna hyperparameter optimization
USE_OPTUNA = True

# Train tiered models (basic + extended)
TRAIN_TIERED_MODELS = True

# Include extended features in training
INCLUDE_EXTENDED_FEATURES = True

# Validate subgroup fairness
VALIDATE_SUBGROUPS = True

# Calibrate probability outputs
CALIBRATE_PROBABILITIES = True
