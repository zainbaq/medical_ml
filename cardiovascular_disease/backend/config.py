"""
Configuration for FastAPI backend
"""
from pathlib import Path
from medical_ml_sdk.core.config import BaseServiceConfig


# Performance targets for model evaluation
PERFORMANCE_TARGETS = {
    'roc_auc': 0.82,
    'accuracy': 0.78,
    'sensitivity': 0.75,
    'specificity': 0.70
}

# XGBoost hyperparameter optimization config
XGBOOST_CONFIG = {
    'n_trials': 100,
    'timeout': 3600,
    'early_stopping_rounds': 50,
    'cv_folds': 5
}

# Dataset configuration
DATASETS = {
    'kaggle': {
        'name': 'Kaggle CVD Dataset',
        'path': 'data/cardio_train.csv',
        'records': 70000
    },
    'uci': {
        'name': 'UCI Heart Disease',
        'path': 'data/raw/uci_heart.csv',
        'records': 920
    },
    'nhanes': {
        'name': 'NHANES 2017-2018',
        'path': 'data/raw/nhanes_cvd.csv',
        'records': 5000
    },
    'framingham': {
        'name': 'Framingham Heart Study',
        'path': 'data/raw/framingham.csv',
        'records': 4434
    }
}

# Cross-validation folds
CV_FOLDS = 5


class Settings(BaseServiceConfig):
    """Application settings - extends BaseServiceConfig from SDK"""

    # Service Identity (required by BaseServiceConfig)
    SERVICE_NAME: str = "Cardiovascular Disease Prediction API"
    SERVICE_ID: str = "cardiovascular_disease"
    SERVICE_VERSION: str = "1.0.0"
    SERVICE_DESCRIPTION: str = "Predicts cardiovascular disease risk from patient data"

    # Service Discovery
    REGISTRY_URL: str = "http://localhost:9000"
    AUTO_REGISTER: bool = True

    # Application
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"

    # Server
    PORT: int = 8003

    # Model (service-specific)
    MODEL_PATH: str = ""
    SCALER_PATH: str = ""

    class Config:
        case_sensitive = True


settings = Settings()
