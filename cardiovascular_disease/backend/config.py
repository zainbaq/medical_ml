"""
Configuration for FastAPI backend
"""
from pathlib import Path
from medical_ml_sdk.core.config import BaseServiceConfig


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
