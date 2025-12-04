"""
Configuration for Alzheimer's disease prediction API
"""
from pathlib import Path
from medical_ml_sdk.core.config import BaseServiceConfig


class Settings(BaseServiceConfig):
    """Application settings - extends BaseServiceConfig from SDK"""

    # Service Identity (required by BaseServiceConfig)
    SERVICE_NAME: str = "Alzheimer's Disease Prediction API"
    SERVICE_ID: str = "alzheimers"
    SERVICE_VERSION: str = "1.0.0"
    SERVICE_DESCRIPTION: str = "Predicts Alzheimer's disease risk from cognitive assessments"

    # Service Discovery
    REGISTRY_URL: str = "http://localhost:9000"
    AUTO_REGISTER: bool = True

    # Application
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"

    # Server
    PORT: int = 8002  # Fixed: was 8001, should be 8002

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
