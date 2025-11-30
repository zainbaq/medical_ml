"""
Configuration for FastAPI backend
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "Breast Cancer Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"

    # Model
    MODEL_PATH: str = ""
    SCALER_PATH: str = ""

    # API
    API_PREFIX: str = "/api/v1"

    # CORS
    ALLOWED_ORIGINS: list = ["*"]

    class Config:
        case_sensitive = True


settings = Settings()
