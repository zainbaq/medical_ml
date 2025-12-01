"""
Configuration for Alzheimer's disease prediction API
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "Alzheimer's Disease Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Directories
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"

    # API settings
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: list = ["*"]
    HOST: str = "0.0.0.0"
    PORT: int = 8001  # Different from CVD (8000) and breast cancer (8002)

    class Config:
        env_file = ".env"


settings = Settings()
