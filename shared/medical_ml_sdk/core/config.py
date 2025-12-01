"""Base configuration for medical ML services."""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


class BaseServiceConfig(BaseSettings):
    """
    Base configuration for all medical ML services.
    Services should extend this class and add service-specific settings.
    """

    # Service Identity
    SERVICE_NAME: str
    SERVICE_ID: str
    SERVICE_VERSION: str = "1.0.0"
    SERVICE_DESCRIPTION: str = ""

    # Service Discovery & Registry
    REGISTRY_URL: str = "http://localhost:9000"
    AUTO_REGISTER: bool = True

    # Application
    APP_NAME: str = ""  # Defaults to SERVICE_NAME if not set
    DEBUG: bool = False

    # Paths (should be overridden by services)
    BASE_DIR: Path = Path.cwd()
    MODELS_DIR: Path = Path.cwd() / "models"

    # API
    API_PREFIX: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    class Config:
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set APP_NAME to SERVICE_NAME if not provided
        if not self.APP_NAME:
            self.APP_NAME = self.SERVICE_NAME
