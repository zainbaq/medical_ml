"""Configuration for the service registry."""

from pydantic_settings import BaseSettings


class RegistrySettings(BaseSettings):
    """Settings for the Medical ML Service Registry."""

    # Application
    APP_NAME: str = "Medical ML Service Registry"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 9000

    # API
    API_PREFIX: str = "/api/v1"

    # CORS
    ALLOWED_ORIGINS: list = ["*"]

    # Service Health
    SERVICE_TIMEOUT_SECONDS: int = 60  # Consider service unhealthy after this many seconds

    class Config:
        case_sensitive = True


settings = RegistrySettings()
