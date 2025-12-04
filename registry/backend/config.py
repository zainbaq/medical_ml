"""Configuration for the service registry."""

from medical_ml_sdk.core.config import BaseServiceConfig


class RegistrySettings(BaseServiceConfig):
    """Settings for the Medical ML Service Registry."""

    # Service Identity (required by BaseServiceConfig)
    SERVICE_NAME: str = "Medical ML Service Registry"
    SERVICE_ID: str = "registry"
    SERVICE_VERSION: str = "1.0.0"
    SERVICE_DESCRIPTION: str = "Central registry for medical ML prediction services"

    # Service Discovery (Registry doesn't register with itself)
    AUTO_REGISTER: bool = False

    # Application
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 9000

    # Service Health
    SERVICE_TIMEOUT_SECONDS: int = 60  # Consider service unhealthy after this many seconds

    class Config:
        case_sensitive = True


settings = RegistrySettings()
