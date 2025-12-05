"""
ML Model loader for Alzheimer's disease prediction
"""
from medical_ml_sdk.core.model_loader import BaseModelLoader
from backend.config import settings
import logging

logger = logging.getLogger(__name__)


class AlzheimerModelLoader(BaseModelLoader):
    """
    Alzheimer's Disease Model Loader - extends BaseModelLoader from SDK.

    Uses the common model loading logic from the SDK.
    """
    pass


# Global singleton instance
model_loader = AlzheimerModelLoader(models_dir=settings.MODELS_DIR)
