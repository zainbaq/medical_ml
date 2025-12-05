"""
ML Model loader and inference
"""
from medical_ml_sdk.core.model_loader import BaseModelLoader
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class BreastCancerModelLoader(BaseModelLoader):
    """
    Breast Cancer Model Loader - extends BaseModelLoader from SDK.

    Uses the common model loading logic from the SDK.
    """
    pass


# Global model instance
model_loader = BreastCancerModelLoader(models_dir=settings.MODELS_DIR)
