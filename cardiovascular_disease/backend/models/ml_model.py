"""
ML Model loader and inference
"""
from medical_ml_sdk.core.model_loader import BaseModelLoader
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class CVDModelLoader(BaseModelLoader):
    """
    Cardiovascular Disease Model Loader - extends BaseModelLoader from SDK.

    Uses the common model loading logic from the SDK.
    Can be extended with CVD-specific logic if needed.
    """
    pass


# Global model instance
model_loader = CVDModelLoader(models_dir=settings.MODELS_DIR)
