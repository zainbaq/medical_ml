"""Base model loader for medical ML services."""

import joblib
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseModelLoader:
    """
    Generic model loader for medical ML services.
    Handles loading model/scaler/metadata from latest_model_info.json.

    This class extracts the common ModelLoader pattern used across all medical ML services,
    eliminating code duplication.
    """

    def __init__(self, models_dir: Path):
        """
        Initialize the model loader.

        Args:
            models_dir: Path to the directory containing models
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None

    def load_latest_model(self) -> bool:
        """
        Load the latest model, scaler, and metadata from latest_model_info.json.

        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Load latest model info
            latest_info_path = self.models_dir / "latest_model_info.json"

            if not latest_info_path.exists():
                logger.error(f"Latest model info not found at {latest_info_path}")
                return False

            with open(latest_info_path, 'r') as f:
                latest_info = json.load(f)

            # Load model
            model_path = Path(latest_info['model_path'])
            if not model_path.exists():
                logger.error(f"Model file not found at {model_path}")
                return False

            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")

            # Load scaler
            scaler_path = Path(latest_info['scaler_path'])
            if not scaler_path.exists():
                logger.error(f"Scaler file not found at {scaler_path}")
                return False

            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")

            # Load metadata
            metadata_path = Path(latest_info['metadata_path'])
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.feature_names = self.metadata.get('feature_names', [])
                logger.info(f"Metadata loaded from {metadata_path}")

            logger.info("Model loading successful!")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction on input features.

        Args:
            features: numpy array of shape (1, n_features)

        Returns:
            Tuple of (prediction, probability)

        Raises:
            ValueError: If model or scaler not loaded
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not loaded")

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]

        return int(prediction), float(probability)

    def is_loaded(self) -> bool:
        """
        Check if model and scaler are loaded.

        Returns:
            bool: True if both model and scaler are loaded
        """
        return self.model is not None and self.scaler is not None

    def get_model_info(self) -> Optional[dict]:
        """
        Get model metadata.

        Returns:
            dict: Model metadata including metrics, feature names, etc.
        """
        return self.metadata
