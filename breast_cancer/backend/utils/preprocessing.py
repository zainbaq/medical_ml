"""
Preprocessing utilities for breast cancer prediction
"""
import numpy as np
from ..models.schemas import TumorFeatures


def prepare_features(tumor_data: TumorFeatures) -> np.ndarray:
    """
    Prepare features from tumor data for model prediction

    Args:
        tumor_data: TumorFeatures object with all 30 features

    Returns:
        numpy array of shape (1, 30) with features in correct order
    """
    # Extract features in the correct order
    features = np.array([[
        # Mean features
        tumor_data.radius_mean,
        tumor_data.texture_mean,
        tumor_data.perimeter_mean,
        tumor_data.area_mean,
        tumor_data.smoothness_mean,
        tumor_data.compactness_mean,
        tumor_data.concavity_mean,
        tumor_data.concave_points_mean,
        tumor_data.symmetry_mean,
        tumor_data.fractal_dimension_mean,
        # Standard error features
        tumor_data.radius_se,
        tumor_data.texture_se,
        tumor_data.perimeter_se,
        tumor_data.area_se,
        tumor_data.smoothness_se,
        tumor_data.compactness_se,
        tumor_data.concavity_se,
        tumor_data.concave_points_se,
        tumor_data.symmetry_se,
        tumor_data.fractal_dimension_se,
        # Worst features
        tumor_data.radius_worst,
        tumor_data.texture_worst,
        tumor_data.perimeter_worst,
        tumor_data.area_worst,
        tumor_data.smoothness_worst,
        tumor_data.compactness_worst,
        tumor_data.concavity_worst,
        tumor_data.concave_points_worst,
        tumor_data.symmetry_worst,
        tumor_data.fractal_dimension_worst
    ]])

    return features


def get_diagnosis(prediction: int) -> str:
    """
    Convert prediction to diagnosis string

    Args:
        prediction: 0 (benign) or 1 (malignant)

    Returns:
        Diagnosis string
    """
    return "malignant" if prediction == 1 else "benign"


def get_confidence_level(probability: float) -> str:
    """
    Determine confidence level based on probability

    Args:
        probability: Probability of malignancy (0-1)

    Returns:
        Confidence level: low, medium, high
    """
    # For binary classification, confidence is based on how far from 0.5
    distance_from_threshold = abs(probability - 0.5)

    if distance_from_threshold >= 0.4:  # Very confident (p < 0.1 or p > 0.9)
        return "high"
    elif distance_from_threshold >= 0.2:  # Moderately confident
        return "medium"
    else:  # Less confident (close to 0.5)
        return "low"
