"""
Preprocessing utilities for Alzheimer's disease prediction
"""
import numpy as np
from backend.models.schemas import CognitiveAssessment

# IMPORTANT: This value was calculated from training data
# Median SES value from OASIS dataset
MEDIAN_SES = 2.0

# CDR to Stage mapping
CDR_STAGE_MAP = {
    0.0: 'none',
    0.5: 'questionable',
    1.0: 'mild',
    2.0: 'moderate',
    3.0: 'severe'
}


def prepare_features(assessment: CognitiveAssessment) -> np.ndarray:
    """
    Convert CognitiveAssessment to numpy array in exact training order.

    Feature order: Age, Gender, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF

    Args:
        assessment: CognitiveAssessment pydantic model

    Returns:
        numpy array of shape (1, 9) with features in correct order
    """
    # Handle missing SES with median
    ses_value = assessment.socioeconomic_status if assessment.socioeconomic_status is not None else MEDIAN_SES

    # Encode gender: M=1, F=0
    gender_encoded = 1 if assessment.gender == 'M' else 0

    # Create feature array in exact order (MUST match training!)
    features = np.array([[
        assessment.age,
        gender_encoded,
        assessment.education_years,
        ses_value,
        assessment.mmse_score,
        assessment.cdr_score,
        assessment.estimated_total_intracranial_volume,
        assessment.normalized_whole_brain_volume,
        assessment.atlas_scaling_factor
    ]])

    return features


def get_stage_from_cdr(cdr_score: float) -> str:
    """
    Convert CDR score to stage description

    Args:
        cdr_score: Clinical Dementia Rating (0, 0.5, 1, 2, or 3)

    Returns:
        Stage string: none, questionable, mild, moderate, or severe
    """
    return CDR_STAGE_MAP.get(cdr_score, 'unknown')


def calculate_risk_score(probability: float) -> float:
    """
    Convert probability to risk score (0-100)

    Args:
        probability: Probability of dementia (0.0-1.0)

    Returns:
        Risk score (0.0-100.0)
    """
    return round(probability * 100, 2)


def get_risk_level(probability: float) -> str:
    """
    Categorize probability into risk levels

    Args:
        probability: Probability of dementia (0.0-1.0)

    Returns:
        Risk level: low, medium, or high
    """
    if probability < 0.33:
        return "low"
    elif probability < 0.66:
        return "medium"
    else:
        return "high"
