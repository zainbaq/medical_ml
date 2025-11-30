"""
Data preprocessing utilities
"""
import numpy as np
from ..models.schemas import PatientData


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """
    Calculate BMI from height and weight

    Args:
        height_cm: Height in centimeters
        weight_kg: Weight in kilograms

    Returns:
        BMI value
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)


def get_risk_level(probability: float) -> str:
    """
    Determine risk level based on probability

    Args:
        probability: Probability of disease (0-1)

    Returns:
        Risk level: low, medium, or high
    """
    if probability < 0.33:
        return "low"
    elif probability < 0.66:
        return "medium"
    else:
        return "high"


def prepare_features(patient_data: PatientData) -> np.ndarray:
    """
    Convert patient data to feature array for model prediction

    Args:
        patient_data: Validated patient data

    Returns:
        numpy array of features in correct order
    """
    # Calculate BMI
    bmi = calculate_bmi(patient_data.height, patient_data.weight)

    # Create feature array in the same order as training
    # ['age_years', 'gender', 'height', 'weight', 'bmi',
    #  'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    #  'smoke', 'alco', 'active']
    features = np.array([[
        patient_data.age_years,
        patient_data.gender,
        patient_data.height,
        patient_data.weight,
        bmi,
        patient_data.ap_hi,
        patient_data.ap_lo,
        patient_data.cholesterol,
        patient_data.gluc,
        patient_data.smoke,
        patient_data.alco,
        patient_data.active
    ]])

    return features, bmi
