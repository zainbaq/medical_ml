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


def prepare_features(patient_data: PatientData) -> tuple[np.ndarray, float]:
    """
    Convert patient data to feature array with 18 engineered features.

    IMPORTANT: Model was trained with 18 features including engineered features.
    Feature order MUST match model metadata: ['age_years', 'gender', 'height', 'bmi',
    'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active',
    'pulse_pressure', 'mean_arterial_pressure', 'hypertension_stage',
    'bmi_category', 'age_group', 'health_risk_composite', 'lifestyle_risk_score']

    Args:
        patient_data: Validated patient data

    Returns:
        Tuple of (feature array with 18 features, calculated BMI)
    """
    # Calculate BMI
    bmi = calculate_bmi(patient_data.height, patient_data.weight)

    # Engineered Feature 1: Pulse Pressure (difference between systolic and diastolic)
    pulse_pressure = patient_data.ap_hi - patient_data.ap_lo

    # Engineered Feature 2: Mean Arterial Pressure
    mean_arterial_pressure = (patient_data.ap_hi + 2 * patient_data.ap_lo) / 3

    # Engineered Feature 3: Hypertension Stage (0-3 based on AHA guidelines)
    if patient_data.ap_hi < 120 and patient_data.ap_lo < 80:
        hypertension_stage = 0  # Normal
    elif patient_data.ap_hi < 130 and patient_data.ap_lo < 80:
        hypertension_stage = 1  # Elevated
    elif patient_data.ap_hi < 140 or patient_data.ap_lo < 90:
        hypertension_stage = 2  # Stage 1
    else:
        hypertension_stage = 3  # Stage 2

    # Engineered Feature 4: BMI Category (0-3)
    if bmi < 18.5:
        bmi_category = 0  # Underweight
    elif bmi < 25:
        bmi_category = 1  # Normal
    elif bmi < 30:
        bmi_category = 2  # Overweight
    else:
        bmi_category = 3  # Obese

    # Engineered Feature 5: Age Group (0-3)
    if patient_data.age_years <= 35:
        age_group = 0  # Young (18-35)
    elif patient_data.age_years <= 55:
        age_group = 1  # Middle (36-55)
    elif patient_data.age_years <= 70:
        age_group = 2  # Senior (56-70)
    else:
        age_group = 3  # Elderly (71+)

    # Engineered Feature 6: Health Risk Composite
    # Sum of cholesterol and glucose levels (range: 2-6)
    health_risk_composite = patient_data.cholesterol + patient_data.gluc

    # Engineered Feature 7: Lifestyle Risk Score
    # Sum of risk factors: smoking + alcohol + physical inactivity (range: 0-3)
    lifestyle_risk_score = patient_data.smoke + patient_data.alco + (1 - patient_data.active)

    # Create feature array - EXACTLY 18 features, NO WEIGHT (removed per metadata)
    # Order MUST match model training
    features = np.array([[
        patient_data.age_years,          # 0
        patient_data.gender,             # 1
        patient_data.height,             # 2
        bmi,                             # 3
        patient_data.ap_hi,              # 4
        patient_data.ap_lo,              # 5
        patient_data.cholesterol,        # 6
        patient_data.gluc,               # 7
        patient_data.smoke,              # 8
        patient_data.alco,               # 9
        patient_data.active,             # 10
        pulse_pressure,                  # 11
        mean_arterial_pressure,          # 12
        hypertension_stage,              # 13
        bmi_category,                    # 14
        age_group,                       # 15
        health_risk_composite,           # 16
        lifestyle_risk_score             # 17
    ]])

    return features, bmi
