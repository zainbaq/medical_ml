"""
Enhanced preprocessing for API v2 with tiered feature support.

This module handles:
- Feature preparation for different prediction tiers
- Missing value handling
- Extended feature engineering for lab values
- Confidence level calculation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

try:
    from ..models.schemas_v2 import PatientDataV2, PredictionTier, ConfidenceLevel
except ImportError:
    from models.schemas_v2 import PatientDataV2, PredictionTier, ConfidenceLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessorV2:
    """
    Enhanced preprocessing with tiered feature support for v2 API.
    """

    # Feature display names for explainability
    FEATURE_DISPLAY_NAMES = {
        'age_years': 'Age',
        'gender': 'Gender',
        'height': 'Height',
        'weight': 'Weight',
        'bmi': 'Body Mass Index',
        'ap_hi': 'Systolic Blood Pressure',
        'ap_lo': 'Diastolic Blood Pressure',
        'cholesterol': 'Cholesterol Level',
        'gluc': 'Glucose Level',
        'smoke': 'Smoking Status',
        'alco': 'Alcohol Consumption',
        'active': 'Physical Activity',
        'pulse_pressure': 'Pulse Pressure',
        'mean_arterial_pressure': 'Mean Arterial Pressure',
        'hypertension_stage': 'Hypertension Stage',
        'bmi_category': 'BMI Category',
        'age_group': 'Age Group',
        'health_risk_composite': 'Health Risk Score',
        'lifestyle_risk_score': 'Lifestyle Risk Score',
        'hdl_cholesterol': 'HDL Cholesterol',
        'ldl_cholesterol': 'LDL Cholesterol',
        'total_cholesterol': 'Total Cholesterol',
        'triglycerides': 'Triglycerides',
        'fasting_glucose': 'Fasting Glucose',
        'hba1c': 'HbA1c',
        'bp_medication': 'BP Medication',
        'diabetes': 'Diabetes',
        'total_hdl_ratio': 'Cholesterol Ratio',
        'metabolic_syndrome_score': 'Metabolic Syndrome Score',
        'diabetes_risk_indicator': 'Diabetes Risk',
        'bp_index': 'Blood Pressure Index',
        'bp_control_status': 'BP Control Status',
        'modifiable_risk_count': 'Modifiable Risk Count',
        'modifiable_risk_score': 'Modifiable Risk Score'
    }

    # Extended feature definitions
    EXTENDED_FEATURES = [
        'hdl_cholesterol', 'ldl_cholesterol', 'total_cholesterol',
        'triglycerides', 'fasting_glucose', 'hba1c',
        'bp_medication', 'diabetes', 'family_history_cvd'
    ]

    # Basic features (required for model input)
    # Note: height/weight removed - using BMI only for multi-dataset compatibility
    # Height/weight are used to calculate BMI but not passed to model
    BASIC_FEATURES = [
        'age_years', 'gender', 'bmi',  # BMI only (no height/weight)
        'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active'
    ]

    # Engineered features (computed from basic)
    ENGINEERED_FEATURES_BASIC = [
        'pulse_pressure', 'mean_arterial_pressure', 'hypertension_stage',
        'bmi_category', 'age_group', 'health_risk_composite', 'lifestyle_risk_score'
    ]

    # Engineered features (computed when extended data available)
    ENGINEERED_FEATURES_EXTENDED = [
        'total_hdl_ratio', 'metabolic_syndrome_score', 'diabetes_risk_indicator',
        'bp_index', 'bp_control_status', 'modifiable_risk_count', 'modifiable_risk_score'
    ]

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize the preprocessor.

        Args:
            feature_names: List of feature names expected by the model
        """
        self.feature_names = feature_names

    def calculate_bmi(self, height: float, weight: float) -> float:
        """Calculate BMI from height (cm) and weight (kg)."""
        height_m = height / 100
        return round(weight / (height_m ** 2), 2)

    def determine_prediction_tier(
        self,
        patient_data: PatientDataV2
    ) -> Tuple[PredictionTier, List[str]]:
        """
        Determine the prediction tier based on available features.

        Args:
            patient_data: Patient data from request

        Returns:
            Tuple of (PredictionTier, list of missing optional features)
        """
        missing_features = []
        extended_available = 0

        for feature in self.EXTENDED_FEATURES:
            value = getattr(patient_data, feature, None)
            if value is None:
                missing_features.append(feature)
            else:
                extended_available += 1

        # Determine tier
        if extended_available >= 3:
            tier = PredictionTier.EXTENDED
        else:
            tier = PredictionTier.BASIC

        return tier, missing_features

    def get_confidence_level(
        self,
        tier: PredictionTier,
        missing_features: List[str]
    ) -> ConfidenceLevel:
        """
        Determine prediction confidence based on data completeness.

        Args:
            tier: Prediction tier
            missing_features: List of missing optional features

        Returns:
            ConfidenceLevel
        """
        if tier == PredictionTier.EXTENDED:
            if len(missing_features) <= 3:
                return ConfidenceLevel.HIGH
            else:
                return ConfidenceLevel.MEDIUM
        else:
            # Basic tier
            if len(missing_features) >= 7:
                return ConfidenceLevel.LOW
            else:
                return ConfidenceLevel.MEDIUM

    def prepare_basic_features(
        self,
        patient_data: PatientDataV2
    ) -> Dict[str, Any]:
        """
        Prepare basic features for model input.

        Height and weight from user input are converted to BMI.
        Only BMI is passed to the model (not height/weight) for
        multi-dataset compatibility.

        Args:
            patient_data: Patient data from request

        Returns:
            Dictionary of feature values
        """
        # Calculate BMI from height/weight (user input)
        bmi = self.calculate_bmi(patient_data.height, patient_data.weight)

        # Note: height/weight NOT included - using BMI only for model
        # This enables multi-dataset training (Framingham has estimated height/weight)
        features = {
            'age_years': patient_data.age_years,
            'gender': patient_data.gender,
            'bmi': bmi,  # Calculated from height/weight
            'ap_hi': patient_data.ap_hi,
            'ap_lo': patient_data.ap_lo,
            'cholesterol': patient_data.cholesterol,
            'gluc': patient_data.gluc,
            'smoke': patient_data.smoke,
            'alco': patient_data.alco,
            'active': patient_data.active
        }

        return features

    def compute_engineered_features_basic(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute basic engineered features.

        Args:
            features: Basic feature dictionary

        Returns:
            Features with engineered features added
        """
        result = features.copy()

        # Pulse Pressure
        result['pulse_pressure'] = features['ap_hi'] - features['ap_lo']

        # Mean Arterial Pressure
        result['mean_arterial_pressure'] = (
            features['ap_hi'] + 2 * features['ap_lo']
        ) / 3

        # Hypertension Stage (ACC/AHA guidelines)
        result['hypertension_stage'] = 0
        if features['ap_hi'] >= 120 or features['ap_lo'] >= 80:
            result['hypertension_stage'] = 1
        if features['ap_hi'] >= 130 or features['ap_lo'] >= 80:
            result['hypertension_stage'] = 2
        if features['ap_hi'] >= 140 or features['ap_lo'] >= 90:
            result['hypertension_stage'] = 3

        # BMI Category
        bmi = features['bmi']
        result['bmi_category'] = 0  # Normal
        if bmi < 18.5:
            result['bmi_category'] = 1  # Underweight
        elif bmi >= 25:
            result['bmi_category'] = 2  # Overweight
        if bmi >= 30:
            result['bmi_category'] = 3  # Obese

        # Age Group
        age = features['age_years']
        if age < 40:
            result['age_group'] = 0
        elif age < 50:
            result['age_group'] = 1
        elif age < 60:
            result['age_group'] = 2
        else:
            result['age_group'] = 3

        # Health Risk Composite
        result['health_risk_composite'] = (
            (features['cholesterol'] - 1) +
            (features['gluc'] - 1) +
            features['smoke'] +
            features['alco'] +
            (1 - features['active'])
        )

        # Lifestyle Risk Score
        result['lifestyle_risk_score'] = (
            features['smoke'] +
            features['alco'] +
            (1 - features['active'])
        )

        return result

    def compute_engineered_features_extended(
        self,
        features: Dict[str, Any],
        patient_data: PatientDataV2
    ) -> Dict[str, Any]:
        """
        Compute extended engineered features when lab values available.

        Args:
            features: Feature dictionary with basic features
            patient_data: Original patient data with optional fields

        Returns:
            Features with extended engineered features added
        """
        result = features.copy()

        # Total/HDL Ratio
        if (patient_data.total_cholesterol is not None and
                patient_data.hdl_cholesterol is not None and
                patient_data.hdl_cholesterol > 0):
            result['total_hdl_ratio'] = (
                patient_data.total_cholesterol / patient_data.hdl_cholesterol
            )
        else:
            result['total_hdl_ratio'] = 0

        # Metabolic Syndrome Score (0-5)
        score = 0
        if features['bmi'] >= 30:
            score += 1
        if patient_data.triglycerides and patient_data.triglycerides >= 150:
            score += 1
        if patient_data.hdl_cholesterol:
            if features['gender'] == 2 and patient_data.hdl_cholesterol < 40:
                score += 1
            elif features['gender'] == 1 and patient_data.hdl_cholesterol < 50:
                score += 1
        if features['ap_hi'] >= 130 or features['ap_lo'] >= 85:
            score += 1
        if patient_data.fasting_glucose and patient_data.fasting_glucose >= 100:
            score += 1
        result['metabolic_syndrome_score'] = score

        # Diabetes Risk Indicator
        result['diabetes_risk_indicator'] = 0
        if patient_data.hba1c:
            if patient_data.hba1c >= 6.5:
                result['diabetes_risk_indicator'] = 2
            elif patient_data.hba1c >= 5.7:
                result['diabetes_risk_indicator'] = 1
        elif patient_data.fasting_glucose:
            if patient_data.fasting_glucose >= 126:
                result['diabetes_risk_indicator'] = 2
            elif patient_data.fasting_glucose >= 100:
                result['diabetes_risk_indicator'] = 1

        # BP Index
        result['bp_index'] = (features['ap_hi'] / 120) * (features['ap_lo'] / 80)

        # BP Control Status
        result['bp_control_status'] = 0
        if patient_data.bp_medication == 1:
            if features['ap_hi'] < 140 and features['ap_lo'] < 90:
                result['bp_control_status'] = 1  # Controlled
            else:
                result['bp_control_status'] = 2  # Uncontrolled

        # Modifiable Risk Count
        mod_count = 0
        if features['ap_hi'] >= 130:
            mod_count += 1
        if features['cholesterol'] >= 2:
            mod_count += 1
        if features['smoke'] == 1:
            mod_count += 1
        if features['active'] == 0:
            mod_count += 1
        if features['bmi'] >= 30:
            mod_count += 1
        if features['alco'] == 1:
            mod_count += 1
        result['modifiable_risk_count'] = mod_count

        # Modifiable Risk Score (weighted)
        mod_score = 0
        if features['ap_hi'] >= 130:
            mod_score += 0.25
        if features['cholesterol'] >= 2:
            mod_score += 0.20
        if features['smoke'] == 1:
            mod_score += 0.25
        if features['active'] == 0:
            mod_score += 0.15
        if features['bmi'] >= 30:
            mod_score += 0.10
        if features['alco'] == 1:
            mod_score += 0.05
        result['modifiable_risk_score'] = mod_score

        return result

    def prepare_features_tiered(
        self,
        patient_data: PatientDataV2
    ) -> Tuple[np.ndarray, PredictionTier, List[str], Dict[str, Any]]:
        """
        Prepare features based on available data, determining appropriate tier.

        Args:
            patient_data: Patient data from request

        Returns:
            Tuple of (features array, prediction tier, missing features, feature dict)
        """
        # Determine tier and missing features
        tier, missing_features = self.determine_prediction_tier(patient_data)

        # Prepare basic features
        features = self.prepare_basic_features(patient_data)

        # Add engineered features (basic)
        features = self.compute_engineered_features_basic(features)

        # Add extended engineered features if tier is extended
        if tier == PredictionTier.EXTENDED:
            features = self.compute_engineered_features_extended(
                features, patient_data
            )

        # Convert to array in correct order
        if self.feature_names:
            feature_array = np.array([
                features.get(name, 0) for name in self.feature_names
            ]).reshape(1, -1)
        else:
            # Default order for basic features
            basic_order = (
                self.BASIC_FEATURES +
                self.ENGINEERED_FEATURES_BASIC
            )
            feature_array = np.array([
                features.get(name, 0) for name in basic_order
            ]).reshape(1, -1)

        return feature_array, tier, missing_features, features

    def get_feature_display_name(self, feature_name: str) -> str:
        """Get human-readable display name for a feature."""
        return self.FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)


def calculate_bmi(height: float, weight: float) -> float:
    """Convenience function for BMI calculation."""
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)


def get_risk_level(probability: float) -> str:
    """
    Get risk level string from probability (v1 compatible).

    Args:
        probability: Prediction probability

    Returns:
        Risk level string
    """
    if probability < 0.33:
        return "low"
    elif probability < 0.66:
        return "medium"
    else:
        return "high"
