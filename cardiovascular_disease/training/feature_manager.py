"""
Feature Manager for Tiered Cardiovascular Disease Prediction

Manages feature availability and preparation across prediction tiers:
- Basic: Core features from user input (18 features)
- Extended: Enhanced with lab values when available
- Clinical: Full clinical features from medical examinations

This module handles:
- Feature tier detection based on available data
- Missing value imputation strategies
- Feature selection for different model tiers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionTier(Enum):
    """Prediction tier based on available features."""
    BASIC = "basic"
    EXTENDED = "extended"
    CLINICAL = "clinical"
    FULL = "full"


@dataclass
class FeatureTierInfo:
    """Information about a feature tier."""
    tier: PredictionTier
    available_features: List[str]
    missing_features: List[str]
    confidence_level: str  # 'high', 'medium', 'low'
    description: str


class FeatureManager:
    """
    Manages feature availability and preparation for tiered prediction.

    The feature manager determines which prediction tier to use based on
    available data, handles missing value imputation, and prepares features
    for model inference.
    """

    # Feature tier definitions
    FEATURE_TIERS = {
        'basic': {
            'required': [
                'age_years', 'gender', 'height', 'weight',
                'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
                'smoke', 'alco', 'active'
            ],
            'engineered': [
                'bmi', 'pulse_pressure', 'mean_arterial_pressure',
                'hypertension_stage', 'bmi_category', 'age_group',
                'health_risk_composite', 'lifestyle_risk_score'
            ],
            'min_required': 9,  # Minimum required features for this tier
            'confidence': 'medium'
        },
        'extended': {
            'optional': [
                'total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol',
                'triglycerides', 'fasting_glucose', 'hba1c',
                'bp_medication', 'diabetes', 'family_history_cvd'
            ],
            'engineered': [
                'total_hdl_ratio', 'metabolic_syndrome_score',
                'diabetes_risk_indicator', 'bp_index', 'bp_control_status',
                'modifiable_risk_count', 'modifiable_risk_score',
                'cardiovascular_age_group', 'gender_adjusted_age_risk'
            ],
            'min_optional': 3,  # Minimum optional features to qualify for this tier
            'confidence': 'high'
        },
        'clinical': {
            'optional': [
                'chest_pain_type', 'resting_ecg', 'max_heart_rate',
                'exercise_angina', 'st_depression', 'thalassemia'
            ],
            'engineered': [
                'hr_reserve_pct', 'exercise_capacity_score',
                'ecg_abnormality', 'ischemia_indicator'
            ],
            'min_optional': 3,
            'confidence': 'very_high'
        }
    }

    # Population-level imputation values (from training data statistics)
    IMPUTATION_VALUES = {
        'age_years': 54.0,
        'gender': 1,  # Default to female (more conservative)
        'height': 165.0,
        'weight': 75.0,
        'bmi': 27.5,
        'ap_hi': 128,
        'ap_lo': 82,
        'cholesterol': 1,
        'gluc': 1,
        'smoke': 0,
        'alco': 0,
        'active': 1,
        'total_cholesterol': 200.0,
        'hdl_cholesterol': 50.0,
        'ldl_cholesterol': 120.0,
        'triglycerides': 130.0,
        'fasting_glucose': 95.0,
        'hba1c': 5.5,
        'bp_medication': 0,
        'diabetes': 0
    }

    # Modifiable feature definitions for explainability
    MODIFIABLE_FEATURES = {
        'smoke': {
            'name': 'Smoking',
            'risk_threshold': 1,
            'recommendation': 'Smoking cessation can reduce CVD risk by up to 50% within 1-2 years'
        },
        'active': {
            'name': 'Physical Activity',
            'risk_threshold': 0,  # Inactive is risk
            'recommendation': '150 minutes of moderate exercise per week can significantly reduce CVD risk'
        },
        'bmi': {
            'name': 'Body Weight',
            'risk_threshold': 30,
            'recommendation': 'Achieving a healthy BMI (18.5-24.9) reduces cardiovascular strain'
        },
        'ap_hi': {
            'name': 'Blood Pressure',
            'risk_threshold': 130,
            'recommendation': 'Blood pressure control through diet, exercise, or medication is crucial'
        },
        'cholesterol': {
            'name': 'Cholesterol',
            'risk_threshold': 2,
            'recommendation': 'Managing cholesterol through diet or statins reduces plaque buildup'
        },
        'alco': {
            'name': 'Alcohol Consumption',
            'risk_threshold': 1,
            'recommendation': 'Limiting alcohol intake improves heart health'
        }
    }

    def __init__(self, imputation_values: Optional[Dict] = None):
        """
        Initialize the feature manager.

        Args:
            imputation_values: Custom imputation values (uses defaults if None)
        """
        self.imputation_values = imputation_values or self.IMPUTATION_VALUES.copy()

    def get_available_features(
        self,
        data: Dict[str, Any]
    ) -> Tuple[PredictionTier, List[str], List[str]]:
        """
        Determine prediction tier based on available features.

        Args:
            data: Dictionary of patient data

        Returns:
            Tuple of (tier, available_features, missing_features)
        """
        available = []
        missing = []

        # Check all known features
        all_features = set()
        for tier_info in self.FEATURE_TIERS.values():
            all_features.update(tier_info.get('required', []))
            all_features.update(tier_info.get('optional', []))

        for feature in all_features:
            if feature in data and data[feature] is not None and not pd.isna(data[feature]):
                available.append(feature)
            else:
                missing.append(feature)

        # Determine tier
        tier = self._determine_tier(available)

        return tier, available, missing

    def _determine_tier(self, available_features: List[str]) -> PredictionTier:
        """
        Determine the appropriate prediction tier.

        Args:
            available_features: List of available feature names

        Returns:
            PredictionTier enum value
        """
        available_set = set(available_features)

        # Check clinical tier first (highest)
        clinical_optional = set(self.FEATURE_TIERS['clinical']['optional'])
        clinical_available = available_set.intersection(clinical_optional)
        if len(clinical_available) >= self.FEATURE_TIERS['clinical']['min_optional']:
            return PredictionTier.CLINICAL

        # Check extended tier
        extended_optional = set(self.FEATURE_TIERS['extended']['optional'])
        extended_available = available_set.intersection(extended_optional)
        if len(extended_available) >= self.FEATURE_TIERS['extended']['min_optional']:
            return PredictionTier.EXTENDED

        # Default to basic tier
        return PredictionTier.BASIC

    def get_tier_info(self, data: Dict[str, Any]) -> FeatureTierInfo:
        """
        Get detailed information about the prediction tier.

        Args:
            data: Patient data dictionary

        Returns:
            FeatureTierInfo with tier details
        """
        tier, available, missing = self.get_available_features(data)

        # Determine confidence level
        if tier == PredictionTier.CLINICAL:
            confidence = 'high'
            description = 'Full clinical assessment available'
        elif tier == PredictionTier.EXTENDED:
            confidence = 'high'
            description = 'Enhanced prediction with lab values'
        else:
            # Check basic feature completeness
            basic_required = set(self.FEATURE_TIERS['basic']['required'])
            basic_available = len(set(available).intersection(basic_required))
            if basic_available >= self.FEATURE_TIERS['basic']['min_required']:
                confidence = 'medium'
                description = 'Standard prediction from user-provided data'
            else:
                confidence = 'low'
                description = 'Limited data - some features imputed'

        return FeatureTierInfo(
            tier=tier,
            available_features=available,
            missing_features=missing,
            confidence_level=confidence,
            description=description
        )

    def impute_missing(
        self,
        data: Dict[str, Any],
        strategy: str = 'population_mean',
        required_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Impute missing values in patient data.

        Args:
            data: Patient data dictionary
            strategy: Imputation strategy ('population_mean', 'conservative', 'none')
            required_features: List of features that must be present after imputation

        Returns:
            Data dictionary with imputed values
        """
        result = data.copy()

        if required_features is None:
            required_features = self.FEATURE_TIERS['basic']['required']

        for feature in required_features:
            if feature not in result or result[feature] is None or pd.isna(result[feature]):
                if strategy == 'none':
                    continue
                elif strategy == 'conservative':
                    # Use more conservative (higher risk) estimates
                    result[feature] = self._get_conservative_value(feature)
                else:  # population_mean
                    result[feature] = self.imputation_values.get(feature)

        return result

    def _get_conservative_value(self, feature: str) -> Any:
        """
        Get conservative (higher risk) imputation value.

        Args:
            feature: Feature name

        Returns:
            Conservative imputation value
        """
        conservative_values = {
            'smoke': 0,  # Don't assume smoking
            'alco': 0,
            'active': 1,  # Assume active
            'cholesterol': 2,  # Slightly elevated
            'gluc': 1,
            'bp_medication': 0,
            'diabetes': 0
        }
        return conservative_values.get(feature, self.imputation_values.get(feature))

    def prepare_features_for_tier(
        self,
        data: Dict[str, Any],
        tier: PredictionTier
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Prepare features for a specific prediction tier.

        Args:
            data: Raw patient data
            tier: Target prediction tier

        Returns:
            Tuple of (prepared_data, list_of_imputed_features)
        """
        # Get features needed for this tier
        tier_name = tier.value
        tier_config = self.FEATURE_TIERS.get(tier_name, self.FEATURE_TIERS['basic'])

        required = tier_config.get('required', self.FEATURE_TIERS['basic']['required'])
        optional = tier_config.get('optional', [])

        # Impute required features
        imputed_features = []
        result = data.copy()

        for feature in required:
            if feature not in result or result[feature] is None:
                if feature in self.imputation_values:
                    result[feature] = self.imputation_values[feature]
                    imputed_features.append(feature)

        # Don't impute optional features - leave as NaN
        for feature in optional:
            if feature not in result:
                result[feature] = None

        return result, imputed_features

    def get_modifiable_recommendations(
        self,
        data: Dict[str, Any],
        top_n: int = 3
    ) -> List[Dict[str, str]]:
        """
        Get recommendations for modifiable risk factors.

        Args:
            data: Patient data dictionary
            top_n: Maximum number of recommendations

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        for feature, info in self.MODIFIABLE_FEATURES.items():
            if feature not in data or data[feature] is None:
                continue

            value = data[feature]
            threshold = info['risk_threshold']

            # Check if this is a risk factor
            is_risk = False
            if feature == 'active':
                is_risk = value == 0  # Inactive is risk
            elif feature in ['smoke', 'alco']:
                is_risk = value >= threshold
            elif feature in ['bmi', 'ap_hi']:
                is_risk = value >= threshold
            elif feature == 'cholesterol':
                is_risk = value >= threshold

            if is_risk:
                recommendations.append({
                    'feature': feature,
                    'name': info['name'],
                    'current_value': value,
                    'recommendation': info['recommendation']
                })

        # Sort by importance (order in MODIFIABLE_FEATURES)
        return recommendations[:top_n]

    def calculate_feature_completeness(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate feature completeness percentages by category.

        Args:
            data: Patient data dictionary

        Returns:
            Dictionary of completeness percentages
        """
        completeness = {}

        for tier_name, tier_config in self.FEATURE_TIERS.items():
            required = tier_config.get('required', [])
            optional = tier_config.get('optional', [])
            all_features = required + optional

            if not all_features:
                continue

            available = sum(
                1 for f in all_features
                if f in data and data[f] is not None and not pd.isna(data[f])
            )
            completeness[tier_name] = available / len(all_features)

        return completeness


def main():
    """Test the feature manager."""
    manager = FeatureManager()

    # Test with basic data
    basic_data = {
        'age_years': 55,
        'gender': 2,
        'height': 170,
        'weight': 80,
        'ap_hi': 140,
        'ap_lo': 90,
        'cholesterol': 2,
        'gluc': 1,
        'smoke': 1,
        'alco': 0,
        'active': 0
    }

    print("=" * 60)
    print("FEATURE MANAGER TEST")
    print("=" * 60)

    tier_info = manager.get_tier_info(basic_data)
    print(f"\nBasic data tier: {tier_info.tier.value}")
    print(f"Confidence: {tier_info.confidence_level}")
    print(f"Description: {tier_info.description}")

    # Test with extended data
    extended_data = basic_data.copy()
    extended_data.update({
        'total_cholesterol': 240,
        'hdl_cholesterol': 45,
        'fasting_glucose': 110,
        'bp_medication': 1
    })

    tier_info = manager.get_tier_info(extended_data)
    print(f"\nExtended data tier: {tier_info.tier.value}")
    print(f"Confidence: {tier_info.confidence_level}")

    # Get recommendations
    recommendations = manager.get_modifiable_recommendations(basic_data)
    print(f"\nModifiable risk recommendations:")
    for rec in recommendations:
        print(f"  - {rec['name']}: {rec['recommendation']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
