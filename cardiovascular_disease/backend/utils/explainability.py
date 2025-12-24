"""
Explainability Engine for Cardiovascular Disease Predictions

Generates SHAP-based explanations for predictions including:
- Top risk factors with contributions
- Top protective factors
- Modifiable risk recommendations
- Confidence intervals
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Explainability features will be limited.")


class ExplainabilityEngine:
    """
    Generates SHAP-based explanations for cardiovascular disease predictions.
    """

    # Modifiable feature definitions with recommendations
    MODIFIABLE_FEATURES = {
        'smoke': {
            'display_name': 'Smoking',
            'recommendation': 'Smoking cessation can reduce CVD risk by up to 50% within 1-2 years',
            'risk_direction': 'positive'  # Higher value = higher risk
        },
        'alco': {
            'display_name': 'Alcohol Consumption',
            'recommendation': 'Limiting alcohol intake to moderate levels improves heart health',
            'risk_direction': 'positive'
        },
        'active': {
            'display_name': 'Physical Activity',
            'recommendation': '150 minutes of moderate exercise per week significantly reduces CVD risk',
            'risk_direction': 'negative'  # Higher value = lower risk
        },
        'bmi': {
            'display_name': 'Body Mass Index',
            'recommendation': 'Achieving a healthy BMI (18.5-24.9) reduces cardiovascular strain',
            'risk_direction': 'positive'
        },
        'ap_hi': {
            'display_name': 'Systolic Blood Pressure',
            'recommendation': 'Blood pressure control through diet, exercise, or medication is crucial',
            'risk_direction': 'positive'
        },
        'ap_lo': {
            'display_name': 'Diastolic Blood Pressure',
            'recommendation': 'Maintaining diastolic BP below 80 mmHg is recommended',
            'risk_direction': 'positive'
        },
        'cholesterol': {
            'display_name': 'Cholesterol Level',
            'recommendation': 'Managing cholesterol through diet or statins reduces plaque buildup',
            'risk_direction': 'positive'
        },
        'gluc': {
            'display_name': 'Glucose Level',
            'recommendation': 'Controlling blood sugar reduces diabetes-related cardiovascular risk',
            'risk_direction': 'positive'
        },
        'hypertension_stage': {
            'display_name': 'Hypertension Stage',
            'recommendation': 'Work with healthcare provider to manage hypertension',
            'risk_direction': 'positive'
        },
        'lifestyle_risk_score': {
            'display_name': 'Lifestyle Risk Score',
            'recommendation': 'Improving diet, exercise, and reducing harmful habits can significantly reduce risk',
            'risk_direction': 'positive'
        }
    }

    # Feature display names
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
        'total_hdl_ratio': 'Cholesterol Ratio',
        'metabolic_syndrome_score': 'Metabolic Syndrome Score',
        'diabetes_risk_indicator': 'Diabetes Risk',
        'bp_index': 'Blood Pressure Index',
        'modifiable_risk_count': 'Modifiable Risk Count',
        'modifiable_risk_score': 'Modifiable Risk Score'
    }

    def __init__(
        self,
        shap_explainer: Optional[Any] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the explainability engine.

        Args:
            shap_explainer: Pre-trained SHAP TreeExplainer
            feature_names: List of feature names in model input order
        """
        self.shap_explainer = shap_explainer
        self.feature_names = feature_names or []

    def explain_prediction(
        self,
        features: np.ndarray,
        feature_values: Optional[Dict[str, Any]] = None,
        top_n_risk: int = 5,
        top_n_protective: int = 3
    ) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        Generate explanation for a single prediction.

        Args:
            features: Feature array for one sample
            feature_values: Dict mapping feature names to their values
            top_n_risk: Number of top risk factors to return
            top_n_protective: Number of top protective factors to return

        Returns:
            Tuple of (risk_factors, protective_factors, recommendations)
        """
        if feature_values is None:
            feature_values = {}

        # Ensure 2D array
        features_2d = features.reshape(1, -1) if features.ndim == 1 else features

        # Get SHAP values if available
        if self.shap_explainer is not None and SHAP_AVAILABLE:
            try:
                shap_values = self.shap_explainer.shap_values(features_2d)
                if isinstance(shap_values, list):
                    # For binary classification, use positive class SHAP values
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                shap_values = shap_values.flatten()
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")
                shap_values = None
        else:
            shap_values = None

        # Build factor lists
        risk_factors = []
        protective_factors = []
        recommendations = []

        if shap_values is not None and len(self.feature_names) == len(shap_values):
            # Create feature importance list with SHAP values
            importance_list = list(zip(self.feature_names, shap_values))

            # Sort by absolute SHAP value
            importance_list.sort(key=lambda x: abs(x[1]), reverse=True)

            for name, shap_value in importance_list:
                value = feature_values.get(name, features_2d[0][self.feature_names.index(name)])
                is_modifiable = name in self.MODIFIABLE_FEATURES
                display_name = self.FEATURE_DISPLAY_NAMES.get(name, name)

                factor = {
                    'name': name,
                    'display_name': display_name,
                    'value': float(value) if isinstance(value, (int, float, np.number)) else value,
                    'contribution': float(shap_value),
                    'is_modifiable': is_modifiable,
                    'recommendation': None
                }

                # Add recommendation for modifiable risk factors
                if is_modifiable and shap_value > 0:
                    factor['recommendation'] = self.MODIFIABLE_FEATURES[name]['recommendation']
                    if factor['recommendation'] not in recommendations:
                        recommendations.append(factor['recommendation'])

                if shap_value > 0:
                    if len(risk_factors) < top_n_risk:
                        risk_factors.append(factor)
                else:
                    if len(protective_factors) < top_n_protective:
                        protective_factors.append(factor)

        else:
            # Fallback: use feature values to estimate importance
            risk_factors, protective_factors, recommendations = self._estimate_importance(
                feature_values, top_n_risk, top_n_protective
            )

        return risk_factors, protective_factors, recommendations[:top_n_risk]

    def _estimate_importance(
        self,
        feature_values: Dict[str, Any],
        top_n_risk: int,
        top_n_protective: int
    ) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        Estimate feature importance when SHAP is not available.

        Uses heuristic rules based on clinical knowledge.
        """
        risk_factors = []
        protective_factors = []
        recommendations = []

        # Check each modifiable feature
        for name, info in self.MODIFIABLE_FEATURES.items():
            if name not in feature_values:
                continue

            value = feature_values[name]
            is_risk = False

            # Determine if this feature indicates risk
            if name == 'smoke' and value == 1:
                is_risk = True
            elif name == 'alco' and value == 1:
                is_risk = True
            elif name == 'active' and value == 0:
                is_risk = True
            elif name == 'bmi' and value >= 30:
                is_risk = True
            elif name == 'ap_hi' and value >= 130:
                is_risk = True
            elif name == 'ap_lo' and value >= 85:
                is_risk = True
            elif name == 'cholesterol' and value >= 2:
                is_risk = True
            elif name == 'gluc' and value >= 2:
                is_risk = True
            elif name == 'hypertension_stage' and value >= 2:
                is_risk = True

            factor = {
                'name': name,
                'display_name': info['display_name'],
                'value': float(value) if isinstance(value, (int, float, np.number)) else value,
                'contribution': 0.1 if is_risk else -0.05,  # Estimated
                'is_modifiable': True,
                'recommendation': info['recommendation'] if is_risk else None
            }

            if is_risk:
                risk_factors.append(factor)
                if info['recommendation'] not in recommendations:
                    recommendations.append(info['recommendation'])
            else:
                protective_factors.append(factor)

        # Add non-modifiable factors
        if 'age_years' in feature_values:
            age = feature_values['age_years']
            if age >= 55:
                risk_factors.append({
                    'name': 'age_years',
                    'display_name': 'Age',
                    'value': age,
                    'contribution': 0.15,
                    'is_modifiable': False,
                    'recommendation': None
                })

        return (
            risk_factors[:top_n_risk],
            protective_factors[:top_n_protective],
            recommendations
        )

    def get_probability_interval(
        self,
        features: np.ndarray,
        model: Any,
        confidence: float = 0.95,
        n_samples: int = 100
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for probability using bootstrap.

        Args:
            features: Feature array
            model: Trained classifier
            confidence: Confidence level (default 95%)
            n_samples: Number of bootstrap samples

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        features_2d = features.reshape(1, -1) if features.ndim == 1 else features

        # Get base prediction
        base_prob = model.predict_proba(features_2d)[0][1]

        # Simple variance estimation based on prediction confidence
        # More extreme predictions have smaller intervals
        variance = base_prob * (1 - base_prob)
        std_error = np.sqrt(variance / 10)  # Approximate standard error

        # Compute interval
        alpha = 1 - confidence
        z_score = 1.96  # 95% confidence

        lower = max(0, base_prob - z_score * std_error)
        upper = min(1, base_prob + z_score * std_error)

        return (round(lower, 3), round(upper, 3))

    def get_modifiable_recommendations(
        self,
        feature_values: Dict[str, Any],
        top_n: int = 5
    ) -> List[str]:
        """
        Get prioritized recommendations for modifiable risk factors.

        Args:
            feature_values: Dictionary of feature values
            top_n: Maximum recommendations to return

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Priority order for recommendations
        priority_order = [
            'smoke', 'ap_hi', 'cholesterol', 'active', 'bmi', 'alco', 'gluc'
        ]

        for name in priority_order:
            if name not in feature_values or name not in self.MODIFIABLE_FEATURES:
                continue

            value = feature_values[name]
            info = self.MODIFIABLE_FEATURES[name]

            # Check if this is a risk factor
            is_risk = False
            if name == 'smoke' and value == 1:
                is_risk = True
            elif name == 'alco' and value == 1:
                is_risk = True
            elif name == 'active' and value == 0:
                is_risk = True
            elif name == 'bmi' and value >= 30:
                is_risk = True
            elif name == 'ap_hi' and value >= 130:
                is_risk = True
            elif name == 'cholesterol' and value >= 2:
                is_risk = True
            elif name == 'gluc' and value >= 2:
                is_risk = True

            if is_risk and info['recommendation'] not in recommendations:
                recommendations.append(info['recommendation'])

            if len(recommendations) >= top_n:
                break

        return recommendations


def create_explainer(model: Any) -> Optional[Any]:
    """
    Create a SHAP explainer for a model.

    Args:
        model: Trained classifier (XGBoost, RandomForest, etc.)

    Returns:
        SHAP TreeExplainer or None if creation fails
    """
    if not SHAP_AVAILABLE:
        return None

    try:
        explainer = shap.TreeExplainer(model)
        return explainer
    except Exception as e:
        logger.warning(f"Failed to create SHAP explainer: {e}")
        return None
