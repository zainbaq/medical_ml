"""
API v2 prediction routes with enhanced features.

Provides:
- Tiered prediction based on available data
- SHAP-based explainability
- Confidence intervals
- Modifiable risk recommendations
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import numpy as np
import logging

from ..models.schemas_v2 import (
    PatientDataV2,
    PredictionResponseV2,
    RiskFactor,
    RiskCategory,
    PredictionTier,
    ConfidenceLevel,
    FeatureRequirementsResponse,
    ModelInfoResponseV2,
    HealthResponseV2
)
from ..models.ml_model import model_loader
from ..utils.preprocessing_v2 import PreprocessorV2, get_risk_level
from ..utils.explainability import ExplainabilityEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Service version
SERVICE_VERSION = "2.0.0"


def get_risk_category(probability: float) -> RiskCategory:
    """
    Map probability to validated risk category.

    Based on clinical guidelines for 10-year CVD risk:
    - Low: < 20%
    - Moderate: 20-40%
    - Moderately High: 40-60%
    - High: 60-80%
    - Very High: > 80%
    """
    if probability < 0.20:
        return RiskCategory.LOW
    elif probability < 0.40:
        return RiskCategory.MODERATE
    elif probability < 0.60:
        return RiskCategory.MODERATELY_HIGH
    elif probability < 0.80:
        return RiskCategory.HIGH
    else:
        return RiskCategory.VERY_HIGH


@router.post(
    "/predict",
    response_model=PredictionResponseV2,
    summary="Enhanced CVD prediction (v2)",
    description="""
    Make a cardiovascular disease prediction with enhanced features.

    This endpoint provides:
    - **Tiered prediction**: Uses basic or extended model based on available data
    - **Explainability**: SHAP-based risk factor analysis
    - **Confidence intervals**: 95% confidence range for probability
    - **Recommendations**: Actionable suggestions for modifiable risks

    Optional lab values (HDL, LDL, etc.) improve prediction accuracy when provided.
    """,
    responses={
        200: {"description": "Successful prediction with explanation"},
        400: {"description": "Invalid input data"},
        503: {"description": "Model not available"}
    }
)
async def predict_v2(patient_data: PatientDataV2) -> PredictionResponseV2:
    """
    Make an enhanced CVD prediction with explainability.

    Args:
        patient_data: Patient data including optional lab values

    Returns:
        PredictionResponseV2 with prediction, explanation, and recommendations
    """
    # Check model availability
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )

    try:
        # Get model metadata
        metadata = model_loader.get_model_info() or {}
        feature_names = metadata.get('feature_names', [])

        # Initialize preprocessor
        preprocessor = PreprocessorV2(feature_names=feature_names)

        # Prepare features based on tier
        features, tier, missing_features, feature_dict = preprocessor.prepare_features_tiered(
            patient_data
        )

        # Get confidence level
        confidence = preprocessor.get_confidence_level(tier, missing_features)

        # Scale features
        features_scaled = model_loader.scaler.transform(features)

        # Get prediction
        prediction = int(model_loader.model.predict(features_scaled)[0])
        probability = float(model_loader.model.predict_proba(features_scaled)[0][1])

        # Calculate BMI
        bmi = preprocessor.calculate_bmi(patient_data.height, patient_data.weight)

        # Get risk category
        risk_category = get_risk_category(probability)

        # Risk level (v1 compatible)
        risk_level = get_risk_level(probability)

        # Initialize explainability engine
        shap_explainer = getattr(model_loader, 'shap_explainer', None)
        explainer = ExplainabilityEngine(
            shap_explainer=shap_explainer,
            feature_names=feature_names
        )

        # Get explanation
        risk_factors, protective_factors, recommendations = explainer.explain_prediction(
            features_scaled,
            feature_dict
        )

        # Get confidence interval
        lower, upper = explainer.get_probability_interval(
            features_scaled,
            model_loader.model
        )

        # Convert to RiskFactor objects
        risk_factor_objs = [
            RiskFactor(
                name=f['name'],
                display_name=f['display_name'],
                value=f['value'],
                contribution=f['contribution'],
                is_modifiable=f['is_modifiable'],
                recommendation=f.get('recommendation')
            )
            for f in risk_factors
        ]

        protective_factor_objs = [
            RiskFactor(
                name=f['name'],
                display_name=f['display_name'],
                value=f['value'],
                contribution=f['contribution'],
                is_modifiable=f['is_modifiable'],
                recommendation=f.get('recommendation')
            )
            for f in protective_factors
        ]

        # Get model version
        model_version = metadata.get('timestamp', SERVICE_VERSION)

        return PredictionResponseV2(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            bmi=bmi,
            probability_range=(lower, upper),
            risk_category=risk_category,
            risk_score=round(probability * 100, 1),
            prediction_tier=tier,
            prediction_confidence=confidence,
            missing_features=missing_features,
            top_risk_factors=risk_factor_objs,
            top_protective_factors=protective_factor_objs,
            modifiable_recommendations=recommendations,
            model_version=model_version
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get(
    "/feature-requirements",
    response_model=FeatureRequirementsResponse,
    summary="Get feature requirements",
    description="Returns feature requirements for each prediction tier"
)
async def get_feature_requirements() -> FeatureRequirementsResponse:
    """
    Get feature requirements for each prediction tier.

    Returns:
        FeatureRequirementsResponse with tier requirements
    """
    return FeatureRequirementsResponse(
        basic={
            "description": "Standard prediction from user-provided data",
            "required_features": [
                "age_years", "gender", "height", "weight",
                "ap_hi", "ap_lo", "cholesterol", "gluc",
                "smoke", "alco", "active"
            ],
            "total_features": 18,
            "confidence_level": "medium"
        },
        extended={
            "description": "Enhanced prediction with lab values",
            "optional_features": [
                "hdl_cholesterol", "ldl_cholesterol", "total_cholesterol",
                "triglycerides", "fasting_glucose", "hba1c",
                "bp_medication", "diabetes", "family_history_cvd"
            ],
            "min_optional_required": 3,
            "total_features": 25,
            "confidence_level": "high"
        },
        clinical={
            "description": "Full clinical assessment (future)",
            "optional_features": [
                "chest_pain_type", "resting_ecg", "max_heart_rate",
                "exercise_angina", "st_depression", "thalassemia"
            ],
            "min_optional_required": 3,
            "total_features": 30,
            "confidence_level": "very_high",
            "status": "Coming soon"
        }
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponseV2,
    summary="Get model information (v2)",
    description="Returns enhanced model information including performance metrics"
)
async def get_model_info_v2() -> ModelInfoResponseV2:
    """
    Get enhanced model information.

    Returns:
        ModelInfoResponseV2 with metrics and feature importance
    """
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    metadata = model_loader.get_model_info() or {}

    # Get feature importance if available
    feature_importance = {}
    if hasattr(model_loader.model, 'feature_importances_'):
        feature_names = metadata.get('feature_names', [])
        importances = model_loader.model.feature_importances_
        if len(feature_names) == len(importances):
            feature_importance = dict(zip(feature_names, importances.tolist()))

    # Performance targets
    performance_targets = {
        'roc_auc': 0.82,
        'accuracy': 0.78,
        'sensitivity': 0.75
    }

    # Check if targets are met
    metrics = metadata.get('metrics', {})
    meets_targets = all([
        metrics.get('roc_auc', 0) >= performance_targets['roc_auc'],
        metrics.get('accuracy', 0) >= performance_targets['accuracy'],
        metrics.get('recall', 0) >= performance_targets['sensitivity']
    ])

    return ModelInfoResponseV2(
        model_type=metadata.get('model_name', 'Unknown'),
        version=metadata.get('timestamp', SERVICE_VERSION),
        training_date=metadata.get('timestamp', 'Unknown'),
        metrics=metrics,
        feature_importance=feature_importance,
        performance_targets=performance_targets,
        meets_targets=meets_targets,
        available_tiers=['basic', 'extended']
    )


@router.get(
    "/health",
    response_model=HealthResponseV2,
    summary="Health check (v2)",
    description="Returns service health status with v2 capabilities"
)
async def health_check_v2() -> HealthResponseV2:
    """
    Enhanced health check for v2 API.

    Returns:
        HealthResponseV2 with service status and capabilities
    """
    model_loaded = model_loader.is_loaded()
    shap_available = hasattr(model_loader, 'shap_explainer') and model_loader.shap_explainer is not None

    metadata = model_loader.get_model_info() or {}

    return HealthResponseV2(
        status="healthy" if model_loaded else "degraded",
        service="cardiovascular-disease-predictor",
        version=SERVICE_VERSION,
        model_loaded=model_loaded,
        model_type=metadata.get('model_name') if model_loaded else None,
        shap_available=shap_available,
        available_tiers=['basic', 'extended'] if model_loaded else []
    )
