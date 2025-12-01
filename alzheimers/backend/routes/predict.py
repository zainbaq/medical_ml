"""
Prediction routes for Alzheimer's disease prediction API
"""
from fastapi import APIRouter, HTTPException
from backend.models.schemas import CognitiveAssessment, PredictionResponse, ModelInfoResponse
from backend.models.ml_model import model_loader
from backend.utils.preprocessing import (
    prepare_features, get_stage_from_cdr,
    calculate_risk_score, get_risk_level
)
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictionResponse)
async def predict_alzheimers(assessment: CognitiveAssessment):
    """
    Predict Alzheimer's disease risk based on cognitive assessment data

    Args:
        assessment: Patient cognitive assessment data

    Returns:
        PredictionResponse with prediction, probability, risk score, stage, and risk level
    """
    try:
        # Check if model is loaded
        if not model_loader.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )

        # Prepare features
        features = prepare_features(assessment)

        # Make prediction
        prediction, probability = model_loader.predict(features)

        # Calculate derived values
        risk_score = calculate_risk_score(probability)
        stage = get_stage_from_cdr(assessment.cdr_score)
        risk_level = get_risk_level(probability)

        logger.info(f"Prediction made: pred={prediction}, prob={probability:.4f}, stage={stage}")

        # Return response
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_score=risk_score,
            stage=stage,
            risk_level=risk_level
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the loaded model

    Returns:
        ModelInfoResponse with model name, timestamp, metrics, and feature names
    """
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    metadata = model_loader.get_model_info()
    return ModelInfoResponse(
        model_name=metadata.get('model_name', 'unknown'),
        timestamp=metadata.get('timestamp', 'unknown'),
        metrics=metadata.get('metrics', {}),
        feature_names=metadata.get('feature_names', [])
    )
