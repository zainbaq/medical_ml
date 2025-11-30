"""
Prediction routes for breast cancer detection
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import logging

from ..models.schemas import TumorFeatures, PredictionResponse, ErrorResponse
from ..models.ml_model import model_loader
from ..utils.preprocessing import prepare_features, get_diagnosis, get_confidence_level

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict breast cancer",
    description="Make a prediction for breast cancer based on tumor features",
    responses={
        200: {
            "description": "Successful prediction",
            "model": PredictionResponse
        },
        400: {
            "description": "Invalid input data",
            "model": ErrorResponse
        },
        503: {
            "description": "Model not available",
            "model": ErrorResponse
        }
    }
)
async def predict_breast_cancer(tumor_data: TumorFeatures):
    """
    Predict breast cancer diagnosis from tumor features

    Args:
        tumor_data: Tumor features extracted from FNA imaging

    Returns:
        Prediction result with probability and diagnosis
    """
    try:
        # Check if model is loaded
        if not model_loader.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please train a model first."
            )

        # Prepare features
        features = prepare_features(tumor_data)

        # Make prediction
        prediction, probability = model_loader.predict(features)

        # Determine diagnosis and confidence
        diagnosis = get_diagnosis(prediction)
        confidence = get_confidence_level(probability)

        # Create response
        response = PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            diagnosis=diagnosis,
            confidence=confidence
        )

        logger.info(f"Prediction made: {response}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )


@router.get(
    "/model-info",
    summary="Get model information",
    description="Retrieve information about the loaded model"
)
async def get_model_info():
    """
    Get information about the currently loaded model

    Returns:
        Model metadata including metrics and parameters
    """
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    metadata = model_loader.get_model_info()
    return metadata if metadata else {"message": "Model loaded but no metadata available"}
