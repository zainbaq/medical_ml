"""
Prediction routes for cardiovascular disease detection
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import logging

from ..models.schemas import PatientData, PredictionResponse, ErrorResponse
from ..models.ml_model import model_loader
from ..utils.preprocessing import prepare_features, get_risk_level

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict cardiovascular disease",
    description="Make a prediction for cardiovascular disease based on patient data",
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
async def predict_cardiovascular_disease(patient_data: PatientData):
    """
    Predict cardiovascular disease risk for a patient

    Args:
        patient_data: Patient information including vitals and lifestyle factors

    Returns:
        Prediction result with probability and risk level
    """
    try:
        # Check if model is loaded
        if not model_loader.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please train a model first."
            )

        # Prepare features
        features, bmi = prepare_features(patient_data)

        # Make prediction
        prediction, probability = model_loader.predict(features)

        # Determine risk level
        risk_level = get_risk_level(probability)

        # Create response
        response = PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            bmi=bmi
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
