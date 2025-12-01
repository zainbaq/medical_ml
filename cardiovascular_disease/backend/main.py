"""
FastAPI application for cardiovascular disease prediction
"""
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .config import settings
from .models.schemas import HealthResponse, PatientData, PredictionResponse
from .models.ml_model import model_loader
from .routes import predict

# Import SDK components for service registration
from medical_ml_sdk.plugin.registry_client import RegistryClient
from medical_ml_sdk.core.schemas import ServiceMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup: Load the model
    logger.info("Starting up application...")
    logger.info("Loading ML model...")

    success = model_loader.load_latest_model()
    if success:
        logger.info("Model loaded successfully!")
        model_info = model_loader.get_model_info()
        if model_info:
            logger.info(f"Model: {model_info.get('model_name', 'Unknown')}")
            logger.info(f"ROC-AUC: {model_info.get('metrics', {}).get('roc_auc', 'N/A')}")
    else:
        logger.warning("Failed to load model. API will be limited.")

    # Auto-register with service registry
    registry_client = None
    if settings.AUTO_REGISTER:
        try:
            logger.info(f"Registering with service registry at {settings.REGISTRY_URL}...")
            registry_client = RegistryClient(settings.REGISTRY_URL)

            metadata = ServiceMetadata(
                service_id=settings.SERVICE_ID,
                service_name=settings.SERVICE_NAME,
                version=settings.SERVICE_VERSION,
                description=settings.SERVICE_DESCRIPTION,
                base_url=f"http://localhost:{settings.PORT}",
                port=settings.PORT,
                endpoints={
                    "predict": f"{settings.API_PREFIX}/predict",
                    "health": "/health",
                    "model_info": f"{settings.API_PREFIX}/model-info"
                },
                input_schema=PatientData.model_json_schema(),
                output_schema=PredictionResponse.model_json_schema(),
                tags=["cardiovascular", "disease", "classification", "health"],
                capabilities=model_loader.get_model_info()
            )

            registered = await registry_client.register_service(metadata)
            if registered:
                logger.info(f"Successfully registered with service registry!")
            else:
                logger.warning("Failed to register with service registry (service will still work)")
        except Exception as e:
            logger.warning(f"Could not register with service registry: {str(e)}")

    yield

    # Shutdown: Unregister from service registry
    logger.info("Shutting down application...")
    if settings.AUTO_REGISTER and registry_client:
        try:
            await registry_client.unregister_service(settings.SERVICE_ID)
            logger.info("Unregistered from service registry")
        except Exception as e:
            logger.warning(f"Could not unregister from service registry: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description="""
    FastAPI backend for cardiovascular disease prediction using machine learning.

    ## Features
    * Predict cardiovascular disease risk based on patient data
    * Calculate BMI automatically
    * Provide risk levels (low, medium, high)
    * Return prediction probabilities

    ## Input Features
    * Age (years)
    * Gender
    * Height (cm) and Weight (kg)
    * Blood Pressure (systolic and diastolic)
    * Cholesterol level
    * Glucose level
    * Lifestyle factors (smoking, alcohol, physical activity)
    """,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    predict.router,
    prefix=settings.API_PREFIX,
    tags=["Predictions"]
)


@app.get(
    "/",
    response_model=dict,
    summary="Root endpoint",
    tags=["General"]
)
async def root():
    """
    Root endpoint with API information
    """
    return {
        "name": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    tags=["General"]
)
async def health_check():
    """
    Health check endpoint to verify API and model status
    """
    model_info = model_loader.get_model_info()

    return HealthResponse(
        status="healthy" if model_loader.is_loaded() else "degraded",
        model_loaded=model_loader.is_loaded(),
        model_name=model_info.get('model_name') if model_info else None,
        version=settings.SERVICE_VERSION
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
