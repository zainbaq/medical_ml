"""
FastAPI application for breast cancer prediction
"""
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .config import settings
from .models.schemas import HealthResponse
from .models.ml_model import model_loader
from .routes import predict

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

    yield

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    FastAPI backend for breast cancer prediction using machine learning.

    ## Features
    * Predict breast cancer diagnosis (benign/malignant) based on tumor features
    * Provide prediction probabilities and confidence levels
    * Support for 30 features extracted from FNA imaging

    ## Input Features
    The model uses 30 features derived from digitized images of fine needle aspirate (FNA):
    * **Mean features**: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
    * **Standard error features**: SE of the above 10 measurements
    * **Worst features**: Worst/largest values of the above 10 measurements

    ## Dataset
    Based on the Wisconsin Breast Cancer dataset with 569 samples.
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
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
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
        version=settings.APP_VERSION
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
