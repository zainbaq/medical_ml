"""
FastAPI main application for Alzheimer's disease prediction
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from backend.config import settings
from backend.routes import predict
from backend.models.ml_model import model_loader
from backend.models.schemas import HealthResponse
from datetime import datetime

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
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Alzheimer's Disease Prediction API...")
    logger.info("Loading ML model...")
    success = model_loader.load_latest_model()
    if success:
        logger.info("Model loaded successfully!")
        model_info = model_loader.get_model_info()
        if model_info:
            logger.info(f"Model: {model_info.get('model_name', 'Unknown')}")
            logger.info(f"Metrics: {model_info.get('metrics', {})}")
    else:
        logger.error("Failed to load model!")

    # Auto-register with service registry
    registry_client = None
    if settings.AUTO_REGISTER:
        try:
            logger.info(f"Registering with service registry at {settings.REGISTRY_URL}...")
            registry_client = RegistryClient(settings.REGISTRY_URL)

            from backend.models.schemas import CognitiveAssessment, PredictionResponse

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
                input_schema=CognitiveAssessment.model_json_schema(),
                output_schema=PredictionResponse.model_json_schema(),
                tags=["alzheimers", "dementia", "classification", "neurology"],
                capabilities=model_loader.get_model_info()
            )

            registered = await registry_client.register_service(metadata)
            if registered:
                logger.info("Successfully registered with service registry!")
            else:
                logger.warning("Failed to register (service will still work)")
        except Exception as e:
            logger.warning(f"Could not register with service registry: {str(e)}")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if settings.AUTO_REGISTER and registry_client:
        try:
            await registry_client.unregister_service(settings.SERVICE_ID)
            logger.info("Unregistered from service registry")
        except Exception as e:
            logger.warning(f"Could not unregister: {str(e)}")


app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description="API for predicting Alzheimer's disease risk based on cognitive assessments and brain imaging metrics",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix=settings.API_PREFIX, tags=["predictions"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_info = model_loader.get_model_info()
    return HealthResponse(
        status="healthy" if model_loader.is_loaded() else "unhealthy",
        service=settings.SERVICE_NAME,
        model_loaded=model_loader.is_loaded(),
        model_name=model_info.get('model_name') if model_info else None,
        version=settings.SERVICE_VERSION,
        timestamp=datetime.now().isoformat()
    )
