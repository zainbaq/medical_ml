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
    else:
        logger.error("Failed to load model!")

    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
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
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loader.is_loaded() else "unhealthy",
        model_loaded=model_loader.is_loaded(),
        timestamp=datetime.now().isoformat()
    )
