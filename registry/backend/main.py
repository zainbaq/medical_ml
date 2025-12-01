"""Medical ML Service Registry - Main FastAPI Application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from routes import services
from storage.service_store import service_store
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Central registry for medical ML prediction services",
    docs_url="/docs",
    redoc_url="/redoc"
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
app.include_router(services.router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Medical ML Service Registry",
        "services_endpoint": f"{settings.API_PREFIX}/services",
        "health_endpoint": "/health",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    service_count = service_store.get_service_count()
    healthy_services = service_store.get_healthy_services(settings.SERVICE_TIMEOUT_SECONDS)

    return {
        "status": "healthy",
        "registered_services": service_count,
        "healthy_services": len(healthy_services),
        "version": settings.APP_VERSION
    }


@app.get(f"{settings.API_PREFIX}/health/all")
async def aggregate_health():
    """
    Get health status of all registered services.

    Returns:
        Health status categorized by healthy and unhealthy services
    """
    all_services = service_store.get_all_services()
    healthy_services = service_store.get_healthy_services(settings.SERVICE_TIMEOUT_SECONDS)
    healthy_ids = {svc.service_id for svc in healthy_services}

    health_status = {}
    for service in all_services:
        status = "healthy" if service.service_id in healthy_ids else "unhealthy"
        health_status[service.service_id] = {
            "service_name": service.service_name,
            "status": status,
            "base_url": service.base_url,
            "last_heartbeat": service.last_heartbeat
        }

    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
