"""Medical ML Service Registry - Main FastAPI Application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
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
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    description=settings.SERVICE_DESCRIPTION,
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
        "name": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "description": settings.SERVICE_DESCRIPTION,
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
        "service": settings.SERVICE_NAME,
        "registered_services": service_count,
        "healthy_services": len(healthy_services),
        "version": settings.SERVICE_VERSION
    }


@app.get(f"{settings.API_PREFIX}/health/all")
async def aggregate_health():
    """
    Get health status of all registered services by querying their health endpoints.

    Returns:
        Health status categorized by healthy and unhealthy services
    """
    all_services = service_store.get_all_services()
    health_status = {}

    async with httpx.AsyncClient(timeout=5.0) as client:
        for service in all_services:
            try:
                # Query the service's health endpoint
                health_url = f"{service.base_url}{service.endpoints.get('health', '/health')}"
                response = await client.get(health_url)

                if response.status_code == 200:
                    health_data = response.json()
                    # Check if the service reports itself as healthy
                    is_healthy = health_data.get('status', '').lower() in ['healthy', 'ok']
                    status = "healthy" if is_healthy else "unhealthy"
                else:
                    status = "unhealthy"
            except Exception as e:
                logger.warning(f"Failed to check health of {service.service_id}: {str(e)}")
                status = "unhealthy"

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
