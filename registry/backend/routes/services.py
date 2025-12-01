"""Service registration and discovery routes."""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from medical_ml_sdk.core.schemas import ServiceMetadata
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.service_store import service_store
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/services", tags=["services"])


@router.post("/register", status_code=201)
async def register_service(metadata: ServiceMetadata):
    """
    Register a new service or update an existing one.

    Args:
        metadata: Service metadata including endpoints, schemas, and capabilities

    Returns:
        Registration confirmation
    """
    try:
        service_store.add_service(metadata)
        return {
            "status": "registered",
            "service_id": metadata.service_id,
            "message": f"Service '{metadata.service_name}' registered successfully"
        }
    except Exception as e:
        logger.error(f"Error registering service: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{service_id}")
async def unregister_service(service_id: str):
    """
    Unregister a service from the registry.

    Args:
        service_id: Unique identifier of the service

    Returns:
        Unregistration confirmation
    """
    if service_store.remove_service(service_id):
        return {
            "status": "unregistered",
            "service_id": service_id,
            "message": f"Service '{service_id}' unregistered successfully"
        }
    else:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")


@router.get("", response_model=List[ServiceMetadata])
async def list_services():
    """
    List all registered services.

    Returns:
        List of all registered services with their metadata
    """
    return service_store.get_all_services()


@router.get("/{service_id}", response_model=ServiceMetadata)
async def get_service(service_id: str):
    """
    Get details about a specific service.

    Args:
        service_id: Unique identifier of the service

    Returns:
        Service metadata including schemas and endpoints
    """
    service = service_store.get_service(service_id)
    if service:
        return service
    else:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")


@router.post("/{service_id}/heartbeat")
async def service_heartbeat(service_id: str):
    """
    Receive a heartbeat from a service to indicate it's alive.

    Args:
        service_id: Unique identifier of the service

    Returns:
        Heartbeat acknowledgment
    """
    if service_store.update_heartbeat(service_id):
        return {
            "status": "acknowledged",
            "service_id": service_id,
            "timestamp": service_store.get_service(service_id).last_heartbeat
        }
    else:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")


@router.get("/search/by-tags", response_model=List[ServiceMetadata])
async def search_services(tags: Optional[str] = Query(None, description="Comma-separated tags")):
    """
    Search for services by tags.

    Args:
        tags: Comma-separated list of tags (e.g., "cancer,classification")

    Returns:
        List of services matching at least one tag
    """
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        return service_store.search_by_tags(tag_list)
    else:
        return service_store.get_all_services()
