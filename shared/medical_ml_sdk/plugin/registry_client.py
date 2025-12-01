"""Registry client for service registration and discovery."""

import httpx
import logging
from typing import Optional
from medical_ml_sdk.core.schemas import ServiceMetadata

logger = logging.getLogger(__name__)


class RegistryClient:
    """
    Client for medical ML services to register themselves with the service registry.

    This client handles:
    - Service registration on startup
    - Service unregistration on shutdown
    - Periodic heartbeat updates
    - Error handling for registry communication
    """

    def __init__(self, registry_url: str, timeout: float = 5.0):
        """
        Initialize the registry client.

        Args:
            registry_url: Base URL of the registry service (e.g., http://localhost:9000)
            timeout: Request timeout in seconds
        """
        self.registry_url = registry_url.rstrip('/')
        self.timeout = timeout
        self.api_base = f"{self.registry_url}/api/v1"

    async def register_service(self, metadata: ServiceMetadata) -> bool:
        """
        Register a service with the registry.

        Args:
            metadata: Service metadata to register

        Returns:
            bool: True if registration was successful
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_base}/services/register",
                    json=metadata.model_dump()
                )
                response.raise_for_status()
                logger.info(f"Service '{metadata.service_id}' registered successfully")
                return True

        except httpx.HTTPError as e:
            logger.warning(f"Failed to register service with registry: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during service registration: {str(e)}")
            return False

    async def unregister_service(self, service_id: str) -> bool:
        """
        Unregister a service from the registry.

        Args:
            service_id: Unique identifier of the service

        Returns:
            bool: True if unregistration was successful
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(
                    f"{self.api_base}/services/{service_id}"
                )
                response.raise_for_status()
                logger.info(f"Service '{service_id}' unregistered successfully")
                return True

        except httpx.HTTPError as e:
            logger.warning(f"Failed to unregister service from registry: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during service unregistration: {str(e)}")
            return False

    async def heartbeat(self, service_id: str) -> bool:
        """
        Send a heartbeat to the registry to indicate the service is alive.

        Args:
            service_id: Unique identifier of the service

        Returns:
            bool: True if heartbeat was acknowledged
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_base}/services/{service_id}/heartbeat"
                )
                response.raise_for_status()
                return True

        except httpx.HTTPError as e:
            logger.debug(f"Heartbeat failed for service '{service_id}': {str(e)}")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error during heartbeat: {str(e)}")
            return False

    async def get_all_services(self) -> Optional[list]:
        """
        Get all registered services from the registry.

        Returns:
            list: List of registered services, or None if request failed
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.api_base}/services")
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.warning(f"Failed to get services from registry: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting services: {str(e)}")
            return None

    async def get_service(self, service_id: str) -> Optional[dict]:
        """
        Get details about a specific service.

        Args:
            service_id: Unique identifier of the service

        Returns:
            dict: Service metadata, or None if not found
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.api_base}/services/{service_id}"
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.warning(f"Failed to get service '{service_id}': {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting service: {str(e)}")
            return None
