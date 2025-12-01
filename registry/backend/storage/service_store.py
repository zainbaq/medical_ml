"""In-memory storage for registered services."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from medical_ml_sdk.core.schemas import ServiceMetadata
import logging

logger = logging.getLogger(__name__)


class ServiceStore:
    """
    In-memory storage for registered medical ML services.

    Maintains:
    - Service metadata
    - Heartbeat timestamps
    - Service search capabilities
    """

    def __init__(self):
        """Initialize the service store."""
        self._services: Dict[str, ServiceMetadata] = {}
        self._heartbeats: Dict[str, datetime] = {}

    def add_service(self, metadata: ServiceMetadata) -> None:
        """
        Add or update a service in the registry.

        Args:
            metadata: Service metadata to register
        """
        self._services[metadata.service_id] = metadata
        self._heartbeats[metadata.service_id] = datetime.now()
        logger.info(f"Service '{metadata.service_id}' added to registry")

    def get_service(self, service_id: str) -> Optional[ServiceMetadata]:
        """
        Get a service by ID.

        Args:
            service_id: Unique identifier of the service

        Returns:
            ServiceMetadata if found, None otherwise
        """
        return self._services.get(service_id)

    def get_all_services(self) -> List[ServiceMetadata]:
        """
        Get all registered services.

        Returns:
            List of all registered services
        """
        return list(self._services.values())

    def remove_service(self, service_id: str) -> bool:
        """
        Remove a service from the registry.

        Args:
            service_id: Unique identifier of the service

        Returns:
            True if service was removed, False if not found
        """
        if service_id in self._services:
            self._services.pop(service_id)
            self._heartbeats.pop(service_id, None)
            logger.info(f"Service '{service_id}' removed from registry")
            return True
        return False

    def update_heartbeat(self, service_id: str) -> bool:
        """
        Update the heartbeat timestamp for a service.

        Args:
            service_id: Unique identifier of the service

        Returns:
            True if heartbeat was updated, False if service not found
        """
        if service_id in self._services:
            self._heartbeats[service_id] = datetime.now()
            # Update last_heartbeat in metadata
            self._services[service_id].last_heartbeat = datetime.now().isoformat()
            return True
        return False

    def get_healthy_services(self, timeout_seconds: int = 60) -> List[ServiceMetadata]:
        """
        Get services that sent a heartbeat within the timeout period.

        Args:
            timeout_seconds: Heartbeat timeout in seconds

        Returns:
            List of healthy services
        """
        cutoff = datetime.now() - timedelta(seconds=timeout_seconds)
        return [
            self._services[sid]
            for sid, hb in self._heartbeats.items()
            if hb > cutoff
        ]

    def search_by_tags(self, tags: List[str]) -> List[ServiceMetadata]:
        """
        Find services matching any of the provided tags.

        Args:
            tags: List of tags to search for

        Returns:
            List of services matching at least one tag
        """
        if not tags:
            return self.get_all_services()

        return [
            svc for svc in self._services.values()
            if any(tag in svc.tags for tag in tags)
        ]

    def get_service_count(self) -> int:
        """
        Get the number of registered services.

        Returns:
            Count of registered services
        """
        return len(self._services)

    def service_exists(self, service_id: str) -> bool:
        """
        Check if a service exists in the registry.

        Args:
            service_id: Unique identifier of the service

        Returns:
            True if service exists
        """
        return service_id in self._services


# Global service store instance
service_store = ServiceStore()
