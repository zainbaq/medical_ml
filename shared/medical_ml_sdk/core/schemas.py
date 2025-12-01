"""Common schemas for medical ML services."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime


class HealthResponse(BaseModel):
    """Standardized health check response for all services."""

    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp of health check"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "random_forest",
                "version": "1.0.0",
                "timestamp": "2025-11-30T12:00:00"
            }
        }


class ServiceMetadata(BaseModel):
    """
    Metadata about a medical ML service for the registry.
    Contains all information needed for service discovery and interaction.
    """

    service_id: str = Field(..., description="Unique identifier for the service")
    service_name: str = Field(..., description="Human-readable service name")
    version: str = Field(..., description="Service version")
    description: str = Field(default="", description="Service description")
    base_url: str = Field(..., description="Base URL of the service (e.g., http://localhost:8000)")
    port: int = Field(..., description="Port the service is running on")
    endpoints: Dict[str, str] = Field(
        ...,
        description="Available endpoints (e.g., {'predict': '/api/v1/predict', 'health': '/health'})"
    )
    input_schema: Dict = Field(..., description="JSON schema for prediction input")
    output_schema: Dict = Field(..., description="JSON schema for prediction output")
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization (e.g., ['cancer', 'classification'])"
    )
    capabilities: Optional[Dict] = Field(
        None,
        description="Model capabilities and metrics"
    )
    registered_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="When the service was registered"
    )
    last_heartbeat: Optional[str] = Field(
        None,
        description="Last heartbeat timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "service_id": "breast_cancer",
                "service_name": "Breast Cancer Prediction API",
                "version": "1.0.0",
                "description": "Predicts breast cancer from tumor features",
                "base_url": "http://localhost:8002",
                "port": 8002,
                "endpoints": {
                    "predict": "/api/v1/predict",
                    "health": "/health",
                    "model_info": "/api/v1/model-info"
                },
                "input_schema": {},
                "output_schema": {},
                "tags": ["cancer", "breast", "classification"],
                "capabilities": {
                    "model_name": "random_forest",
                    "metrics": {"accuracy": 0.97}
                },
                "registered_at": "2025-11-30T12:00:00"
            }
        }
