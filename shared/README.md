# Medical ML SDK

Shared framework for medical ML prediction services.

## Overview

The Medical ML SDK provides common base classes and utilities for building standardized medical ML prediction services. It eliminates code duplication and provides a consistent pattern for:

- Model loading and inference
- Service configuration
- Service registration and discovery
- Health checks and monitoring

## Installation

Install in development mode:

```bash
cd shared
pip install -e .
```

## Components

### Core

- **BaseModelLoader**: Common model loading logic for scikit-learn models
- **BaseServiceConfig**: Base configuration class for services
- **HealthResponse**: Standardized health check response
- **ServiceMetadata**: Service information for registry

### Plugin

- **RegistryClient**: Client for registering services with the registry

## Usage

### Extending BaseModelLoader

```python
from medical_ml_sdk.core import BaseModelLoader

class MyModelLoader(BaseModelLoader):
    pass  # Use base implementation or extend

model_loader = MyModelLoader(models_dir=Path("./models"))
```

### Extending BaseServiceConfig

```python
from medical_ml_sdk.core import BaseServiceConfig

class MySettings(BaseServiceConfig):
    SERVICE_NAME = "My Medical ML API"
    SERVICE_ID = "my_service"
    PORT = 8000

settings = MySettings()
```

### Auto-Registration

```python
from medical_ml_sdk.plugin import RegistryClient
from medical_ml_sdk.core import ServiceMetadata

registry_client = RegistryClient("http://localhost:9000")

metadata = ServiceMetadata(
    service_id="my_service",
    service_name="My Medical ML API",
    base_url="http://localhost:8000",
    port=8000,
    endpoints={"predict": "/api/v1/predict"},
    input_schema={},
    output_schema={},
    tags=["medical", "classification"]
)

await registry_client.register_service(metadata)
```

## License

Proprietary - Medical ML Project
