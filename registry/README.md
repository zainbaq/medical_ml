# Medical ML Service Registry

Central registry for medical ML prediction services.

## Overview

The Service Registry provides:
- **Service Discovery**: Central catalog of all available medical ML services
- **Health Monitoring**: Track which services are alive and healthy
- **Metadata Management**: Store service schemas, endpoints, and capabilities
- **Search**: Find services by tags and categories

## Installation

```bash
cd registry/backend
pip install -r requirements.txt
```

## Running the Registry

```bash
cd registry
./start_registry.sh
```

Or directly:

```bash
cd registry/backend
uvicorn main:app --host 0.0.0.0 --port 9000 --reload
```

## API Endpoints

### Service Discovery

- `GET /api/v1/services` - List all registered services
- `GET /api/v1/services/{service_id}` - Get specific service details
- `GET /api/v1/services/search/by-tags?tags=cancer,classification` - Search by tags

### Service Registration

- `POST /api/v1/services/register` - Register a service (called by services)
- `DELETE /api/v1/services/{service_id}` - Unregister a service
- `POST /api/v1/services/{service_id}/heartbeat` - Service heartbeat

### Health Checks

- `GET /health` - Registry health
- `GET /api/v1/health/all` - Aggregate health of all services

### Documentation

- `GET /docs` - OpenAPI documentation (Swagger UI)
- `GET /redoc` - ReDoc documentation

## Configuration

Edit `backend/config.py` to customize:
- Port (default: 9000)
- CORS settings
- Service timeout threshold

## Architecture

```
registry/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Configuration
│   ├── routes/
│   │   └── services.py      # Service endpoints
│   └── storage/
│       └── service_store.py # In-memory registry
├── start_registry.sh        # Startup script
└── README.md
```

## Usage Example

### From Frontend (Flask)

```python
import requests

# Discover all services
services = requests.get("http://localhost:9000/api/v1/services").json()

for service in services:
    print(f"{service['service_name']}: {service['base_url']}")
    print(f"  Tags: {service['tags']}")
    print(f"  Endpoints: {service['endpoints']}")

# Search for cancer-related services
cancer_services = requests.get(
    "http://localhost:9000/api/v1/services/search/by-tags?tags=cancer"
).json()
```

### From Service (Auto-Registration)

Services using the `medical_ml_sdk` automatically register themselves on startup:

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

## Testing

Test the registry:

```bash
# Start the registry
./start_registry.sh

# In another terminal, check health
curl http://localhost:9000/health

# View docs
open http://localhost:9000/docs
```
