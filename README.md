# Medical ML Service Registry

A unified microservices platform for medical machine learning predictions, featuring automatic service discovery, health monitoring, and a standardized API interface.

## Overview

This repository provides a complete medical ML prediction platform with:
- **Unified Service Registry**: Centralized service discovery and health monitoring
- **Multiple ML Services**: Independent prediction services for different medical conditions
- **Shared SDK**: Common framework eliminating code duplication
- **Auto-Registration**: Services automatically register on startup
- **REST API**: Standardized prediction interface across all services
- **Interactive Documentation**: Auto-generated API docs for each service

## Implemented Services

✅ **Cardiovascular Disease Prediction** (Port 8000)
- Predicts cardiovascular disease risk from patient vitals and lifestyle factors
- Features: 18 engineered features including pulse pressure, hypertension staging, BMI categories
- Model: Gradient Boosting Classifier
- Accuracy: 73.7% | ROC-AUC: 0.80

✅ **Breast Cancer Detection** (Port 8001)
- Predicts benign vs malignant tumors from FNA imaging features
- Features: 30 features (mean, standard error, worst values of 10 measurements)
- Model: Random Forest Classifier
- Accuracy: 97.4% | ROC-AUC: 0.99

✅ **Alzheimer's Disease Prediction** (Port 8002)
- Predicts dementia risk from cognitive assessments and brain imaging
- Features: 9 features including MMSE, CDR, brain volume measurements
- Model: Support Vector Machine (RBF kernel)
- Accuracy: 100% | ROC-AUC: 1.0

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Service Registry (Port 9000)               │
│  • Service registration & discovery                     │
│  • Health monitoring (active health checks)             │
│  • Tag-based search                                     │
│  • Dynamic schema serving                               │
└────────────┬────────────────────────────────────────────┘
             │
             │ Auto-registration on startup
             │
    ┌────────┴────────┬────────────┬──────────────┐
    │                 │            │              │
┌───▼────┐      ┌────▼───┐   ┌────▼────┐   ┌────▼──────┐
│  CVD   │      │ Breast │   │Alzheimer│   │  Future   │
│Service │      │ Cancer │   │  Service│   │ Services  │
│  8000  │      │  8001  │   │   8002  │   │    ...    │
└────────┘      └────────┘   └─────────┘   └───────────┘

All services built with medical_ml_sdk:
  • BaseModelLoader (scikit-learn model management)
  • BaseServiceConfig (standardized configuration)
  • RegistryClient (automatic registration)
  • Common health checks & monitoring
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
cd medical_ml

# Set up unified virtual environment
./setup_env.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Start All Services

```bash
./start_all_services.sh
```

This starts:
- Registry Service (http://localhost:9000)
- Cardiovascular Disease Service (http://localhost:8000)
- Breast Cancer Service (http://localhost:8001)
- Alzheimer's Service (http://localhost:8002)

### 3. Verify Services

```bash
# Check service status
./check_services.sh

# Or manually
curl http://localhost:9000/api/v1/services | jq
curl http://localhost:9000/api/v1/health/all | jq
```

### 4. Make Predictions

```bash
# CVD prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age_years": 55,
    "gender": 2,
    "height": 170,
    "weight": 80,
    "ap_hi": 130,
    "ap_lo": 85,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
  }'

# Breast Cancer prediction
curl -X POST http://localhost:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @breast_cancer/test_data.json

# Alzheimer's prediction
curl -X POST http://localhost:8002/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @alzheimers/test_data.json
```

## Testing

### Postman Collection

Import and run the comprehensive test suite:

```bash
# Import Medical_ML_Registry.postman_collection.json into Postman
# Run all 53 tests covering:
#  - Registry functionality
#  - Service discovery
#  - All prediction endpoints
#  - Health monitoring
#  - Tag-based search
```

### Automated Tests

```bash
# Run all tests
./run_tests.sh

# Smoke test only (fastest)
./run_tests.sh --smoke-only

# With interactive demo
./run_tests.sh --demo
```

See [HOW_TO_RUN_TESTS.md](HOW_TO_RUN_TESTS.md) and [QUICK_START.md](QUICK_START.md) for detailed testing instructions.

## API Documentation

Each service provides interactive Swagger documentation:

- **Registry**: http://localhost:9000/docs
- **CVD Service**: http://localhost:8000/docs
- **Breast Cancer**: http://localhost:8001/docs
- **Alzheimer's**: http://localhost:8002/docs

## Project Structure

```
medical_ml/
├── registry/              # Service registry (port 9000)
│   └── backend/
│       ├── main.py       # Registry API
│       ├── routes/       # Service registration & discovery
│       └── storage/      # In-memory service catalog
│
├── medical_ml_sdk/       # Shared SDK framework
│   ├── core/            # Base classes (ModelLoader, Config)
│   └── plugin/          # Registry client
│
├── cardiovascular_disease/  # CVD service (port 8000)
│   ├── backend/
│   │   ├── main.py      # FastAPI application
│   │   ├── models/      # Pydantic schemas & model loader
│   │   ├── routes/      # Prediction endpoints
│   │   └── utils/       # Feature engineering (18 features)
│   ├── models/          # Trained model files
│   └── training/        # Model training scripts
│
├── breast_cancer/       # Breast cancer service (port 8001)
│   ├── backend/
│   ├── models/
│   └── training/
│
├── alzheimers/          # Alzheimer's service (port 8002)
│   ├── backend/
│   ├── models/
│   └── training/
│
├── start_all_services.sh    # Start all services
├── stop_all_services.sh     # Stop all services
├── check_services.sh        # Check service status
└── Medical_ML_Registry.postman_collection.json  # API tests
```

## Key Features

### Unified Service Registry
- **Automatic Discovery**: Services auto-register on startup, unregister on shutdown
- **Active Health Checks**: Registry actively queries service health endpoints (no heartbeats needed)
- **Tag-based Search**: Find services by medical condition, model type, etc.
- **Dynamic Schemas**: Services expose input/output schemas for dynamic form generation

### Medical ML SDK
- **BaseModelLoader**: Load scikit-learn models with metadata
- **BaseServiceConfig**: Standardized service configuration
- **RegistryClient**: Automatic registration with retry logic
- **Common Patterns**: Health checks, model info endpoints, error handling

### Engineered Features
- **CVD**: Pulse pressure, mean arterial pressure, hypertension staging, BMI categories, age groups, composite risk scores
- **Breast Cancer**: Mean/SE/Worst values of 10 tumor measurements
- **Alzheimer's**: Cognitive scores (MMSE, CDR) + brain volume metrics

## Management Commands

```bash
# Start everything
./start_all_services.sh

# Check status
./check_services.sh

# View logs
tail -f logs/registry.log
tail -f logs/cvd.log

# Stop everything
./stop_all_services.sh

# List registered services
curl http://localhost:9000/api/v1/services | jq

# Check aggregate health
curl http://localhost:9000/api/v1/health/all | jq

# Search by tag
curl "http://localhost:9000/api/v1/services/search/by-tags?tags=cardiovascular" | jq
```

## Development

### Adding a New Service

1. Create service directory
2. Install medical_ml_sdk: `pip install -e medical_ml_sdk`
3. Extend BaseModelLoader and BaseServiceConfig
4. Configure auto-registration in main.py
5. Add to start_all_services.sh
6. Service will auto-register on startup

Example:
```python
from medical_ml_sdk.core import BaseModelLoader, BaseServiceConfig
from medical_ml_sdk.plugin import RegistryClient

class MyServiceConfig(BaseServiceConfig):
    SERVICE_ID = "my_service"
    SERVICE_NAME = "My Medical ML API"
    PORT = 8003
    AUTO_REGISTER = True
```

### Training Models

```bash
# Train CVD model
cd cardiovascular_disease/training
python train_model.py

# Train Breast Cancer model
cd breast_cancer/training
python train_model.py

# Train Alzheimer's model
cd alzheimers/training
python train_model.py
```

## Technologies

- **Framework**: FastAPI (async Python web framework)
- **ML**: scikit-learn (Random Forest, SVM, Gradient Boosting)
- **Validation**: Pydantic (request/response schemas)
- **HTTP Client**: httpx (async HTTP for health checks)
- **Testing**: Postman + pytest
- **Deployment**: Uvicorn ASGI server

## Datasets

- **Cardiovascular Disease**: Kaggle cardiovascular dataset (70,000 patients)
- **Breast Cancer**: Wisconsin Breast Cancer dataset (569 samples)
- **Alzheimer's**: OASIS brain imaging dataset (MRI + cognitive assessments)

## License

Proprietary - Medical ML Project

## Support

For issues or questions:
- Check service logs in `logs/` directory
- Review API documentation at `/docs` endpoints
- See [QUICK_START.md](QUICK_START.md) for troubleshooting
- Run `./check_services.sh` to verify service status

---

**Last Updated**: December 4, 2025
**Status**: ✅ Production Ready - All 3 services operational
