# Medical ML Documentation Index

Quick reference guide to all project documentation.

## Quick Start

📘 **[README.md](README.md)**
Main project overview with architecture, features, and quick start guide.

📗 **[QUICK_START.md](QUICK_START.md)**
Step-by-step guide to starting services, testing, and managing the system.

📙 **[HOW_TO_RUN_TESTS.md](HOW_TO_RUN_TESTS.md)**
Comprehensive testing guide including Postman collection (53 tests).

🔌 **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)**
Complete API reference for frontend development with examples and workflows.

## Service Documentation

Each service has its own README with specific implementation details:

- **[registry/README.md](registry/README.md)** - Service Registry documentation
- **[cardiovascular_disease/README.md](cardiovascular_disease/README.md)** - CVD Service docs
- **[breast_cancer/README.md](breast_cancer/README.md)** - Breast Cancer Service docs
- **[alzheimers/README.md](alzheimers/README.md)** - Alzheimer's Service docs

## SDK Documentation

📕 **[medical_ml_sdk/README.md](medical_ml_sdk/README.md)** (also at shared/README.md)
Shared SDK framework documentation with examples for extending base classes.

## Technical References

📊 **[cardiovascular_disease/training/FEATURE_ANALYSIS_REPORT.md](cardiovascular_disease/training/FEATURE_ANALYSIS_REPORT.md)**
Detailed analysis of the 18 engineered features for CVD prediction.

## Testing

🧪 **Postman Collection**: `Medical_ML_Registry.postman_collection.json`
- 53 tests across 18 requests
- Tests all services, registry, and end-to-end workflows
- Import into Postman and run against running services

## Getting Started

**For Users:**
1. **[README.md](README.md)** - Understand the architecture
2. **[QUICK_START.md](QUICK_START.md)** - Get services running
3. **[HOW_TO_RUN_TESTS.md](HOW_TO_RUN_TESTS.md)** - Verify everything works

**For Frontend Developers:**
1. **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - Complete API reference with examples
2. **[README.md](README.md)** - Understand the service architecture
3. **Interactive Docs** - http://localhost:9000/docs (when services are running)

**For Backend Developers:**
1. **[README.md](README.md)** - Architecture overview
2. **Service READMEs** - Deep dive into specific services
3. **[medical_ml_sdk/README.md](medical_ml_sdk/README.md)** - Learn to add new services

## Common Tasks

| Task | Documentation |
|------|--------------|
| Start services | [QUICK_START.md](QUICK_START.md) |
| Run tests | [HOW_TO_RUN_TESTS.md](HOW_TO_RUN_TESTS.md) |
| Build frontend | [API_DOCUMENTATION.md](API_DOCUMENTATION.md) |
| Call APIs | [API_DOCUMENTATION.md](API_DOCUMENTATION.md#frontend-integration-guide) |
| Add new service | [README.md](README.md#adding-a-new-service) |
| Troubleshoot | [QUICK_START.md](QUICK_START.md#troubleshooting) |
| Train models | [README.md](README.md#training-models) |
| View API docs | http://localhost:9000/docs (when running) |

## Management Scripts

Quick reference:

```bash
./setup_env.sh              # Setup unified environment
./start_all_services.sh     # Start all services
./stop_all_services.sh      # Stop all services
./check_services.sh         # Check service status
```

## Last Updated

December 4, 2025

All documentation has been updated to reflect:
- ✅ 3 operational services (CVD, Breast Cancer, Alzheimer's)
- ✅ Unified virtual environment setup
- ✅ Active health check system (no heartbeats)
- ✅ 53-test Postman collection
- ✅ medical_ml_sdk framework
- ✅ 18-feature CVD model with engineered features
- ✅ Complete API documentation for frontend developers
