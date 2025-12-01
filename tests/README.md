# Medical ML Service Registry - Test Suite

Comprehensive test suite for the unified medical ML service registry system.

## Overview

This test suite validates:
- ✅ Service registry functionality
- ✅ Service auto-registration
- ✅ Service discovery via the unified interface
- ✅ Making predictions through all services
- ✅ Health monitoring
- ✅ Tag-based service search

## Test Files

### 1. `test_integration.py`
**Comprehensive pytest integration tests**

Tests all aspects of the system:
- Service startup and registration
- Service discovery endpoints
- Making predictions to services
- Health checks
- End-to-end workflows

**Run with:**
```bash
pytest test_integration.py -v -s
```

### 2. `smoke_test.sh`
**Quick validation bash script**

Validates basic functionality:
- Registry starts successfully
- Services start and register
- Service discovery works
- Predictions can be made
- Health checks pass

**Run with:**
```bash
./smoke_test.sh
```

### 3. `demo_unified_interface.py`
**Interactive demonstration script**

Shows how a frontend would use the system:
- Discover available services
- Get service schemas
- Make predictions
- Monitor health
- Search by tags

**Run with:**
```bash
python3 demo_unified_interface.py
```

### 4. `test_data.py`
**Test fixtures and sample data**

Contains sample input data for all services.

### 5. `conftest.py`
**Pytest configuration and fixtures**

Manages service lifecycle:
- Starts registry service
- Starts medical ML services
- Cleans up after tests

## Running Tests

### Quick Start

Run all tests using the master test runner:

```bash
cd /path/to/medical_ml
./run_tests.sh
```

### Options

```bash
# Run only smoke tests (fastest)
./run_tests.sh --smoke-only

# Run only integration tests
./run_tests.sh --integration-only

# Run tests and demo
./run_tests.sh --demo

# Get help
./run_tests.sh --help
```

### Individual Test Runs

**Smoke Test:**
```bash
cd tests
./smoke_test.sh
```

**Integration Tests:**
```bash
cd tests
pytest test_integration.py -v
```

**Demo:**
```bash
cd tests
python3 demo_unified_interface.py
```

## Requirements

Install test dependencies:

```bash
pip install -r tests/requirements.txt
```

Dependencies:
- `pytest>=7.0.0`
- `pytest-asyncio>=0.21.0`
- `requests>=2.28.0`
- `httpx>=0.24.0`

## Test Flow

### 1. Automated Test Flow (pytest)

```
Start Registry (port 9000)
    ↓
Start CVD Service (port 8000)
    ↓
Wait for Services to Register
    ↓
Run Test Cases:
  ✓ Service startup
  ✓ Service registration
  ✓ Service discovery
  ✓ Unified interface
  ✓ Predictions
  ✓ Health checks
    ↓
Cleanup (stop all services)
```

### 2. Smoke Test Flow (bash)

```
1. Start registry
2. Start CVD service
3. Verify service registration
4. Test service discovery
5. Test health checks
6. Test prediction
7. Cleanup
```

## Expected Results

### Successful Test Run

```
✓ Registry service is running
✓ CVD service is running and registered
✓ Service discovery working
✓ Health checks passing
✓ Prediction endpoint responding
✓ All integration tests passed
```

### Test Coverage

The test suite validates:

**Registry Functionality:**
- [x] Service registration
- [x] Service unregistration
- [x] Service discovery
- [x] Tag-based search
- [x] Health monitoring
- [x] Metadata storage

**Unified Interface:**
- [x] Dynamic service discovery
- [x] Schema retrieval
- [x] Direct service calls
- [x] End-to-end prediction workflow

**Services:**
- [x] Auto-registration on startup
- [x] Auto-unregistration on shutdown
- [x] Health checks
- [x] Prediction endpoints
- [x] Metadata completeness

## Troubleshooting

### Services Don't Start

**Issue:** Services fail to start or timeout

**Solutions:**
- Check if ports 8000, 9000 are available: `lsof -i:8000 -i:9000`
- Kill existing processes: `pkill -f uvicorn`
- Check service logs in test output

### Services Don't Register

**Issue:** Registry shows 0 services

**Solutions:**
- Ensure `AUTO_REGISTER=True` in service config
- Check registry is running: `curl http://localhost:9000/health`
- Wait longer for registration (may take 2-3 seconds)
- Check service logs for registration errors

### Predictions Fail

**Issue:** Prediction endpoint returns errors

**Solutions:**
- Check if model files exist in `models/` directory
- Verify `latest_model_info.json` is present
- Model may not be loaded (service will still work, just degraded)
- Check model compatibility (numpy version issues)

### Tests Hang

**Issue:** Tests never complete

**Solutions:**
- Use Ctrl+C to interrupt
- Manually kill services: `pkill -9 -f uvicorn`
- Clean up ports: `lsof -ti:8000 -ti:9000 | xargs kill -9`

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test Medical ML Registry

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r shared/requirements.txt
          pip install -r registry/backend/requirements.txt
          pip install -r tests/requirements.txt
      - name: Run tests
        run: ./run_tests.sh
```

## Next Steps

After tests pass:

1. **Add More Services**: Tests automatically discover and test new services
2. **Extend Test Data**: Add more test cases in `test_data.py`
3. **Load Testing**: Create `test_load.py` for performance testing
4. **E2E Tests**: Add frontend integration tests
5. **Monitoring**: Set up continuous health checking

## Support

For issues or questions:
1. Check test output for specific error messages
2. Review service logs
3. Ensure all dependencies are installed
4. Verify ports are available

---

**Last Updated:** 2025-11-30
**Test Coverage:** Registry + CVD Service
**Status:** ✅ All tests passing
