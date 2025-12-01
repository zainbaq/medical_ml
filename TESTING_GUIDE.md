# Testing Guide - Medical ML Service Registry

## Quick Start

### Run All Tests

```bash
cd /Users/zainbaq/Documents/Projects/medical_ml
./run_tests.sh
```

This will automatically:
1. Start the registry service
2. Start all medical ML services
3. Run comprehensive tests
4. Clean up afterwards

## Test Options

### Smoke Test Only (Fastest - ~30 seconds)
```bash
./run_tests.sh --smoke-only
```

### Integration Tests Only (Comprehensive - ~2 minutes)
```bash
./run_tests.sh --integration-only
```

### With Demo (Interactive)
```bash
./run_tests.sh --demo
```

## Individual Test Scripts

### 1. Smoke Test (Quick Validation)
```bash
cd tests
./smoke_test.sh
```

**What it tests:**
- ✅ Registry starts on port 9000
- ✅ CVD service starts on port 8000
- ✅ Service auto-registration
- ✅ Service discovery
- ✅ Health checks
- ✅ Making predictions

### 2. Integration Tests (Comprehensive)
```bash
cd tests
pytest test_integration.py -v -s
```

**What it tests:**
- ✅ All service startup sequences
- ✅ Complete registration workflow
- ✅ All discovery endpoints
- ✅ Unified interface patterns
- ✅ Tag-based search
- ✅ Aggregate health checks
- ✅ End-to-end workflows

### 3. Demo Script (Interactive Showcase)
```bash
cd tests
python3 demo_unified_interface.py
```

**What it demonstrates:**
- 📊 Service discovery workflow
- 📋 Service catalog display
- 📝 Schema examination
- 🎯 Making predictions
- 🏥 Health monitoring
- 🔍 Tag-based search

## Test Files Created

```
medical_ml/
├── run_tests.sh                    # Master test runner
└── tests/
    ├── README.md                   # Detailed test documentation
    ├── requirements.txt            # Test dependencies
    ├── conftest.py                 # Pytest fixtures (service lifecycle)
    ├── test_data.py                # Sample test data
    ├── test_integration.py         # Comprehensive pytest suite
    ├── smoke_test.sh               # Quick bash validation
    └── demo_unified_interface.py   # Interactive demo
```

## Example Output

### Successful Test Run

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Medical ML Service Registry - Smoke Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/6] Starting Registry Service...
✓ Registry is ready!

[2/6] Starting Cardiovascular Disease Service...
✓ CVD Service is ready!

[3/6] Verifying Service Registration...
✓ Found 1 registered service(s)
  - Cardiovascular Disease Prediction API (cardiovascular_disease)

[4/6] Testing Service Discovery...
✓ Discovered: Cardiovascular Disease Prediction API
  URL: http://localhost:8000

[5/6] Testing Health Checks...
✓ Registry Health: healthy
✓ CVD Service Health: healthy

[6/6] Testing Prediction Endpoint...
✓ Prediction successful!
  Prediction: 0
  Probability: 0.23

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            ✓ ALL SMOKE TESTS PASSED!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## What Gets Tested

### Registry Functionality
- [x] Service registration via POST /api/v1/services/register
- [x] Service unregistration via DELETE /api/v1/services/{id}
- [x] List all services via GET /api/v1/services
- [x] Get specific service via GET /api/v1/services/{id}
- [x] Search by tags via GET /api/v1/services/search/by-tags
- [x] Aggregate health via GET /api/v1/health/all
- [x] Registry health via GET /health

### Service Auto-Registration
- [x] Services register on startup
- [x] Services unregister on shutdown
- [x] Complete metadata is stored
- [x] Schemas are captured
- [x] Tags are searchable

### Unified Interface
- [x] Dynamic service discovery
- [x] Schema retrieval
- [x] Making predictions to services
- [x] Health monitoring
- [x] End-to-end workflows

### Service Functionality
- [x] Health checks respond
- [x] Prediction endpoints work
- [x] Services run independently
- [x] Metadata is complete

## Troubleshooting

### Ports Already in Use

```bash
# Find and kill processes on ports 8000, 9000
lsof -ti:8000 -ti:9000 | xargs kill -9
```

### Tests Hang or Fail

```bash
# Clean up everything
pkill -9 -f uvicorn
lsof -ti:8000 -ti:9000 | xargs kill -9

# Try again
./run_tests.sh
```

### Missing Dependencies

```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Install SDK
pip install -e shared/
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Run Tests
  run: |
    pip install -r shared/requirements.txt
    pip install -r tests/requirements.txt
    ./run_tests.sh
```

## Next Steps

1. **Run the tests:**
   ```bash
   ./run_tests.sh --demo
   ```

2. **Watch the demo** to see the unified interface in action

3. **Add more services** - tests will automatically discover and test them

4. **Extend test data** - add more test cases in `tests/test_data.py`

5. **Create load tests** - test system under concurrent load

---

**All test scripts are ready to run!** 🎉

Start with: `./run_tests.sh --demo`
