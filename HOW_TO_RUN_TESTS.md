# How to Run Tests - Medical ML Service Registry

## Prerequisites

1. **Install test dependencies:**
```bash
pip install -r tests/requirements.txt
```

2. **Install the shared SDK:**
```bash
pip install -e shared/
```

3. **Ensure ports 8000 and 9000 are available:**
```bash
# Check if ports are in use
lsof -i:8000 -i:9000

# Kill processes if needed
lsof -ti:8000 -ti:9000 | xargs kill -9
```

## Option 1: Run All Tests (Recommended)

**Quick validation with smoke tests + comprehensive integration tests:**

```bash
cd /Users/zainbaq/Documents/Projects/medical_ml
./run_tests.sh
```

This will:
1. Run smoke tests (~30 seconds)
2. Run pytest integration tests (~2 minutes)
3. Show summary of results

## Option 2: Smoke Test Only (Fastest - ~30 seconds)

**Quick validation that everything works:**

```bash
cd /Users/zainbaq/Documents/Projects/medical_ml
./run_tests.sh --smoke-only
```

This validates:
- Registry starts successfully
- CVD service starts and registers
- Service discovery works
- Health checks pass
- Predictions can be made

## Option 3: Integration Tests Only (~2 minutes)

**Comprehensive pytest test suite:**

```bash
cd /Users/zainbaq/Documents/Projects/medical_ml
./run_tests.sh --integration-only
```

This tests:
- All service startup sequences
- Complete registration workflow
- All discovery endpoints
- Unified interface patterns
- Tag-based search
- Aggregate health checks
- End-to-end workflows

## Option 4: Run with Demo (Interactive)

**See the unified interface in action:**

```bash
cd /Users/zainbaq/Documents/Projects/medical_ml
./run_tests.sh --demo
```

This runs all tests + an interactive demo showing:
- Service discovery workflow
- Service catalog display
- Schema examination
- Making predictions
- Health monitoring
- Tag-based search

## Option 5: Run Individual Test Scripts

### Smoke Test Only
```bash
cd /Users/zainbaq/Documents/Projects/medical_ml/tests
chmod +x smoke_test.sh
./smoke_test.sh
```

### Pytest Integration Tests Only
```bash
cd /Users/zainbaq/Documents/Projects/medical_ml/tests
pytest test_integration.py -v -s
```

### Demo Only
```bash
cd /Users/zainbaq/Documents/Projects/medical_ml/tests
python3 demo_unified_interface.py
```

## Manual Testing (Without Test Scripts)

If you want to test manually:

### 1. Start the Registry
```bash
cd /Users/zainbaq/Documents/Projects/medical_ml/registry/backend
uvicorn main:app --host 0.0.0.0 --port 9000 --reload
```

### 2. Start the CVD Service (in a new terminal)
```bash
cd /Users/zainbaq/Documents/Projects/medical_ml/cardiovascular_disease
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test endpoints manually
```bash
# Check registry health
curl http://localhost:9000/health

# List registered services
curl http://localhost:9000/api/v1/services

# Get specific service
curl http://localhost:9000/api/v1/services/cardiovascular_disease

# Make a prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age_years": 55.0,
    "gender": 2,
    "height": 170.0,
    "weight": 80.0,
    "ap_hi": 130,
    "ap_lo": 85,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
  }'
```

## Using Postman (Recommended)

A comprehensive Postman collection with 53 tests is provided: `Medical_ML_Registry.postman_collection.json`

### Import the collection:
1. Open Postman
2. Click "Import" button (top left)
3. Select `Medical_ML_Registry.postman_collection.json`
4. The collection will appear in your Collections sidebar

### Start Services:
```bash
./start_all_services.sh
```

### Run the collection:
1. Click on "Medical ML Service Registry" collection
2. Click "Run" button (top right)
3. Select all requests or specific test sections
4. Click "Run Medical ML Service Registry"
5. View detailed results with pass/fail for each assertion

### What's tested (53 tests across 18 requests):
- **Registry Service**: Health, service listing, discovery, tag search, aggregate health (8 tests)
- **CVD Service**: Health checks, low/high risk predictions (8 tests)
- **Breast Cancer Service**: Health checks, benign/malignant predictions (5 tests)
- **Alzheimer's Service**: Health checks, dementia/no-dementia predictions (5 tests)
- **End-to-End Workflows**: Service discovery, schema retrieval, health monitoring (3 tests)

### Expected Results:
- ✅ All 53 tests should pass
- Status codes validated (200 OK for success)
- Response schemas validated (required fields present)
- Business logic validated (predictions in expected ranges)
- Health monitoring validated (services report healthy)

## Troubleshooting

### Ports Already in Use
```bash
# Find and kill processes
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
# Install all dependencies
pip install -r shared/requirements.txt
pip install -r registry/backend/requirements.txt
pip install -r cardiovascular_disease/backend/requirements.txt
pip install -r tests/requirements.txt

# Install SDK
pip install -e shared/
```

### ImportError for medical_ml_sdk
```bash
# Make sure SDK is installed in editable mode
pip install -e shared/
```

## Expected Results

### Successful Test Run
You should see:
- ✅ Registry service is running
- ✅ CVD service is running and registered
- ✅ Service discovery working
- ✅ Health checks passing
- ✅ Prediction endpoint responding
- ✅ All tests passed

### Test Output Example
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

[4/6] Testing Service Discovery...
✓ Discovered: Cardiovascular Disease Prediction API

[5/6] Testing Health Checks...
✓ Registry Health: healthy
✓ CVD Service Health: healthy

[6/6] Testing Prediction Endpoint...
✓ Prediction successful!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            ✓ ALL SMOKE TESTS PASSED!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Quick Start Recommendation

For your first test run:

```bash
cd /Users/zainbaq/Documents/Projects/medical_ml
./run_tests.sh --demo
```

This will run all tests and show you an interactive demo of how the unified interface works!
