# Medical ML Service Registry - Quick Start Guide

Complete guide for starting, testing, and managing your unified medical ML service registry.

## Table of Contents

1. [Starting Services](#starting-services)
2. [Testing](#testing)
3. [Managing Services](#managing-services)
4. [API Testing with Postman](#api-testing-with-postman)
5. [Troubleshooting](#troubleshooting)

---

## Starting Services

### One-Command Startup (Recommended)

Start all services (registry + all ML services) with one command:

```bash
./start_all_services.sh
```

This will:
- Start the Registry Service on port 9000
- Start Cardiovascular Disease Service on port 8000
- Start Breast Cancer Service on port 8001 (if available)
- Start Alzheimers Service on port 8002 (if available)
- Wait for all services to be ready
- Show registration status
- Run in background with logs

**What you'll see:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Starting Medical ML Service Registry System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/5] Checking for port conflicts...
   ✓ All ports available

[2/5] Starting Registry Service (port 9000)...
   ✓ Registry is ready!

[3/5] Starting Cardiovascular Disease Service (port 8000)...
   ✓ CVD Service is ready!

✓ Registered Services: 1
  • Cardiovascular Disease Prediction API - http://localhost:8000
```

### Manual Startup (Individual Services)

If you prefer to start services individually:

```bash
# Terminal 1 - Registry
cd registry/backend
uvicorn main:app --host 0.0.0.0 --port 9000 --reload

# Terminal 2 - CVD Service
cd cardiovascular_disease
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3 - Breast Cancer (if available)
cd breast_cancer
uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 4 - Alzheimers (if available)
cd alzheimers
uvicorn backend.main:app --host 0.0.0.0 --port 8002 --reload
```

---

## Testing

### Quick Smoke Test (30 seconds)

Fastest way to validate everything works:

```bash
./run_tests.sh --smoke-only
```

### Comprehensive Test Suite (2 minutes)

Run all tests (smoke + integration):

```bash
./run_tests.sh
```

### Interactive Demo

See the unified interface in action:

```bash
./run_tests.sh --demo
```

This demonstrates:
- Service discovery
- Schema retrieval
- Making predictions
- Health monitoring
- Tag-based search

### Individual Test Scripts

```bash
# Smoke test only
cd tests
./smoke_test.sh

# Integration tests only
cd tests
pytest test_integration.py -v

# Demo only
cd tests
python3 demo_unified_interface.py
```

For detailed testing instructions, see: [HOW_TO_RUN_TESTS.md](HOW_TO_RUN_TESTS.md)

---

## Managing Services

### Check Service Status

See what's currently running:

```bash
./check_services.sh
```

**Example output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Medical ML Service Registry - Status Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Port Status:

  Registry (9000):        ✓ Running (healthy)
  CVD Service (8000):     ✓ Running (healthy)
  Breast Cancer (8001):   ○ Not running
  Alzheimers (8002):      ○ Not running

Registered Services:

  • Cardiovascular Disease Prediction API
    ID: cardiovascular_disease
    URL: http://localhost:8000
    Tags: cardiovascular, disease, classification, health
```

### Stop All Services

```bash
./stop_all_services.sh
```

This will:
- Stop all services gracefully
- Clean up PID files
- Kill any remaining processes on ports 8000-8002, 9000
- Show summary of stopped services

### View Logs

Logs are stored in the `logs/` directory:

```bash
# View registry logs
tail -f logs/registry.log

# View CVD service logs
tail -f logs/cvd.log

# View breast cancer logs
tail -f logs/breast_cancer.log

# View all logs
ls -la logs/
```

---

## API Testing with Postman

### Import the Collection

1. **Open Postman**
2. Click **Import** (top left)
3. Click **Upload Files**
4. Select: `Medical_ML_Registry.postman_collection.json`
5. Click **Import**

### Start Services

Before running the collection, ensure services are running:

```bash
./start_all_services.sh
```

### Run the Collection

1. Click on **"Medical ML Service Registry"** collection
2. Click **"Run"** button (top right)
3. Select all requests (or specific ones you want to test)
4. Click **"Run Medical ML Service Registry"**

### What Gets Tested

The Postman collection includes **16 comprehensive tests**:

**Section 1: Registry Service Tests (6 tests)**
- ✅ Registry health check
- ✅ List all registered services
- ✅ Get service details with schemas
- ✅ Search by tags (cardiovascular)
- ✅ Search by tags (classification)
- ✅ Aggregate health monitoring

**Section 2: CVD Service Tests (6 tests)**
- ✅ CVD service health check
- ✅ Prediction with low risk patient
- ✅ Prediction with medium risk patient
- ✅ Prediction with high risk patient
- ✅ Validation test (missing fields)
- ✅ Validation test (invalid data)

**Section 3: End-to-End Workflow (4 tests)**
- ✅ Discover services (frontend simulation)
- ✅ Get schemas for dynamic forms
- ✅ Make prediction
- ✅ Monitor system health

### Expected Results

- Green checkmarks = All tests passed ✅
- Test assertions validate response structure, status codes, and business logic
- Detailed results shown for each request

---

## Troubleshooting

### Port Already in Use

**Problem:** Services fail to start because ports are in use

**Solution:**
```bash
# Check what's using the ports
lsof -i:8000 -i:8001 -i:8002 -i:9000

# Kill all processes on those ports
./stop_all_services.sh

# Or manually
lsof -ti:8000 -ti:8001 -ti:8002 -ti:9000 | xargs kill -9
```

### Services Not Registering

**Problem:** Services start but don't appear in registry

**Solution:**
1. Check registry is running:
   ```bash
   curl http://localhost:9000/health
   ```

2. Check service config has `AUTO_REGISTER=True`

3. Wait 2-3 seconds for registration to complete

4. Check service logs:
   ```bash
   tail -f logs/cvd.log
   ```

### Tests Fail

**Problem:** Tests hang or fail

**Solution:**
```bash
# Clean up everything
./stop_all_services.sh
pkill -9 -f uvicorn

# Reinstall dependencies
pip install -r tests/requirements.txt
pip install -e shared/

# Try again
./start_all_services.sh
./run_tests.sh --smoke-only
```

### Model Not Loaded

**Problem:** Service health shows `model_loaded: false`

**Solution:**
- This is expected if model files don't exist
- Service will still work in "degraded mode"
- Check if models exist: `ls cardiovascular_disease/models/`
- Train a model or copy existing model files

### Logs Too Large

**Problem:** Log files growing too large

**Solution:**
```bash
# Clear all logs
rm -f logs/*.log

# Or truncate specific log
> logs/registry.log
```

---

## Service URLs

Once services are running:

| Service | Health Check | API Docs | Prediction |
|---------|-------------|----------|------------|
| **Registry** | http://localhost:9000/health | http://localhost:9000/docs | N/A |
| **CVD Service** | http://localhost:8000/health | http://localhost:8000/docs | http://localhost:8000/api/v1/predict |
| **Breast Cancer** | http://localhost:8001/health | http://localhost:8001/docs | http://localhost:8001/api/v1/predict |
| **Alzheimers** | http://localhost:8002/health | http://localhost:8002/docs | http://localhost:8002/api/v1/predict |

---

## Common Commands

```bash
# Start everything
./start_all_services.sh

# Check status
./check_services.sh

# Run tests
./run_tests.sh

# Stop everything
./stop_all_services.sh

# View logs
tail -f logs/registry.log

# List registered services
curl http://localhost:9000/api/v1/services | jq

# Check aggregate health
curl http://localhost:9000/api/v1/health/all | jq

# Make a prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @tests/test_data.json
```

---

## Next Steps

1. **Start the system:**
   ```bash
   ./start_all_services.sh
   ```

2. **Verify everything is working:**
   ```bash
   ./check_services.sh
   ```

3. **Run tests:**
   ```bash
   ./run_tests.sh --demo
   ```

4. **Test APIs with Postman:**
   - Import `Medical_ML_Registry.postman_collection.json`
   - Run the collection

5. **Explore the API:**
   - Open http://localhost:9000/docs
   - Try making predictions
   - Explore service discovery

6. **Add more services:**
   - Migrate breast_cancer and alzheimers to use SDK
   - They'll auto-register when started
   - Tests will automatically discover them

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (Flask)                    │
│                                                         │
│  • Discovers services from registry                    │
│  • Gets schemas for dynamic form generation             │
│  • Makes predictions to services                        │
│  • Monitors health                                      │
└────────────┬────────────────────────────────────────────┘
             │
             │ HTTP
             │
┌────────────▼────────────────────────────────────────────┐
│              Registry Service (Port 9000)               │
│                                                         │
│  • Service registration/discovery                       │
│  • Health monitoring                                    │
│  • Tag-based search                                     │
│  • In-memory service catalog                            │
└────────────┬────────────────────────────────────────────┘
             │
             │ Auto-registration
             │
    ┌────────┴────────┬────────────┬──────────────┐
    │                 │            │              │
┌───▼────┐      ┌────▼───┐   ┌────▼────┐   ┌────▼──────┐
│  CVD   │      │ Breast │   │Alzheimer│   │  Future   │
│Service │      │ Cancer │   │  Service│   │ Services  │
│  8000  │      │  8001  │   │   8002  │   │    ...    │
└────────┘      └────────┘   └─────────┘   └───────────┘

All services use the shared SDK:
  • BaseModelLoader
  • BaseServiceConfig
  • RegistryClient
  • Common schemas
```

---

**Ready to get started?**

```bash
./start_all_services.sh
```

For detailed documentation:
- **Testing:** [HOW_TO_RUN_TESTS.md](HOW_TO_RUN_TESTS.md)
- **Architecture:** [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **API Docs:** http://localhost:9000/docs
