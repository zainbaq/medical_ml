# Cardiovascular Disease API - Testing Guide

## Overview
This guide provides instructions for testing the Cardiovascular Disease Prediction API.

## Model Information
- **Model Type**: Gradient Boosting Classifier
- **ROC-AUC Score**: 80.3%
- **Accuracy**: 73.6%
- **Trained**: 2025-11-30

## Starting the API

```bash
cd cardiovascular_disease
./start_api.sh
```

Or manually:
```bash
cd cardiovascular_disease/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

## API Documentation
Once the API is running, visit:
- **Interactive Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

## Available Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Root Endpoint
```bash
curl http://localhost:8000/
```

### 3. Prediction
```bash
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

### 4. Model Info
```bash
curl http://localhost:8000/api/v1/model-info
```

## Test Scripts

### 1. Basic API Test (test_api.py)
Quick test of core functionality with high/low risk patients.

```bash
python test_api.py
```

**Features:**
- Health check test
- High risk patient prediction
- Low risk patient prediction
- Model info endpoint test
- Input validation test

### 2. Comprehensive Test Suite (test_comprehensive.py)
Extensive testing with 12+ test cases including edge cases and validation.

```bash
python test_comprehensive.py
```

**Test Coverage:**
- ✓ Health check endpoint
- ✓ Model info endpoint
- ✓ High risk patient predictions
- ✓ Low risk patient predictions
- ✓ Medium risk patient predictions
- ✓ Minimum value edge cases
- ✓ Maximum value edge cases
- ✓ Age validation
- ✓ Blood pressure validation
- ✓ Missing field validation
- ✓ Obese patient scenarios
- ✓ Hypertensive patient scenarios

### 3. Load Testing (test_load.py)
Performance testing with concurrent requests.

```bash
python test_load.py
```

**Features:**
- Interactive scenario selection (Light/Moderate/Heavy load)
- Concurrent request handling
- Response time statistics (min, max, mean, median, percentiles)
- Throughput measurement (requests/second)
- Success rate tracking
- Error analysis
- Prediction distribution analysis

**Available Scenarios:**
1. Light Load: 50 requests, 5 concurrent workers
2. Moderate Load: 100 requests, 10 concurrent workers
3. Heavy Load: 200 requests, 20 concurrent workers
4. Custom: User-defined parameters

### 4. Test Data Generator (generate_test_data.py)
Generate realistic patient profiles for testing.

```bash
python generate_test_data.py
```

**Generates:**
- 10 low-risk patient profiles
- 10 medium-risk patient profiles
- 10 high-risk patient profiles
- 8 specific medical scenario test cases
- Saves to `test_patients.json`

## Input Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| age_years | float | 18-100 | Age in years |
| gender | int | 1-2 | 1=female, 2=male |
| height | float | 140-210 | Height in cm |
| weight | float | 40-200 | Weight in kg |
| ap_hi | int | 80-200 | Systolic blood pressure |
| ap_lo | int | 60-130 | Diastolic blood pressure |
| cholesterol | int | 1-3 | 1=normal, 2=above normal, 3=well above normal |
| gluc | int | 1-3 | 1=normal, 2=above normal, 3=well above normal |
| smoke | int | 0-1 | 0=no, 1=yes |
| alco | int | 0-1 | 0=no, 1=yes |
| active | int | 0-1 | 0=no, 1=yes |

## Response Format

```json
{
  "prediction": 1,
  "probability": 0.7234,
  "risk_level": "high",
  "bmi": 27.68
}
```

- **prediction**: 0 (no disease) or 1 (disease present)
- **probability**: Probability of cardiovascular disease (0-1)
- **risk_level**: "low", "medium", or "high"
- **bmi**: Calculated Body Mass Index

## Risk Level Thresholds

- **Low Risk**: probability < 0.3
- **Medium Risk**: 0.3 ≤ probability < 0.7
- **High Risk**: probability ≥ 0.7

## Common Test Scenarios

### High Risk Patient
```json
{
  "age_years": 65.0,
  "gender": 2,
  "height": 170.0,
  "weight": 95.0,
  "ap_hi": 170,
  "ap_lo": 105,
  "cholesterol": 3,
  "gluc": 3,
  "smoke": 1,
  "alco": 1,
  "active": 0
}
```

### Low Risk Patient
```json
{
  "age_years": 25.0,
  "gender": 1,
  "height": 165.0,
  "weight": 60.0,
  "ap_hi": 110,
  "ap_lo": 70,
  "cholesterol": 1,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1
}
```

## Troubleshooting

### API Connection Error
```
✗ ERROR: Could not connect to API
```
**Solution**: Ensure the API is running:
```bash
cd cardiovascular_disease && ./start_api.sh
```

### Model Not Loaded
```
{
  "detail": "Model not loaded. Please train a model first."
}
```
**Solution**: Train the model first:
```bash
cd cardiovascular_disease
./train.sh
```

### Validation Error (422)
```
{
  "detail": [
    {
      "loc": ["body", "ap_lo"],
      "msg": "Diastolic BP must be less than systolic BP",
      "type": "value_error"
    }
  ]
}
```
**Solution**: Check that input values meet validation requirements (e.g., ap_lo < ap_hi)

## Performance Benchmarks

Expected performance on a standard machine:
- **Response Time**: < 100ms (average)
- **Throughput**: 50-200 requests/second (depending on concurrency)
- **Success Rate**: > 99%

## Next Steps

1. Run basic tests to verify API functionality
2. Run comprehensive tests to ensure edge cases are handled
3. Run load tests to verify performance under load
4. Use the API in your application
5. Monitor performance in production

## Support

For issues or questions:
- Check the API documentation at http://localhost:8000/docs
- Review the model metadata at the /model-info endpoint
- Check logs for detailed error messages
