"""
Test script for the Cardiovascular Disease Prediction API
"""
import requests
import json


def test_health_endpoint(base_url):
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)

    response = requests.get(f"{base_url}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_prediction_endpoint(base_url):
    """Test the prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Prediction Endpoint")
    print("="*60)

    # Test data - High risk patient
    test_patient_high_risk = {
        "age_years": 60.0,
        "gender": 2,
        "height": 170.0,
        "weight": 95.0,
        "ap_hi": 160,
        "ap_lo": 100,
        "cholesterol": 3,
        "gluc": 2,
        "smoke": 1,
        "alco": 1,
        "active": 0
    }

    print("\nTest Case 1: High Risk Patient")
    print(f"Input: {json.dumps(test_patient_high_risk, indent=2)}")

    response = requests.post(f"{base_url}/api/v1/predict", json=test_patient_high_risk)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Test data - Low risk patient
    test_patient_low_risk = {
        "age_years": 30.0,
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

    print("\n" + "-"*60)
    print("Test Case 2: Low Risk Patient")
    print(f"Input: {json.dumps(test_patient_low_risk, indent=2)}")

    response = requests.post(f"{base_url}/api/v1/predict", json=test_patient_low_risk)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_model_info_endpoint(base_url):
    """Test the model info endpoint"""
    print("\n" + "="*60)
    print("Testing Model Info Endpoint")
    print("="*60)

    response = requests.get(f"{base_url}/api/v1/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_validation(base_url):
    """Test input validation"""
    print("\n" + "="*60)
    print("Testing Input Validation")
    print("="*60)

    # Invalid data - diastolic > systolic
    invalid_patient = {
        "age_years": 50.0,
        "gender": 2,
        "height": 170.0,
        "weight": 80.0,
        "ap_hi": 120,
        "ap_lo": 130,  # Invalid: higher than systolic
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }

    print("\nTest Case: Invalid Blood Pressure (diastolic > systolic)")
    print(f"Input: {json.dumps(invalid_patient, indent=2)}")

    response = requests.post(f"{base_url}/api/v1/predict", json=invalid_patient)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 422  # Unprocessable Entity


def main():
    """Run all tests"""
    base_url = "http://localhost:8000"

    print("\n" + "="*60)
    print("CARDIOVASCULAR DISEASE API - TEST SUITE")
    print("="*60)
    print(f"Testing API at: {base_url}")

    try:
        # Test health endpoint
        health_ok = test_health_endpoint(base_url)

        # Test prediction endpoint
        prediction_ok = test_prediction_endpoint(base_url)

        # Test model info endpoint
        model_info_ok = test_model_info_endpoint(base_url)

        # Test validation
        validation_ok = test_validation(base_url)

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Health Endpoint: {'✓ PASSED' if health_ok else '✗ FAILED'}")
        print(f"Prediction Endpoint: {'✓ PASSED' if prediction_ok else '✗ FAILED'}")
        print(f"Model Info Endpoint: {'✓ PASSED' if model_info_ok else '✗ FAILED'}")
        print(f"Validation: {'✓ PASSED' if validation_ok else '✗ FAILED'}")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to API")
        print("Make sure the API is running at http://localhost:8000")
        print("Run: cd backend && uvicorn main:app --reload")


if __name__ == "__main__":
    main()
