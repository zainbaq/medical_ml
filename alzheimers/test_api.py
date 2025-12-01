"""
API testing script for Alzheimer's disease prediction API
"""
import requests
import json

BASE_URL = "http://localhost:8001"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("Testing Health Endpoint")
    print("="*70)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_prediction():
    """Test prediction endpoint with sample data"""
    print("\n" + "="*70)
    print("Testing Prediction Endpoint")
    print("="*70)

    # Test case 1: Non-demented patient (low risk)
    patient_1 = {
        "age": 75,
        "gender": "M",
        "EDUC": 14,
        "SES": 2,
        "MMSE": 30,
        "CDR": 0.0,
        "eTIV": 1500,
        "nWBV": 0.75,
        "ASF": 1.2
    }

    print("\nTest Case 1: Non-demented patient (high MMSE, CDR=0)")
    print(f"Input: {json.dumps(patient_1, indent=2)}")
    response = requests.post(f"{BASE_URL}/api/v1/predict", json=patient_1)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Test case 2: Potentially demented patient (SES omitted)
    patient_2 = {
        "age": 80,
        "gender": "F",
        "EDUC": 12,
        "MMSE": 20,
        "CDR": 1.0,
        "eTIV": 1400,
        "nWBV": 0.68,
        "ASF": 1.3
    }

    print("\nTest Case 2: Potentially demented patient (SES omitted, low MMSE, CDR=1)")
    print(f"Input: {json.dumps(patient_2, indent=2)}")
    response = requests.post(f"{BASE_URL}/api/v1/predict", json=patient_2)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Test case 3: Moderate dementia
    patient_3 = {
        "age": 85,
        "gender": "M",
        "EDUC": 10,
        "SES": 3,
        "MMSE": 15,
        "CDR": 2.0,
        "eTIV": 1350,
        "nWBV": 0.65,
        "ASF": 1.35
    }

    print("\nTest Case 3: Moderate dementia (low MMSE, CDR=2)")
    print(f"Input: {json.dumps(patient_3, indent=2)}")
    response = requests.post(f"{BASE_URL}/api/v1/predict", json=patient_3)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*70)
    print("Testing Model Info Endpoint")
    print("="*70)
    response = requests.get(f"{BASE_URL}/api/v1/model-info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_invalid_input():
    """Test validation with invalid input"""
    print("\n" + "="*70)
    print("Testing Validation with Invalid Input")
    print("="*70)

    # Invalid CDR value
    invalid_patient = {
        "age": 75,
        "gender": "M",
        "EDUC": 14,
        "SES": 2,
        "MMSE": 28,
        "CDR": 1.5,  # Invalid - must be 0, 0.5, 1, 2, or 3
        "eTIV": 1500,
        "nWBV": 0.75,
        "ASF": 1.2
    }

    print("\nTest Case: Invalid CDR value (1.5)")
    print(f"Input: {json.dumps(invalid_patient, indent=2)}")
    response = requests.post(f"{BASE_URL}/api/v1/predict", json=invalid_patient)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Alzheimer's Prediction API Test Suite")
    print("="*70)

    try:
        test_health()
        test_prediction()
        test_model_info()
        test_invalid_input()

        print("\n" + "="*70)
        print("All tests completed!")
        print("="*70)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the API is running:")
        print("  ./start_api.sh")
