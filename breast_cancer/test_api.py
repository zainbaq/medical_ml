"""
Basic test script for the Breast Cancer Prediction API
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

    # Test data - Malignant tumor (from actual dataset)
    test_malignant = {
        "radius_mean": 17.99,
        "texture_mean": 10.38,
        "perimeter_mean": 122.8,
        "area_mean": 1001.0,
        "smoothness_mean": 0.1184,
        "compactness_mean": 0.2776,
        "concavity_mean": 0.3001,
        "concave points_mean": 0.1471,
        "symmetry_mean": 0.2419,
        "fractal_dimension_mean": 0.07871,
        "radius_se": 1.095,
        "texture_se": 0.9053,
        "perimeter_se": 8.589,
        "area_se": 153.4,
        "smoothness_se": 0.006399,
        "compactness_se": 0.04904,
        "concavity_se": 0.05373,
        "concave points_se": 0.01587,
        "symmetry_se": 0.03003,
        "fractal_dimension_se": 0.006193,
        "radius_worst": 25.38,
        "texture_worst": 17.33,
        "perimeter_worst": 184.6,
        "area_worst": 2019.0,
        "smoothness_worst": 0.1622,
        "compactness_worst": 0.6656,
        "concavity_worst": 0.7119,
        "concave points_worst": 0.2654,
        "symmetry_worst": 0.4601,
        "fractal_dimension_worst": 0.1189
    }

    print("\nTest Case 1: Malignant Tumor Sample")
    print(f"Input: Features from actual malignant case")

    response = requests.post(f"{base_url}/api/v1/predict", json=test_malignant)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Test data - Benign tumor (smaller values, less aggressive features)
    test_benign = {
        "radius_mean": 11.42,
        "texture_mean": 18.61,
        "perimeter_mean": 72.74,
        "area_mean": 402.0,
        "smoothness_mean": 0.08683,
        "compactness_mean": 0.05456,
        "concavity_mean": 0.02948,
        "concave points_mean": 0.01525,
        "symmetry_mean": 0.1628,
        "fractal_dimension_mean": 0.05781,
        "radius_se": 0.2027,
        "texture_se": 0.7157,
        "perimeter_se": 1.33,
        "area_se": 14.45,
        "smoothness_se": 0.004477,
        "compactness_se": 0.007994,
        "concavity_se": 0.01145,
        "concave points_se": 0.00482,
        "symmetry_se": 0.01485,
        "fractal_dimension_se": 0.001514,
        "radius_worst": 12.73,
        "texture_worst": 24.31,
        "perimeter_worst": 81.37,
        "area_worst": 496.5,
        "smoothness_worst": 0.1098,
        "compactness_worst": 0.1266,
        "concavity_worst": 0.1407,
        "concave points_worst": 0.05814,
        "symmetry_worst": 0.2574,
        "fractal_dimension_worst": 0.07234
    }

    print("\n" + "-"*60)
    print("Test Case 2: Benign Tumor Sample")
    print(f"Input: Features from actual benign case")

    response = requests.post(f"{base_url}/api/v1/predict", json=test_benign)
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


def main():
    """Run all tests"""
    base_url = "http://localhost:8000"

    print("\n" + "="*60)
    print("BREAST CANCER PREDICTION API - TEST SUITE")
    print("="*60)
    print(f"Testing API at: {base_url}")

    try:
        # Test health endpoint
        health_ok = test_health_endpoint(base_url)

        # Test prediction endpoint
        prediction_ok = test_prediction_endpoint(base_url)

        # Test model info endpoint
        model_info_ok = test_model_info_endpoint(base_url)

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Health Endpoint: {'✓ PASSED' if health_ok else '✗ FAILED'}")
        print(f"Prediction Endpoint: {'✓ PASSED' if prediction_ok else '✗ FAILED'}")
        print(f"Model Info Endpoint: {'✓ PASSED' if model_info_ok else '✗ FAILED'}")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to API")
        print("Make sure the API is running at http://localhost:8000")
        print("Run: cd breast_cancer && ./start_api.sh")


if __name__ == "__main__":
    main()
