"""
Comprehensive test suite for Cardiovascular Disease Prediction API
Tests various scenarios including edge cases and error conditions
"""
import requests
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1"


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_test_case(name: str):
    """Print test case name"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}Test Case: {name}{Colors.RESET}")
    print(f"{Colors.YELLOW}{'-'*70}{Colors.RESET}")


def print_result(passed: bool, message: str = ""):
    """Print test result"""
    if passed:
        print(f"{Colors.GREEN}✓ PASSED{Colors.RESET} {message}")
    else:
        print(f"{Colors.RED}✗ FAILED{Colors.RESET} {message}")
    return passed


def test_health_check() -> bool:
    """Test 1: Health check endpoint"""
    print_test_case("Health Check Endpoint")

    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        success = (
            response.status_code == 200 and
            data.get('model_loaded') == True and
            'version' in data
        )

        return print_result(success, "Health check working")
    except Exception as e:
        print(f"Error: {str(e)}")
        return print_result(False, f"Exception: {str(e)}")


def test_model_info() -> bool:
    """Test 2: Model info endpoint"""
    print_test_case("Model Information Endpoint")

    try:
        response = requests.get(f"{API_URL}/model-info")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")

        success = (
            response.status_code == 200 and
            'model_name' in data and
            'metrics' in data
        )

        return print_result(success, "Model info retrieved")
    except Exception as e:
        print(f"Error: {str(e)}")
        return print_result(False, f"Exception: {str(e)}")


def test_prediction(patient_data: Dict[str, Any], expected_risk: str = None) -> bool:
    """Generic prediction test"""
    try:
        response = requests.post(f"{API_URL}/predict", json=patient_data)
        print(f"Input: {json.dumps(patient_data, indent=2)}")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            has_required_fields = all(
                key in data for key in ['prediction', 'probability', 'risk_level', 'bmi']
            )

            if expected_risk:
                risk_matches = data.get('risk_level') == expected_risk
                return print_result(
                    has_required_fields and risk_matches,
                    f"Risk level: {data.get('risk_level')}"
                )
            else:
                return print_result(has_required_fields, f"Risk level: {data.get('risk_level')}")
        else:
            print(f"Response: {response.text}")
            return print_result(False, f"Unexpected status code: {response.status_code}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return print_result(False, f"Exception: {str(e)}")


def test_high_risk_patient() -> bool:
    """Test 3: High risk patient prediction"""
    print_test_case("High Risk Patient - Elderly with Multiple Risk Factors")

    patient = {
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

    return test_prediction(patient)


def test_low_risk_patient() -> bool:
    """Test 4: Low risk patient prediction"""
    print_test_case("Low Risk Patient - Young and Healthy")

    patient = {
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

    return test_prediction(patient)


def test_medium_risk_patient() -> bool:
    """Test 5: Medium risk patient prediction"""
    print_test_case("Medium Risk Patient - Middle-aged with Some Risk Factors")

    patient = {
        "age_years": 50.0,
        "gender": 2,
        "height": 175.0,
        "weight": 85.0,
        "ap_hi": 140,
        "ap_lo": 90,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }

    return test_prediction(patient)


def test_edge_case_minimum_values() -> bool:
    """Test 6: Edge case with minimum allowed values"""
    print_test_case("Edge Case - Minimum Allowed Values")

    patient = {
        "age_years": 18.0,
        "gender": 1,
        "height": 140.0,
        "weight": 40.0,
        "ap_hi": 80,
        "ap_lo": 60,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 0
    }

    return test_prediction(patient)


def test_edge_case_maximum_values() -> bool:
    """Test 7: Edge case with maximum allowed values"""
    print_test_case("Edge Case - Maximum Allowed Values")

    patient = {
        "age_years": 100.0,
        "gender": 2,
        "height": 210.0,
        "weight": 200.0,
        "ap_hi": 200,
        "ap_lo": 129,
        "cholesterol": 3,
        "gluc": 3,
        "smoke": 1,
        "alco": 1,
        "active": 1
    }

    return test_prediction(patient)


def test_validation_age_too_young() -> bool:
    """Test 8: Validation - Age too young"""
    print_test_case("Validation Error - Age Below Minimum")

    patient = {
        "age_years": 10.0,  # Too young
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

    try:
        response = requests.post(f"{API_URL}/predict", json=patient)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        success = response.status_code == 422  # Unprocessable Entity
        return print_result(success, "Validation rejected invalid age")
    except Exception as e:
        return print_result(False, f"Exception: {str(e)}")


def test_validation_invalid_bp() -> bool:
    """Test 9: Validation - Diastolic higher than systolic"""
    print_test_case("Validation Error - Invalid Blood Pressure")

    patient = {
        "age_years": 50.0,
        "gender": 2,
        "height": 170.0,
        "weight": 80.0,
        "ap_hi": 120,
        "ap_lo": 130,  # Higher than systolic - invalid
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=patient)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        success = response.status_code == 422
        return print_result(success, "Validation rejected invalid BP")
    except Exception as e:
        return print_result(False, f"Exception: {str(e)}")


def test_validation_missing_field() -> bool:
    """Test 10: Validation - Missing required field"""
    print_test_case("Validation Error - Missing Required Field")

    patient = {
        "age_years": 50.0,
        "gender": 2,
        "height": 170.0,
        # weight is missing
        "ap_hi": 130,
        "ap_lo": 85,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=patient)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        success = response.status_code == 422
        return print_result(success, "Validation rejected missing field")
    except Exception as e:
        return print_result(False, f"Exception: {str(e)}")


def test_obese_patient() -> bool:
    """Test 11: Obese patient (BMI > 30)"""
    print_test_case("Obese Patient with High BMI")

    patient = {
        "age_years": 45.0,
        "gender": 2,
        "height": 170.0,
        "weight": 110.0,  # BMI ~38
        "ap_hi": 145,
        "ap_lo": 95,
        "cholesterol": 2,
        "gluc": 2,
        "smoke": 0,
        "alco": 0,
        "active": 0
    }

    return test_prediction(patient)


def test_hypertensive_patient() -> bool:
    """Test 12: Patient with hypertension"""
    print_test_case("Hypertensive Patient")

    patient = {
        "age_years": 55.0,
        "gender": 1,
        "height": 160.0,
        "weight": 70.0,
        "ap_hi": 180,  # Stage 2 hypertension
        "ap_lo": 110,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }

    return test_prediction(patient)


def main():
    """Run all comprehensive tests"""
    print_header("CARDIOVASCULAR DISEASE API - COMPREHENSIVE TEST SUITE")
    print(f"Testing API at: {BASE_URL}")

    results = []

    try:
        # Basic functionality tests
        results.append(("Health Check", test_health_check()))
        results.append(("Model Info", test_model_info()))

        # Prediction tests - different risk levels
        results.append(("High Risk Patient", test_high_risk_patient()))
        results.append(("Low Risk Patient", test_low_risk_patient()))
        results.append(("Medium Risk Patient", test_medium_risk_patient()))

        # Edge cases
        results.append(("Minimum Values", test_edge_case_minimum_values()))
        results.append(("Maximum Values", test_edge_case_maximum_values()))

        # Validation tests
        results.append(("Age Validation", test_validation_age_too_young()))
        results.append(("BP Validation", test_validation_invalid_bp()))
        results.append(("Missing Field Validation", test_validation_missing_field()))

        # Specific medical conditions
        results.append(("Obese Patient", test_obese_patient()))
        results.append(("Hypertensive Patient", test_hypertensive_patient()))

    except requests.exceptions.ConnectionError:
        print(f"\n{Colors.RED}✗ ERROR: Could not connect to API{Colors.RESET}")
        print(f"Make sure the API is running at {BASE_URL}")
        print("Run: cd cardiovascular_disease && ./start_api.sh")
        return

    # Print summary
    print_header("TEST SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = f"{Colors.GREEN}✓ PASSED{Colors.RESET}" if result else f"{Colors.RED}✗ FAILED{Colors.RESET}"
        print(f"{name:.<50} {status}")

    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.RESET}")

    if passed == total:
        print(f"{Colors.GREEN}All tests passed!{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}{total - passed} test(s) failed{Colors.RESET}")

    print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
