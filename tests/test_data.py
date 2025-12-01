"""Sample test data for all medical ML services."""

# Cardiovascular Disease test data
CVD_SAMPLE_DATA = {
    "age_years": 55.0,
    "gender": 2,  # Male
    "height": 170.0,  # cm
    "weight": 80.0,  # kg
    "ap_hi": 130,  # Systolic BP
    "ap_lo": 85,  # Diastolic BP
    "cholesterol": 2,  # Above normal
    "gluc": 1,  # Normal
    "smoke": 0,  # No
    "alco": 0,  # No
    "active": 1  # Yes
}

# Expected service IDs
EXPECTED_SERVICES = {
    "cardiovascular_disease": {
        "service_id": "cardiovascular_disease",
        "port": 8000,
        "tags": ["cardiovascular", "disease", "classification", "health"],
        "test_data": CVD_SAMPLE_DATA
    }
}

# Registry configuration
REGISTRY_URL = "http://localhost:9000"
REGISTRY_HEALTH = f"{REGISTRY_URL}/health"
REGISTRY_SERVICES = f"{REGISTRY_URL}/api/v1/services"
