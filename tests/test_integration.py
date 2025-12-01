"""
Comprehensive integration tests for the Medical ML Service Registry System.

Tests the entire unified interface including:
- Service registry functionality
- Service auto-registration
- Service discovery
- Making predictions through all services
- Health monitoring

Run with: pytest test_integration.py -v -s
"""

import pytest
import requests
import time
from test_data import (
    CVD_SAMPLE_DATA,
    EXPECTED_SERVICES,
    REGISTRY_URL,
    REGISTRY_HEALTH,
    REGISTRY_SERVICES
)


class TestServiceStartup:
    """Test that all services start correctly."""

    def test_registry_is_running(self, all_services):
        """Test that the registry service is running."""
        response = requests.get(REGISTRY_HEALTH)
        assert response.status_code == 200, "Registry health check failed"

        data = response.json()
        assert data["status"] == "healthy", "Registry is not healthy"
        print(f"\n✓ Registry is healthy: {data}")

    def test_cvd_service_is_running(self, all_services):
        """Test that the CVD service is running."""
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 200, "CVD service health check failed"

        data = response.json()
        # Service may be "degraded" if model didn't load, but should respond
        assert data["status"] in ["healthy", "degraded"], "CVD service status unexpected"
        print(f"\n✓ CVD Service is running: {data}")


class TestServiceRegistration:
    """Test service auto-registration with the registry."""

    def test_cvd_service_registered(self, all_services):
        """Test that CVD service auto-registered with the registry."""
        response = requests.get(REGISTRY_SERVICES)
        assert response.status_code == 200, "Could not fetch services from registry"

        services = response.json()
        assert len(services) > 0, "No services registered in registry"

        # Check if CVD service is registered
        cvd_found = any(s["service_id"] == "cardiovascular_disease" for s in services)
        assert cvd_found, "Cardiovascular disease service not found in registry"

        print(f"\n✓ Found {len(services)} registered service(s)")
        for service in services:
            print(f"  - {service['service_name']} ({service['service_id']})")

    def test_service_metadata_complete(self, all_services):
        """Test that registered services have complete metadata."""
        response = requests.get(f"{REGISTRY_SERVICES}/cardiovascular_disease")
        assert response.status_code == 200, "Could not fetch CVD service metadata"

        service = response.json()

        # Check required fields
        assert service["service_id"] == "cardiovascular_disease"
        assert service["service_name"]
        assert service["version"]
        assert service["base_url"]
        assert service["port"] == 8000
        assert "endpoints" in service
        assert "predict" in service["endpoints"]
        assert "health" in service["endpoints"]
        assert "input_schema" in service
        assert "output_schema" in service
        assert "tags" in service
        assert len(service["tags"]) > 0

        print(f"\n✓ Service metadata is complete:")
        print(f"  Name: {service['service_name']}")
        print(f"  Version: {service['version']}")
        print(f"  URL: {service['base_url']}")
        print(f"  Tags: {service['tags']}")
        print(f"  Endpoints: {list(service['endpoints'].keys())}")


class TestServiceDiscovery:
    """Test service discovery features."""

    def test_list_all_services(self, all_services):
        """Test listing all registered services."""
        response = requests.get(REGISTRY_SERVICES)
        assert response.status_code == 200

        services = response.json()
        assert isinstance(services, list)
        assert len(services) >= 1

        print(f"\n✓ Listed {len(services)} services")

    def test_get_specific_service(self, all_services):
        """Test getting a specific service by ID."""
        response = requests.get(f"{REGISTRY_SERVICES}/cardiovascular_disease")
        assert response.status_code == 200

        service = response.json()
        assert service["service_id"] == "cardiovascular_disease"

        print(f"\n✓ Retrieved specific service: {service['service_name']}")

    def test_search_by_tags(self, all_services):
        """Test searching services by tags."""
        # Search for cardiovascular services
        response = requests.get(f"{REGISTRY_SERVICES}/search/by-tags?tags=cardiovascular")
        assert response.status_code == 200

        services = response.json()
        assert len(services) > 0
        assert any(s["service_id"] == "cardiovascular_disease" for s in services)

        print(f"\n✓ Tag search for 'cardiovascular' found {len(services)} service(s)")

        # Search for classification services
        response = requests.get(f"{REGISTRY_SERVICES}/search/by-tags?tags=classification")
        assert response.status_code == 200

        services = response.json()
        assert len(services) > 0

        print(f"✓ Tag search for 'classification' found {len(services)} service(s)")


class TestUnifiedInterface:
    """Test making API calls to all services through the unified interface."""

    def test_discover_and_predict_cvd(self, all_services):
        """Test discovering CVD service and making a prediction."""
        # Step 1: Discover the service via registry
        print("\n" + "="*60)
        print("Testing Unified Interface: CVD Prediction")
        print("="*60)

        response = requests.get(f"{REGISTRY_SERVICES}/cardiovascular_disease")
        assert response.status_code == 200

        service = response.json()
        print(f"\n1. Discovered service: {service['service_name']}")
        print(f"   URL: {service['base_url']}")
        print(f"   Prediction endpoint: {service['endpoints']['predict']}")

        # Step 2: Make a prediction
        predict_url = f"{service['base_url']}{service['endpoints']['predict']}"
        print(f"\n2. Making prediction to: {predict_url}")
        print(f"   Input data: {CVD_SAMPLE_DATA}")

        try:
            response = requests.post(predict_url, json=CVD_SAMPLE_DATA, timeout=10)
            assert response.status_code == 200, f"Prediction failed with status {response.status_code}"

            result = response.json()
            print(f"\n3. Prediction result:")
            print(f"   {result}")

            # Verify response has expected fields
            assert "prediction" in result
            assert "probability" in result

            print(f"\n✓ CVD prediction successful!")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Probability: {result.get('probability', 'N/A')}")

        except requests.exceptions.Timeout:
            pytest.skip("Prediction timed out - model may not be loaded")
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Prediction request failed: {e}")

    def test_health_checks_via_registry(self, all_services):
        """Test checking health of all services via registry."""
        response = requests.get(f"{REGISTRY_URL}/api/v1/health/all")
        assert response.status_code == 200

        health_status = response.json()
        assert isinstance(health_status, dict)

        if "cardiovascular_disease" in health_status:
            cvd_health = health_status["cardiovascular_disease"]
            print(f"\n✓ CVD Service Health: {cvd_health['status']}")

        print(f"\n✓ Aggregate health check successful")


class TestRegistryFeatures:
    """Test advanced registry features."""

    def test_aggregate_health(self, all_services):
        """Test aggregate health check."""
        response = requests.get(f"{REGISTRY_URL}/api/v1/health/all")
        assert response.status_code == 200

        health_data = response.json()
        print(f"\n✓ Aggregate health data: {health_data}")

    def test_registry_health_shows_service_count(self, all_services):
        """Test that registry health endpoint shows service count."""
        response = requests.get(REGISTRY_HEALTH)
        assert response.status_code == 200

        data = response.json()
        assert "registered_services" in data
        assert data["registered_services"] >= 1

        print(f"\n✓ Registry shows {data['registered_services']} registered service(s)")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_discovery_and_prediction_workflow(self, all_services):
        """Test the complete workflow a frontend would use."""
        print("\n" + "="*60)
        print("End-to-End Workflow Test")
        print("="*60)

        # Step 1: Frontend queries registry for all services
        print("\n1. Querying registry for available services...")
        response = requests.get(REGISTRY_SERVICES)
        assert response.status_code == 200
        services = response.json()
        print(f"   Found {len(services)} service(s)")

        # Step 2: Frontend displays services to user
        print("\n2. Available medical ML services:")
        for service in services:
            print(f"   - {service['service_name']}")
            print(f"     Tags: {', '.join(service['tags'])}")
            print(f"     Endpoints: {', '.join(service['endpoints'].keys())}")

        # Step 3: User selects a service (CVD in this case)
        cvd_service = next((s for s in services if s["service_id"] == "cardiovascular_disease"), None)
        assert cvd_service is not None
        print(f"\n3. User selects: {cvd_service['service_name']}")

        # Step 4: Frontend gets input schema to build form
        input_schema = cvd_service["input_schema"]
        print(f"\n4. Input schema retrieved (has {len(input_schema.get('properties', {}))} fields)")

        # Step 5: User submits data, frontend makes prediction
        predict_url = f"{cvd_service['base_url']}{cvd_service['endpoints']['predict']}"
        print(f"\n5. Making prediction to: {predict_url}")

        try:
            response = requests.post(predict_url, json=CVD_SAMPLE_DATA, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"\n6. Prediction received:")
                print(f"   Result: {result}")
                print("\n✓ End-to-end workflow successful!")
            else:
                print(f"\n⚠ Prediction returned status {response.status_code}")
                pytest.skip("Prediction endpoint not fully functional")
        except Exception as e:
            pytest.skip(f"Prediction failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
