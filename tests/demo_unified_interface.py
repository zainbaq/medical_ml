#!/usr/bin/env python3
"""
Demo Script for Medical ML Unified Service Registry

This script demonstrates how a frontend application would interact
with the unified service registry to discover and use medical ML services.

Usage:
    python demo_unified_interface.py
"""

import requests
import json
import time
from typing import List, Dict
from test_data import CVD_SAMPLE_DATA, REGISTRY_URL


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_section(text: str):
    """Print a formatted section header."""
    print(f"\n{text}")
    print("-" * len(text))


def discover_services() -> List[Dict]:
    """Discover all available medical ML services."""
    print_header("STEP 1: DISCOVER AVAILABLE SERVICES")

    print("\nQuerying service registry at:", REGISTRY_URL)
    response = requests.get(f"{REGISTRY_URL}/api/v1/services")

    if response.status_code != 200:
        print(f"❌ Error: Could not connect to registry (status {response.status_code})")
        return []

    services = response.json()
    print(f"\n✅ Found {len(services)} registered service(s)\n")

    return services


def display_service_catalog(services: List[Dict]):
    """Display available services in a user-friendly format."""
    print_header("STEP 2: MEDICAL ML SERVICE CATALOG")

    for i, service in enumerate(services, 1):
        print(f"\n🏥 Service {i}: {service['service_name']}")
        print(f"   ID: {service['service_id']}")
        print(f"   Version: {service['version']}")
        print(f"   Description: {service.get('description', 'N/A')}")
        print(f"   URL: {service['base_url']}")
        print(f"   Tags: {', '.join(service['tags'])}")
        print(f"   Endpoints:")
        for endpoint_name, endpoint_path in service['endpoints'].items():
            print(f"     - {endpoint_name}: {endpoint_path}")


def get_service_schema(service: Dict):
    """Display service input/output schemas."""
    print_header(f"STEP 3: EXAMINE SERVICE SCHEMA - {service['service_name']}")

    print("\n📝 Input Schema:")
    input_schema = service.get('input_schema', {})
    properties = input_schema.get('properties', {})

    if properties:
        print(f"\n   Required fields ({len(properties)}):")
        for field_name, field_info in list(properties.items())[:10]:  # Show first 10
            field_type = field_info.get('type', 'unknown')
            description = field_info.get('description', '')
            print(f"     • {field_name} ({field_type})")
            if description:
                print(f"       {description}")
        if len(properties) > 10:
            print(f"     ... and {len(properties) - 10} more fields")
    else:
        print("   No schema information available")

    print("\n📤 Output Schema:")
    output_schema = service.get('output_schema', {})
    output_props = output_schema.get('properties', {})

    if output_props:
        for field_name in output_props.keys():
            print(f"     • {field_name}")
    else:
        print("   No schema information available")


def make_prediction(service: Dict, sample_data: Dict):
    """Make a prediction using a service."""
    print_header(f"STEP 4: MAKE PREDICTION - {service['service_name']}")

    predict_url = f"{service['base_url']}{service['endpoints']['predict']}"

    print(f"\n🎯 Making prediction to: {predict_url}")
    print(f"\n📊 Sample Input Data:")
    print(json.dumps(sample_data, indent=2))

    try:
        response = requests.post(predict_url, json=sample_data, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Prediction Successful!")
            print(f"\n📈 Results:")
            print(json.dumps(result, indent=2))

            # Extract key results
            if 'prediction' in result:
                print(f"\n🔍 Key Findings:")
                print(f"   Prediction: {result['prediction']}")
                if 'probability' in result:
                    print(f"   Confidence: {result['probability']:.2%}")
                if 'risk_level' in result:
                    print(f"   Risk Level: {result['risk_level']}")

        else:
            print(f"\n❌ Prediction failed with status {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.Timeout:
        print("\n⏱️  Prediction timed out (model may not be loaded)")
    except Exception as e:
        print(f"\n❌ Error making prediction: {e}")


def check_service_health(services: List[Dict]):
    """Check health of all services."""
    print_header("STEP 5: HEALTH CHECK ALL SERVICES")

    print("\n🏥 Checking health of all registered services...\n")

    for service in services:
        health_url = f"{service['base_url']}{service['endpoints'].get('health', '/health')}"

        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get('status', 'unknown')
                model_loaded = health_data.get('model_loaded', False)

                status_emoji = "✅" if status == "healthy" else "⚠️"
                model_emoji = "✅" if model_loaded else "❌"

                print(f"{status_emoji} {service['service_name']}")
                print(f"   Status: {status}")
                print(f"   Model Loaded: {model_emoji} {model_loaded}")
            else:
                print(f"❌ {service['service_name']} - HTTP {response.status_code}")

        except Exception as e:
            print(f"❌ {service['service_name']} - Error: {e}")


def demonstrate_tag_search():
    """Demonstrate searching services by tags."""
    print_header("STEP 6: SEARCH SERVICES BY TAGS")

    # Search for cardiovascular services
    print("\n🔍 Searching for 'cardiovascular' services...")
    response = requests.get(f"{REGISTRY_URL}/api/v1/services/search/by-tags?tags=cardiovascular")

    if response.status_code == 200:
        services = response.json()
        print(f"   Found {len(services)} service(s):")
        for service in services:
            print(f"     • {service['service_name']}")

    # Search for classification services
    print("\n🔍 Searching for 'classification' services...")
    response = requests.get(f"{REGISTRY_URL}/api/v1/services/search/by-tags?tags=classification")

    if response.status_code == 200:
        services = response.json()
        print(f"   Found {len(services)} service(s):")
        for service in services:
            print(f"     • {service['service_name']}")


def main():
    """Run the demo."""
    print_header("🏥 MEDICAL ML UNIFIED SERVICE REGISTRY DEMO 🏥")

    print("\nThis demo shows how a frontend application can:")
    print("  1. Discover available medical ML services")
    print("  2. Get service metadata and schemas")
    print("  3. Make predictions to services")
    print("  4. Monitor service health")
    print("  5. Search services by tags")

    time.sleep(2)

    # Step 1: Discover services
    services = discover_services()

    if not services:
        print("\n❌ No services found. Please ensure:")
        print("   1. Registry is running on port 9000")
        print("   2. Medical ML services are started")
        return

    # Step 2: Display catalog
    display_service_catalog(services)
    time.sleep(1)

    # Step 3 & 4: Focus on CVD service
    cvd_service = next((s for s in services if s['service_id'] == 'cardiovascular_disease'), None)

    if cvd_service:
        get_service_schema(cvd_service)
        time.sleep(1)
        make_prediction(cvd_service, CVD_SAMPLE_DATA)
    else:
        print("\n⚠️  CVD service not found, skipping prediction demo")

    time.sleep(1)

    # Step 5: Health checks
    check_service_health(services)
    time.sleep(1)

    # Step 6: Tag search
    demonstrate_tag_search()

    # Final summary
    print_header("✅ DEMO COMPLETE")

    print("\n📚 Key Takeaways:")
    print("   ✓ Services auto-register with the registry on startup")
    print("   ✓ Frontend can dynamically discover services")
    print("   ✓ Service schemas enable dynamic form generation")
    print("   ✓ Predictions are made directly to services (not through registry)")
    print("   ✓ Health monitoring is centralized")
    print("   ✓ Services can be searched by tags")

    print("\n🚀 The unified interface makes it easy to:")
    print("   • Add new medical ML services")
    print("   • Build dynamic frontends")
    print("   • Monitor system health")
    print("   • Scale services independently")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
