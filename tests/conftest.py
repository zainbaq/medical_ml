"""Pytest configuration and fixtures for medical ML tests."""

import pytest
import subprocess
import time
import requests
import signal
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def wait_for_service(url, timeout=30, service_name="Service"):
    """Wait for a service to become available."""
    print(f"Waiting for {service_name} at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✓ {service_name} is ready!")
                return True
        except requests.exceptions.RequestException:
            time.sleep(0.5)
    raise TimeoutError(f"{service_name} at {url} did not start within {timeout}s")


@pytest.fixture(scope="session")
def registry_service():
    """Start the registry service for the test session."""
    print("\n" + "="*60)
    print("Starting Registry Service")
    print("="*60)

    # Start registry
    registry_dir = PROJECT_ROOT / "registry" / "backend"
    process = subprocess.Popen(
        ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"],
        cwd=str(registry_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid if os.name != 'nt' else None
    )

    try:
        # Wait for registry to be ready
        wait_for_service("http://localhost:9000/health", service_name="Registry")

        yield process

    finally:
        # Cleanup
        print("\nStopping Registry Service...")
        if os.name != 'nt':
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
        print("✓ Registry Service stopped")


@pytest.fixture(scope="session")
def cvd_service(registry_service):
    """Start the cardiovascular disease service."""
    print("\n" + "="*60)
    print("Starting Cardiovascular Disease Service")
    print("="*60)

    # Start CVD service
    cvd_dir = PROJECT_ROOT / "cardiovascular_disease"
    process = subprocess.Popen(
        ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(cvd_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid if os.name != 'nt' else None
    )

    try:
        # Wait for service to be ready
        wait_for_service("http://localhost:8000/health", service_name="CVD Service")

        # Give it a moment to register
        time.sleep(2)

        yield process

    finally:
        # Cleanup
        print("\nStopping CVD Service...")
        if os.name != 'nt':
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
        print("✓ CVD Service stopped")


@pytest.fixture(scope="session")
def all_services(registry_service, cvd_service):
    """Ensure all services are running."""
    print("\n" + "="*60)
    print("All Services Started Successfully")
    print("="*60)
    yield {
        "registry": registry_service,
        "cvd": cvd_service
    }
    print("\n" + "="*60)
    print("Test Session Complete - Cleaning Up")
    print("="*60)
