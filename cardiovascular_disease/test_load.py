"""
Load testing script for Cardiovascular Disease Prediction API
Tests API performance under concurrent requests
"""
import requests
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import statistics


BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1"


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def generate_random_patient() -> Dict[str, Any]:
    """Generate random patient data for testing"""
    return {
        "age_years": float(random.randint(18, 80)),
        "gender": random.randint(1, 2),
        "height": float(random.randint(150, 195)),
        "weight": float(random.randint(50, 120)),
        "ap_hi": random.randint(90, 180),
        "ap_lo": random.randint(60, 110),
        "cholesterol": random.randint(1, 3),
        "gluc": random.randint(1, 3),
        "smoke": random.randint(0, 1),
        "alco": random.randint(0, 1),
        "active": random.randint(0, 1)
    }


def make_prediction_request(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make a single prediction request and measure response time"""
    start_time = time.time()

    try:
        response = requests.post(f"{API_URL}/predict", json=patient_data, timeout=10)
        elapsed_time = time.time() - start_time

        return {
            'success': response.status_code == 200,
            'status_code': response.status_code,
            'response_time': elapsed_time,
            'data': response.json() if response.status_code == 200 else None,
            'error': None
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'success': False,
            'status_code': None,
            'response_time': elapsed_time,
            'data': None,
            'error': str(e)
        }


def run_load_test(num_requests: int, num_workers: int) -> List[Dict[str, Any]]:
    """Run load test with specified number of requests and concurrent workers"""
    print(f"\n{Colors.BOLD}Running load test...{Colors.RESET}")
    print(f"Total requests: {num_requests}")
    print(f"Concurrent workers: {num_workers}")
    print(f"{Colors.BLUE}{'-'*70}{Colors.RESET}\n")

    # Generate test data
    test_patients = [generate_random_patient() for _ in range(num_requests)]

    results = []
    start_time = time.time()

    # Execute requests concurrently
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(make_prediction_request, patient): i
            for i, patient in enumerate(test_patients)
        }

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            # Progress indicator
            if completed % 10 == 0 or completed == num_requests:
                print(f"Progress: {completed}/{num_requests} requests completed", end='\r')

    total_time = time.time() - start_time
    print(f"\n\nTotal test time: {total_time:.2f} seconds")

    return results, total_time


def analyze_results(results: List[Dict[str, Any]], total_time: float):
    """Analyze and display test results"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'LOAD TEST RESULTS'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

    # Success rate
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    total = len(results)

    success_rate = (len(successful) / total) * 100 if total > 0 else 0

    print(f"{Colors.BOLD}Success Rate:{Colors.RESET}")
    print(f"  Total requests: {total}")
    print(f"  Successful: {len(successful)} ({success_rate:.1f}%)")
    print(f"  Failed: {len(failed)} ({100-success_rate:.1f}%)")

    # Response time statistics
    if successful:
        response_times = [r['response_time'] for r in successful]

        print(f"\n{Colors.BOLD}Response Time Statistics (successful requests):{Colors.RESET}")
        print(f"  Min: {min(response_times)*1000:.2f} ms")
        print(f"  Max: {max(response_times)*1000:.2f} ms")
        print(f"  Mean: {statistics.mean(response_times)*1000:.2f} ms")
        print(f"  Median: {statistics.median(response_times)*1000:.2f} ms")

        if len(response_times) > 1:
            print(f"  Std Dev: {statistics.stdev(response_times)*1000:.2f} ms")

        # Percentiles
        sorted_times = sorted(response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p99_index = int(len(sorted_times) * 0.99)

        print(f"  95th percentile: {sorted_times[p95_index]*1000:.2f} ms")
        print(f"  99th percentile: {sorted_times[p99_index]*1000:.2f} ms")

    # Throughput
    requests_per_second = total / total_time if total_time > 0 else 0

    print(f"\n{Colors.BOLD}Throughput:{Colors.RESET}")
    print(f"  Requests/second: {requests_per_second:.2f}")
    print(f"  Total time: {total_time:.2f} seconds")

    # Prediction distribution
    if successful:
        predictions = [r['data']['prediction'] for r in successful if r['data']]
        positive_predictions = sum(predictions)
        negative_predictions = len(predictions) - positive_predictions

        print(f"\n{Colors.BOLD}Prediction Distribution:{Colors.RESET}")
        print(f"  Positive (Disease): {positive_predictions} ({positive_predictions/len(predictions)*100:.1f}%)")
        print(f"  Negative (No Disease): {negative_predictions} ({negative_predictions/len(predictions)*100:.1f}%)")

        # Risk level distribution
        risk_levels = [r['data']['risk_level'] for r in successful if r['data']]
        risk_counts = {
            'low': risk_levels.count('low'),
            'medium': risk_levels.count('medium'),
            'high': risk_levels.count('high')
        }

        print(f"\n{Colors.BOLD}Risk Level Distribution:{Colors.RESET}")
        for level, count in risk_counts.items():
            percentage = (count / len(risk_levels) * 100) if risk_levels else 0
            print(f"  {level.capitalize()}: {count} ({percentage:.1f}%)")

    # Errors
    if failed:
        print(f"\n{Colors.BOLD}{Colors.RED}Errors:{Colors.RESET}")
        error_types = {}
        for result in failed:
            error = result['error'] or f"HTTP {result['status_code']}"
            error_types[error] = error_types.get(error, 0) + 1

        for error, count in error_types.items():
            print(f"  {error}: {count}")

    # Overall assessment
    print(f"\n{Colors.BOLD}Overall Assessment:{Colors.RESET}")
    if success_rate >= 99:
        print(f"  {Colors.GREEN}Excellent - API is highly reliable{Colors.RESET}")
    elif success_rate >= 95:
        print(f"  {Colors.GREEN}Good - API is performing well{Colors.RESET}")
    elif success_rate >= 90:
        print(f"  {Colors.YELLOW}Fair - Some reliability issues{Colors.RESET}")
    else:
        print(f"  {Colors.RED}Poor - Significant reliability issues{Colors.RESET}")

    if successful:
        avg_response = statistics.mean(response_times) * 1000
        if avg_response < 100:
            print(f"  {Colors.GREEN}Excellent response time (< 100ms){Colors.RESET}")
        elif avg_response < 500:
            print(f"  {Colors.GREEN}Good response time (< 500ms){Colors.RESET}")
        elif avg_response < 1000:
            print(f"  {Colors.YELLOW}Acceptable response time (< 1s){Colors.RESET}")
        else:
            print(f"  {Colors.RED}Slow response time (> 1s){Colors.RESET}")

    print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def test_health_before_load():
    """Check API health before running load test"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('model_loaded'):
                print(f"{Colors.GREEN}✓ API is healthy and model is loaded{Colors.RESET}")
                return True
            else:
                print(f"{Colors.RED}✗ API is running but model is not loaded{Colors.RESET}")
                return False
        else:
            print(f"{Colors.RED}✗ API health check failed{Colors.RESET}")
            return False
    except Exception as e:
        print(f"{Colors.RED}✗ Cannot connect to API: {str(e)}{Colors.RESET}")
        return False


def main():
    """Main function to run load tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'CARDIOVASCULAR DISEASE API - LOAD TEST'.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")
    print(f"Testing API at: {BASE_URL}\n")

    # Check API health first
    if not test_health_before_load():
        print("\nPlease ensure the API is running:")
        print("  cd cardiovascular_disease && ./start_api.sh")
        return

    # Run different load test scenarios
    scenarios = [
        {"name": "Light Load", "requests": 50, "workers": 5},
        {"name": "Moderate Load", "requests": 100, "workers": 10},
        {"name": "Heavy Load", "requests": 200, "workers": 20},
    ]

    print(f"\n{Colors.BOLD}Select load test scenario:{Colors.RESET}")
    for i, scenario in enumerate(scenarios, 1):
        print(f"  {i}. {scenario['name']}: {scenario['requests']} requests, {scenario['workers']} concurrent workers")
    print(f"  4. Custom")

    try:
        choice = input(f"\nEnter choice (1-4) [default: 1]: ").strip() or "1"
        choice_num = int(choice)

        if 1 <= choice_num <= 3:
            scenario = scenarios[choice_num - 1]
            num_requests = scenario['requests']
            num_workers = scenario['workers']
        elif choice_num == 4:
            num_requests = int(input("Number of requests: "))
            num_workers = int(input("Concurrent workers: "))
        else:
            print("Invalid choice, using Light Load")
            num_requests = 50
            num_workers = 5

    except (ValueError, KeyboardInterrupt):
        print("\nUsing default Light Load scenario")
        num_requests = 50
        num_workers = 5

    # Run load test
    results, total_time = run_load_test(num_requests, num_workers)

    # Analyze results
    analyze_results(results, total_time)


if __name__ == "__main__":
    main()
