"""
Generate test data for cardiovascular disease prediction
Creates realistic patient profiles with various risk levels
"""
import json
import random
from typing import List, Dict, Any


def generate_low_risk_patient() -> Dict[str, Any]:
    """Generate a low-risk patient profile"""
    return {
        "age_years": float(random.randint(18, 35)),
        "gender": random.randint(1, 2),
        "height": float(random.randint(160, 185)),
        "weight": float(random.randint(50, 75)),
        "ap_hi": random.randint(90, 120),
        "ap_lo": random.randint(60, 80),
        "cholesterol": 1,  # Normal
        "gluc": 1,  # Normal
        "smoke": 0,
        "alco": random.randint(0, 1),
        "active": 1
    }


def generate_medium_risk_patient() -> Dict[str, Any]:
    """Generate a medium-risk patient profile"""
    age = random.randint(40, 55)
    height = random.randint(160, 185)
    # Calculate weight for BMI 25-30 (overweight)
    height_m = height / 100
    bmi = random.uniform(25, 30)
    weight = bmi * (height_m ** 2)

    return {
        "age_years": float(age),
        "gender": random.randint(1, 2),
        "height": float(height),
        "weight": round(weight, 1),
        "ap_hi": random.randint(120, 140),
        "ap_lo": random.randint(80, 90),
        "cholesterol": random.randint(1, 2),
        "gluc": random.randint(1, 2),
        "smoke": random.randint(0, 1),
        "alco": random.randint(0, 1),
        "active": random.randint(0, 1)
    }


def generate_high_risk_patient() -> Dict[str, Any]:
    """Generate a high-risk patient profile"""
    age = random.randint(55, 75)
    height = random.randint(160, 185)
    # Calculate weight for BMI > 30 (obese)
    height_m = height / 100
    bmi = random.uniform(30, 38)
    weight = bmi * (height_m ** 2)

    return {
        "age_years": float(age),
        "gender": random.randint(1, 2),
        "height": float(height),
        "weight": round(weight, 1),
        "ap_hi": random.randint(140, 180),
        "ap_lo": random.randint(90, 110),
        "cholesterol": random.randint(2, 3),
        "gluc": random.randint(2, 3),
        "smoke": random.randint(0, 1),
        "alco": random.randint(0, 1),
        "active": 0
    }


def generate_test_cases() -> List[Dict[str, Any]]:
    """Generate a comprehensive set of test cases"""
    test_cases = []

    # Specific medical scenarios
    test_cases.extend([
        {
            "name": "Healthy Young Adult",
            "description": "25-year-old with optimal health metrics",
            "data": {
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
        },
        {
            "name": "Elderly with Hypertension",
            "description": "70-year-old with high blood pressure",
            "data": {
                "age_years": 70.0,
                "gender": 2,
                "height": 170.0,
                "weight": 75.0,
                "ap_hi": 170,
                "ap_lo": 105,
                "cholesterol": 2,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1
            }
        },
        {
            "name": "Obese with Multiple Risk Factors",
            "description": "55-year-old obese smoker with high cholesterol",
            "data": {
                "age_years": 55.0,
                "gender": 2,
                "height": 175.0,
                "weight": 110.0,
                "ap_hi": 150,
                "ap_lo": 95,
                "cholesterol": 3,
                "gluc": 2,
                "smoke": 1,
                "alco": 1,
                "active": 0
            }
        },
        {
            "name": "Pre-diabetic with Hypertension",
            "description": "60-year-old with elevated glucose and BP",
            "data": {
                "age_years": 60.0,
                "gender": 1,
                "height": 160.0,
                "weight": 75.0,
                "ap_hi": 145,
                "ap_lo": 92,
                "cholesterol": 2,
                "gluc": 3,
                "smoke": 0,
                "alco": 0,
                "active": 0
            }
        },
        {
            "name": "Active Middle-aged Adult",
            "description": "45-year-old physically active with good metrics",
            "data": {
                "age_years": 45.0,
                "gender": 2,
                "height": 180.0,
                "weight": 80.0,
                "ap_hi": 120,
                "ap_lo": 75,
                "cholesterol": 1,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1
            }
        },
        {
            "name": "Smoker with Normal Weight",
            "description": "50-year-old smoker but otherwise healthy",
            "data": {
                "age_years": 50.0,
                "gender": 1,
                "height": 165.0,
                "weight": 65.0,
                "ap_hi": 125,
                "ap_lo": 80,
                "cholesterol": 1,
                "gluc": 1,
                "smoke": 1,
                "alco": 0,
                "active": 1
            }
        },
        {
            "name": "Sedentary Lifestyle",
            "description": "40-year-old inactive with borderline metrics",
            "data": {
                "age_years": 40.0,
                "gender": 2,
                "height": 175.0,
                "weight": 90.0,
                "ap_hi": 135,
                "ap_lo": 88,
                "cholesterol": 2,
                "gluc": 1,
                "smoke": 0,
                "alco": 1,
                "active": 0
            }
        },
        {
            "name": "High Risk Elderly",
            "description": "75-year-old with multiple severe risk factors",
            "data": {
                "age_years": 75.0,
                "gender": 2,
                "height": 170.0,
                "weight": 95.0,
                "ap_hi": 180,
                "ap_lo": 110,
                "cholesterol": 3,
                "gluc": 3,
                "smoke": 1,
                "alco": 1,
                "active": 0
            }
        }
    ])

    return test_cases


def save_test_data(filename: str = "test_patients.json"):
    """Generate and save test data to JSON file"""
    data = {
        "description": "Test patient data for cardiovascular disease prediction API",
        "risk_profiles": {
            "low_risk": [generate_low_risk_patient() for _ in range(10)],
            "medium_risk": [generate_medium_risk_patient() for _ in range(10)],
            "high_risk": [generate_high_risk_patient() for _ in range(10)]
        },
        "specific_cases": generate_test_cases()
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Test data saved to {filename}")
    print(f"Generated {len(data['risk_profiles']['low_risk'])} low-risk patients")
    print(f"Generated {len(data['risk_profiles']['medium_risk'])} medium-risk patients")
    print(f"Generated {len(data['risk_profiles']['high_risk'])} high-risk patients")
    print(f"Generated {len(data['specific_cases'])} specific test cases")

    return data


def print_sample_patients():
    """Print sample patients for each risk category"""
    print("\n" + "="*70)
    print("SAMPLE PATIENT PROFILES")
    print("="*70 + "\n")

    print("LOW RISK PATIENT:")
    print(json.dumps(generate_low_risk_patient(), indent=2))

    print("\nMEDIUM RISK PATIENT:")
    print(json.dumps(generate_medium_risk_patient(), indent=2))

    print("\nHIGH RISK PATIENT:")
    print(json.dumps(generate_high_risk_patient(), indent=2))

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print_sample_patients()
    save_test_data()
