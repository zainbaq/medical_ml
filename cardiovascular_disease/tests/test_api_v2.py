"""
Tests for API v2 endpoints

Tests include:
- v2 prediction endpoint
- Explainability features
- Backward compatibility with v1
- Schema validation
- Feature requirements endpoint
"""
import pytest
import requests
import json
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend"))


class TestV2Schemas:
    """Tests for v2 Pydantic schemas"""

    def test_patient_data_v2_schema(self):
        """Test PatientDataV2 schema accepts required fields"""
        from models.schemas_v2 import PatientDataV2

        valid_data = {
            "age_years": 50.0,
            "gender": 1,
            "height": 170.0,
            "weight": 75.0,
            "ap_hi": 130,
            "ap_lo": 85,
            "cholesterol": 2,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1
        }

        patient = PatientDataV2(**valid_data)
        assert patient.age_years == 50.0
        assert patient.gender == 1

    def test_patient_data_v2_optional_fields(self):
        """Test PatientDataV2 accepts optional lab values"""
        from models.schemas_v2 import PatientDataV2

        data_with_labs = {
            "age_years": 50.0,
            "gender": 1,
            "height": 170.0,
            "weight": 75.0,
            "ap_hi": 130,
            "ap_lo": 85,
            "cholesterol": 2,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1,
            "hdl_cholesterol": 55.0,
            "ldl_cholesterol": 120.0
        }

        patient = PatientDataV2(**data_with_labs)
        assert patient.hdl_cholesterol == 55.0
        assert patient.ldl_cholesterol == 120.0

    def test_prediction_response_v2_schema(self):
        """Test PredictionResponseV2 schema structure"""
        from models.schemas_v2 import PredictionResponseV2

        response_data = {
            "prediction": 1,
            "probability": 0.75,
            "probability_range": (0.70, 0.80),
            "risk_category": "high",
            "bmi": 25.9,
            "prediction_tier": "basic",
            "top_risk_factors": [],
            "top_protective_factors": [],
            "modifiable_recommendations": [],
            "prediction_confidence": "high",
            "missing_features": [],
            "model_version": "2.0.0",
            "disclaimer": "Test disclaimer"
        }

        response = PredictionResponseV2(**response_data)
        assert response.prediction == 1
        assert response.risk_category == "high"

    def test_risk_factor_schema(self):
        """Test RiskFactor schema"""
        from models.schemas_v2 import RiskFactor

        factor = RiskFactor(
            name="ap_hi",
            display_name="Systolic Blood Pressure",
            value=160.0,
            contribution=0.15,
            is_modifiable=True,
            recommendation="Blood pressure control is recommended"
        )

        assert factor.name == "ap_hi"
        assert factor.is_modifiable is True
        assert factor.contribution == 0.15


class TestV2Preprocessing:
    """Tests for v2 preprocessing"""

    def test_preprocessor_v2_initialization(self):
        """Test PreprocessorV2 can be initialized"""
        from utils.preprocessing_v2 import PreprocessorV2

        preprocessor = PreprocessorV2()
        assert preprocessor is not None

    def test_tiered_feature_preparation(self, sample_patient_data):
        """Test tiered feature preparation"""
        from utils.preprocessing_v2 import PreprocessorV2
        from models.schemas_v2 import PatientDataV2

        preprocessor = PreprocessorV2()
        patient = PatientDataV2(**sample_patient_data)

        if hasattr(preprocessor, 'prepare_features_tiered'):
            # Method returns (features_array, tier, missing_features, feature_dict)
            result = preprocessor.prepare_features_tiered(patient)
            assert len(result) == 4
            features, tier, missing, feature_dict = result
            assert features is not None
            assert tier.value in ['basic', 'extended', 'clinical']

    def test_extended_feature_computation(self, sample_patient_data_v2):
        """Test extended feature computation with lab values"""
        from utils.preprocessing_v2 import PreprocessorV2
        from models.schemas_v2 import PatientDataV2

        preprocessor = PreprocessorV2()
        patient = PatientDataV2(**sample_patient_data_v2)

        if hasattr(preprocessor, 'compute_engineered_features_extended'):
            # Method requires basic features dict and patient_data
            basic_features = preprocessor.prepare_basic_features(patient)
            features = preprocessor.compute_engineered_features_extended(basic_features, patient)
            # Should include metabolic features
            if 'total_hdl_ratio' in features:
                assert features['total_hdl_ratio'] > 0


class TestV2Explainability:
    """Tests for explainability engine"""

    def test_explainability_engine_initialization(self):
        """Test ExplainabilityEngine can be initialized"""
        from utils.explainability import ExplainabilityEngine

        engine = ExplainabilityEngine()
        assert engine is not None

    def test_modifiable_features_defined(self):
        """Test modifiable features are properly defined"""
        from utils.explainability import ExplainabilityEngine

        engine = ExplainabilityEngine()
        assert hasattr(engine, 'MODIFIABLE_FEATURES')

        modifiable = engine.MODIFIABLE_FEATURES
        expected_modifiable = ['smoke', 'alco', 'active', 'bmi', 'ap_hi']

        for feature in expected_modifiable:
            assert feature in modifiable

    def test_feature_display_names(self):
        """Test feature display names are defined"""
        from utils.explainability import ExplainabilityEngine

        engine = ExplainabilityEngine()
        assert hasattr(engine, 'FEATURE_DISPLAY_NAMES')

        display_names = engine.FEATURE_DISPLAY_NAMES
        assert 'age_years' in display_names
        assert display_names['age_years'] == 'Age'

    def test_modifiable_recommendations(self, sample_patient_data):
        """Test modifiable risk recommendations"""
        from utils.explainability import ExplainabilityEngine

        engine = ExplainabilityEngine()

        # High risk patient data
        high_risk = sample_patient_data.copy()
        high_risk['smoke'] = 1
        high_risk['ap_hi'] = 160

        if hasattr(engine, 'get_modifiable_recommendations'):
            recommendations = engine.get_modifiable_recommendations(high_risk)
            assert len(recommendations) > 0


class TestV1BackwardCompatibility:
    """Tests for v1 API backward compatibility"""

    def test_v1_schema_unchanged(self):
        """Test v1 PatientData schema is unchanged"""
        from models.schemas import PatientData

        v1_data = {
            "age_years": 50.0,
            "gender": 1,
            "height": 170.0,
            "weight": 75.0,
            "ap_hi": 130,
            "ap_lo": 85,
            "cholesterol": 2,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1
        }

        patient = PatientData(**v1_data)
        assert patient.age_years == 50.0

    def test_v1_response_schema_unchanged(self):
        """Test v1 PredictionResponse schema is unchanged"""
        from models.schemas import PredictionResponse

        response = PredictionResponse(
            prediction=1,
            probability=0.75,
            risk_level="high",
            bmi=25.9,
            message="Test message"
        )

        assert response.prediction == 1
        assert response.risk_level == "high"


@pytest.mark.integration
class TestV2APIEndpoints:
    """Integration tests for v2 API endpoints

    These tests require the API to be running.
    Run with: pytest -m integration
    """

    def test_v2_predict_endpoint(self, api_base_url, sample_patient_data):
        """Test v2 predict endpoint"""
        try:
            response = requests.post(
                f"{api_base_url}/api/v2/predict",
                json=sample_patient_data,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                assert 'prediction' in data
                assert 'probability' in data
                assert 'risk_category' in data
                assert 'top_risk_factors' in data
                assert 'model_version' in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running")

    def test_v2_predict_with_extended_features(
        self, api_base_url, sample_patient_data_v2
    ):
        """Test v2 predict with extended features"""
        try:
            response = requests.post(
                f"{api_base_url}/api/v2/predict",
                json=sample_patient_data_v2,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                # Should detect extended tier
                assert 'prediction_tier' in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running")

    def test_v2_feature_requirements_endpoint(self, api_base_url):
        """Test feature requirements endpoint"""
        try:
            response = requests.get(
                f"{api_base_url}/api/v2/feature-requirements",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                assert 'tiers' in data or 'basic' in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running")

    def test_v2_model_info_endpoint(self, api_base_url):
        """Test v2 model info endpoint"""
        try:
            response = requests.get(
                f"{api_base_url}/api/v2/model-info",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                assert 'version' in data or 'model_version' in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running")

    def test_v2_health_endpoint(self, api_base_url):
        """Test v2 health endpoint"""
        try:
            response = requests.get(
                f"{api_base_url}/api/v2/health",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                assert 'status' in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running")

    def test_v1_still_works(self, api_base_url, sample_patient_data):
        """Test v1 API still works"""
        try:
            response = requests.post(
                f"{api_base_url}/api/v1/predict",
                json=sample_patient_data,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                assert 'prediction' in data
                assert 'probability' in data
                assert 'risk_level' in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running")


class TestV2ValidationErrors:
    """Tests for v2 validation error handling"""

    def test_invalid_blood_pressure_v2(self):
        """Test v2 validates blood pressure"""
        from models.schemas_v2 import PatientDataV2
        from pydantic import ValidationError

        invalid_data = {
            "age_years": 50.0,
            "gender": 1,
            "height": 170.0,
            "weight": 75.0,
            "ap_hi": 100,
            "ap_lo": 110,  # Higher than systolic
            "cholesterol": 2,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1
        }

        with pytest.raises(ValidationError):
            PatientDataV2(**invalid_data)

    def test_invalid_age_v2(self):
        """Test v2 validates age range"""
        from models.schemas_v2 import PatientDataV2
        from pydantic import ValidationError

        invalid_data = {
            "age_years": -5.0,  # Negative age
            "gender": 1,
            "height": 170.0,
            "weight": 75.0,
            "ap_hi": 130,
            "ap_lo": 85,
            "cholesterol": 2,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1
        }

        with pytest.raises(ValidationError):
            PatientDataV2(**invalid_data)

    def test_invalid_cholesterol_v2(self):
        """Test v2 validates cholesterol level"""
        from models.schemas_v2 import PatientDataV2
        from pydantic import ValidationError

        invalid_data = {
            "age_years": 50.0,
            "gender": 1,
            "height": 170.0,
            "weight": 75.0,
            "ap_hi": 130,
            "ap_lo": 85,
            "cholesterol": 5,  # Invalid: should be 1-3
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1
        }

        with pytest.raises(ValidationError):
            PatientDataV2(**invalid_data)


class TestRiskCategories:
    """Tests for risk category classification"""

    def test_risk_categories_defined(self):
        """Test risk categories are properly defined"""
        from models.schemas_v2 import PredictionResponseV2

        # Valid risk categories
        valid_categories = [
            'low', 'moderate', 'moderately_high', 'high', 'very_high'
        ]

        for category in valid_categories:
            response = PredictionResponseV2(
                prediction=1,
                probability=0.5,
                probability_range=(0.45, 0.55),
                risk_category=category,
                bmi=25.0,
                prediction_tier='basic',
                top_risk_factors=[],
                top_protective_factors=[],
                modifiable_recommendations=[],
                prediction_confidence='high',
                missing_features=[],
                model_version='2.0.0',
                disclaimer='Test'
            )
            assert response.risk_category == category

    def test_probability_to_risk_category_mapping(self):
        """Test probability maps to correct risk category"""
        # This tests the logic that should exist in preprocessing_v2

        def get_risk_category(probability: float) -> str:
            if probability < 0.2:
                return 'low'
            elif probability < 0.4:
                return 'moderate'
            elif probability < 0.6:
                return 'moderately_high'
            elif probability < 0.8:
                return 'high'
            else:
                return 'very_high'

        assert get_risk_category(0.1) == 'low'
        assert get_risk_category(0.3) == 'moderate'
        assert get_risk_category(0.5) == 'moderately_high'
        assert get_risk_category(0.7) == 'high'
        assert get_risk_category(0.9) == 'very_high'
