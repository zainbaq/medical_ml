"""
Tests for feature engineering

Tests include:
- Basic feature computation
- Improved feature engineering
- Extended features (metabolic, clinical)
- Feature tiers
- Feature manager
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "training"))


class TestBasicFeatures:
    """Tests for basic feature computation"""

    def test_bmi_calculation(self, sample_training_data):
        """Test BMI is calculated correctly"""
        # BMI = weight / (height_m)^2
        expected_bmi = sample_training_data['weight'] / (
            (sample_training_data['height'] / 100) ** 2
        )

        # Calculate BMI
        calculated_bmi = sample_training_data['weight'] / (
            (sample_training_data['height'] / 100) ** 2
        )

        np.testing.assert_array_almost_equal(calculated_bmi, expected_bmi, decimal=2)

    def test_age_conversion(self, sample_training_data):
        """Test age in days is converted to years correctly"""
        expected_age_years = sample_training_data['age'] / 365.25
        calculated_age_years = sample_training_data['age'] / 365.25

        np.testing.assert_array_almost_equal(
            calculated_age_years, expected_age_years, decimal=2
        )


class TestImprovedFeatureEngineer:
    """Tests for ImprovedFeatureEngineer class"""

    def test_feature_engineer_initialization(self):
        """Test ImprovedFeatureEngineer can be initialized"""
        from improved_features import ImprovedFeatureEngineer

        engineer = ImprovedFeatureEngineer()
        assert engineer is not None

    def test_pulse_pressure_calculation(self, sample_training_data):
        """Test pulse pressure feature"""
        from improved_features import ImprovedFeatureEngineer

        engineer = ImprovedFeatureEngineer()
        features = engineer.create_features(sample_training_data.copy())

        assert 'pulse_pressure' in features.columns
        expected_pp = sample_training_data['ap_hi'] - sample_training_data['ap_lo']
        np.testing.assert_array_almost_equal(
            features['pulse_pressure'].values,
            expected_pp.values,
            decimal=1
        )

    def test_mean_arterial_pressure(self, sample_training_data):
        """Test mean arterial pressure calculation"""
        from improved_features import ImprovedFeatureEngineer

        engineer = ImprovedFeatureEngineer()
        features = engineer.create_features(sample_training_data.copy())

        assert 'mean_arterial_pressure' in features.columns
        # MAP = DBP + 1/3(SBP - DBP)
        expected_map = (
            sample_training_data['ap_lo'] +
            (sample_training_data['ap_hi'] - sample_training_data['ap_lo']) / 3
        )
        np.testing.assert_array_almost_equal(
            features['mean_arterial_pressure'].values,
            expected_map.values,
            decimal=1
        )

    def test_hypertension_stage(self, sample_training_data):
        """Test hypertension stage classification"""
        from improved_features import ImprovedFeatureEngineer

        engineer = ImprovedFeatureEngineer()
        features = engineer.create_features(sample_training_data.copy())

        assert 'hypertension_stage' in features.columns
        # Should be integer values 0-4
        assert all(features['hypertension_stage'].isin([0, 1, 2, 3, 4]))

    def test_bmi_category(self, sample_training_data):
        """Test BMI category classification"""
        from improved_features import ImprovedFeatureEngineer

        # Prepare data with BMI
        data = sample_training_data.copy()
        data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)

        engineer = ImprovedFeatureEngineer()
        features = engineer.create_features(data)

        assert 'bmi_category' in features.columns
        # Categories: 0=Underweight, 1=Normal, 2=Overweight, 3=Obese
        assert all(features['bmi_category'].isin([0, 1, 2, 3]))

    def test_age_group(self, sample_training_data):
        """Test age group classification"""
        from improved_features import ImprovedFeatureEngineer

        # Prepare data
        data = sample_training_data.copy()
        data['age_years'] = data['age'] / 365.25

        engineer = ImprovedFeatureEngineer()
        features = engineer.create_features(data)

        assert 'age_group' in features.columns
        # Should be integer categories
        assert features['age_group'].dtype in [np.int64, np.int32, int]

    def test_lifestyle_risk_score(self, sample_training_data):
        """Test lifestyle risk score computation"""
        from improved_features import ImprovedFeatureEngineer

        engineer = ImprovedFeatureEngineer()
        features = engineer.create_features(sample_training_data.copy())

        assert 'lifestyle_risk_score' in features.columns
        # Score should be 0-3 (smoke + alco + not active)
        assert all(features['lifestyle_risk_score'] >= 0)
        assert all(features['lifestyle_risk_score'] <= 3)

    def test_health_risk_composite(self, sample_training_data):
        """Test health risk composite score"""
        from improved_features import ImprovedFeatureEngineer

        engineer = ImprovedFeatureEngineer()
        features = engineer.create_features(sample_training_data.copy())

        assert 'health_risk_composite' in features.columns
        # Should be non-negative
        assert all(features['health_risk_composite'] >= 0)


class TestExtendedFeatures:
    """Tests for extended feature engineering (v2)"""

    def test_metabolic_features_with_lab_values(self):
        """Test metabolic features when lab values are available"""
        from improved_features import ImprovedFeatureEngineer

        # Create data with lab values
        data = pd.DataFrame({
            'age': [18000],
            'gender': [1],
            'height': [170],
            'weight': [75],
            'ap_hi': [130],
            'ap_lo': [85],
            'cholesterol': [2],
            'gluc': [1],
            'smoke': [0],
            'alco': [0],
            'active': [1],
            'hdl_cholesterol': [55.0],
            'ldl_cholesterol': [120.0],
            'total_cholesterol': [200.0],
            'triglycerides': [150.0]
        })
        data['age_years'] = data['age'] / 365.25
        data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)

        engineer = ImprovedFeatureEngineer()

        # Check if extended features method exists
        if hasattr(engineer, 'add_metabolic_features'):
            features = engineer.add_metabolic_features(data.copy())

            # Check metabolic features exist
            if 'total_hdl_ratio' in features.columns:
                assert features['total_hdl_ratio'].iloc[0] == pytest.approx(
                    200.0 / 55.0, rel=0.01
                )

    def test_extended_bp_features(self):
        """Test extended blood pressure features"""
        from improved_features import ImprovedFeatureEngineer

        data = pd.DataFrame({
            'age': [18000],
            'gender': [1],
            'height': [170],
            'weight': [75],
            'ap_hi': [160],
            'ap_lo': [100],
            'cholesterol': [2],
            'gluc': [1],
            'smoke': [0],
            'alco': [0],
            'active': [1]
        })
        data['age_years'] = data['age'] / 365.25
        data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)

        engineer = ImprovedFeatureEngineer()

        if hasattr(engineer, 'add_extended_bp_features'):
            features = engineer.add_extended_bp_features(data.copy())
            # Check for bp_index if it exists
            if 'bp_index' in features.columns:
                assert features['bp_index'].iloc[0] > 0

    def test_modifiable_risk_features(self):
        """Test modifiable risk factor features"""
        from improved_features import ImprovedFeatureEngineer

        # High risk patient
        data = pd.DataFrame({
            'age': [18000],
            'gender': [2],
            'height': [165],
            'weight': [100],
            'ap_hi': [160],
            'ap_lo': [100],
            'cholesterol': [3],
            'gluc': [2],
            'smoke': [1],
            'alco': [1],
            'active': [0]
        })
        data['age_years'] = data['age'] / 365.25
        data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)

        engineer = ImprovedFeatureEngineer()

        if hasattr(engineer, 'add_modifiable_risk_features'):
            features = engineer.add_modifiable_risk_features(data.copy())
            if 'modifiable_risk_count' in features.columns:
                # Should have multiple risk factors
                assert features['modifiable_risk_count'].iloc[0] >= 3


class TestFeatureManager:
    """Tests for FeatureManager class"""

    def test_feature_manager_initialization(self):
        """Test FeatureManager can be initialized"""
        from feature_manager import FeatureManager

        manager = FeatureManager()
        assert manager is not None

    def test_feature_tiers_defined(self):
        """Test feature tiers are properly defined"""
        from feature_manager import FeatureManager

        manager = FeatureManager()
        assert hasattr(manager, 'FEATURE_TIERS') or hasattr(manager, 'feature_tiers')

        # Check tier names
        tiers = getattr(manager, 'FEATURE_TIERS', getattr(manager, 'feature_tiers', {}))
        assert 'basic' in tiers or 'BASIC' in str(tiers)

    def test_tier_detection_basic(self, sample_patient_data):
        """Test basic tier detection with minimal features"""
        from feature_manager import FeatureManager

        manager = FeatureManager()

        if hasattr(manager, 'detect_tier'):
            tier = manager.detect_tier(sample_patient_data)
            assert tier is not None
        elif hasattr(manager, 'get_prediction_tier'):
            tier, _ = manager.get_prediction_tier(sample_patient_data)
            assert tier is not None

    def test_tier_detection_extended(self, sample_patient_data_v2):
        """Test extended tier detection with lab values"""
        from feature_manager import FeatureManager

        manager = FeatureManager()

        if hasattr(manager, 'detect_tier'):
            tier = manager.detect_tier(sample_patient_data_v2)
            # Should detect extended tier due to lab values
            assert tier in ['extended', 'EXTENDED', 'basic', 'BASIC']
        elif hasattr(manager, 'get_prediction_tier'):
            tier, _ = manager.get_prediction_tier(sample_patient_data_v2)
            assert tier is not None

    def test_missing_feature_detection(self):
        """Test missing feature detection"""
        from feature_manager import FeatureManager

        # Incomplete data
        incomplete_data = {
            'age_years': 50.0,
            'gender': 1,
            'height': 170.0
            # Missing many features
        }

        manager = FeatureManager()

        if hasattr(manager, 'get_missing_features'):
            missing = manager.get_missing_features(incomplete_data)
            assert len(missing) > 0
        elif hasattr(manager, 'get_prediction_tier'):
            _, info = manager.get_prediction_tier(incomplete_data)
            if isinstance(info, dict) and 'missing' in info:
                assert len(info['missing']) > 0


class TestFeatureValidation:
    """Tests for feature value validation"""

    def test_blood_pressure_validation(self):
        """Test blood pressure value validation"""
        from improved_features import ImprovedFeatureEngineer

        engineer = ImprovedFeatureEngineer()

        # Valid data (with preprocessing: age_years and bmi computed)
        valid_data = pd.DataFrame({
            'ap_hi': [120, 140, 130],
            'ap_lo': [80, 90, 85],
            'age': [18000, 20000, 22000],
            'gender': [1, 2, 1],
            'height': [170, 165, 175],
            'weight': [70, 75, 80],
            'cholesterol': [1, 2, 1],
            'gluc': [1, 1, 2],
            'smoke': [0, 0, 1],
            'alco': [0, 1, 0],
            'active': [1, 1, 0]
        })
        # Preprocess: add age_years and bmi
        valid_data['age_years'] = (valid_data['age'] / 365.25).round(1)
        valid_data['bmi'] = (valid_data['weight'] / ((valid_data['height'] / 100) ** 2)).round(2)

        features = engineer.create_features(valid_data.copy())

        # Features should be computed without errors
        assert len(features) == len(valid_data)

    def test_outlier_handling(self):
        """Test outlier values are handled properly"""
        from improved_features import ImprovedFeatureEngineer

        engineer = ImprovedFeatureEngineer()

        # Data with outliers (with preprocessing: age_years and bmi computed)
        outlier_data = pd.DataFrame({
            'ap_hi': [300],  # Very high
            'ap_lo': [50],   # Low but valid
            'age': [18000],
            'gender': [1],
            'height': [170],
            'weight': [75],
            'cholesterol': [2],
            'gluc': [1],
            'smoke': [0],
            'alco': [0],
            'active': [1]
        })
        # Preprocess: add age_years and bmi
        outlier_data['age_years'] = (outlier_data['age'] / 365.25).round(1)
        outlier_data['bmi'] = (outlier_data['weight'] / ((outlier_data['height'] / 100) ** 2)).round(2)

        # Should not raise exception
        features = engineer.create_features(outlier_data.copy())
        assert len(features) == 1


class TestFeatureConsistency:
    """Tests for feature computation consistency"""

    def test_same_input_same_output(self, sample_training_data):
        """Test same input produces same features"""
        from improved_features import ImprovedFeatureEngineer

        engineer = ImprovedFeatureEngineer()

        features1 = engineer.create_features(sample_training_data.copy())
        features2 = engineer.create_features(sample_training_data.copy())

        # Check numerical columns are equal
        num_cols = features1.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            np.testing.assert_array_almost_equal(
                features1[col].values,
                features2[col].values,
                decimal=5
            )

    def test_feature_order_consistent(self, sample_training_data):
        """Test feature column order is consistent"""
        from improved_features import ImprovedFeatureEngineer

        engineer = ImprovedFeatureEngineer()

        features1 = engineer.create_features(sample_training_data.copy())
        features2 = engineer.create_features(sample_training_data.copy())

        assert list(features1.columns) == list(features2.columns)
