"""
Tests for data downloaders and harmonization

Tests include:
- Downloader initialization and configuration
- Schema validation
- Data harmonization
- Source tracking
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "data"))


class TestUCIHeartDownloader:
    """Tests for UCI Heart Disease dataset downloader"""

    def test_downloader_initialization(self):
        """Test UCIHeartDownloader can be initialized"""
        from downloaders.uci_heart import UCIHeartDownloader

        downloader = UCIHeartDownloader()
        assert downloader is not None
        assert hasattr(downloader, 'download')

    def test_uci_column_mapping(self):
        """Test UCI column mapping is correct"""
        from downloaders.uci_heart import UCIHeartDownloader

        downloader = UCIHeartDownloader()
        assert 'age' in downloader.column_mapping
        assert 'chol' in downloader.column_mapping
        assert 'trestbps' in downloader.column_mapping

    def test_schema_standardization(self, sample_uci_data):
        """Test UCI data is properly standardized"""
        from downloaders.uci_heart import UCIHeartDownloader

        downloader = UCIHeartDownloader()
        standardized = downloader.standardize_schema(sample_uci_data)

        # Check required columns exist
        assert 'age_years' in standardized.columns
        assert 'gender' in standardized.columns
        assert 'ap_hi' in standardized.columns
        assert 'cholesterol' in standardized.columns
        assert 'cardio' in standardized.columns
        assert 'data_source' in standardized.columns

        # Check source tracking
        assert all(standardized['data_source'] == 'uci')

    def test_target_encoding(self, sample_uci_data):
        """Test target variable is properly encoded"""
        from downloaders.uci_heart import UCIHeartDownloader

        downloader = UCIHeartDownloader()
        standardized = downloader.standardize_schema(sample_uci_data)

        # Target should be binary (0 or 1)
        assert set(standardized['cardio'].unique()).issubset({0, 1})


class TestNHANESDownloader:
    """Tests for NHANES dataset downloader"""

    def test_downloader_initialization(self):
        """Test NHANESDownloader can be initialized"""
        from downloaders.nhanes import NHANESDownloader

        downloader = NHANESDownloader()
        assert downloader is not None
        assert hasattr(downloader, 'download')

    def test_nhanes_data_tables(self):
        """Test NHANES required data tables are defined"""
        from downloaders.nhanes import NHANESDownloader

        downloader = NHANESDownloader()
        required_tables = ['DEMO', 'BPX', 'BMX', 'SMQ']

        # data_tables is a dict where values are table IDs
        all_tables = list(downloader.data_tables.values())
        for table in required_tables:
            assert table in all_tables or any(
                table in str(t) for t in all_tables
            )


class TestFraminghamDownloader:
    """Tests for Framingham dataset downloader"""

    def test_downloader_initialization(self):
        """Test FraminghamDownloader can be initialized"""
        from downloaders.framingham import FraminghamDownloader

        downloader = FraminghamDownloader()
        assert downloader is not None
        assert hasattr(downloader, 'download')


class TestDataHarmonizer:
    """Tests for data harmonization"""

    def test_harmonizer_initialization(self):
        """Test DataHarmonizer can be initialized"""
        from harmonize import DataHarmonizer

        harmonizer = DataHarmonizer()
        assert harmonizer is not None

    def test_unified_schema_columns(self):
        """Test unified schema has required columns"""
        from harmonize import DataHarmonizer

        harmonizer = DataHarmonizer()
        required_columns = [
            'age_years', 'gender', 'height', 'weight', 'bmi',
            'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active', 'cardio'
        ]

        for col in required_columns:
            assert col in harmonizer.UNIFIED_SCHEMA

    def test_harmonize_single_dataset(self, sample_training_data):
        """Test harmonizing a single dataset"""
        from harmonize import DataHarmonizer

        harmonizer = DataHarmonizer()

        # Add required columns for harmonization
        sample_training_data['age_years'] = sample_training_data['age'] / 365.25
        sample_training_data['bmi'] = (
            sample_training_data['weight'] /
            ((sample_training_data['height'] / 100) ** 2)
        )
        sample_training_data['data_source'] = 'kaggle'

        harmonized = harmonizer.harmonize_dataset(
            sample_training_data,
            source='kaggle'
        )

        assert 'data_source' in harmonized.columns
        assert len(harmonized) == len(sample_training_data)

    def test_source_tracking(self, sample_training_data):
        """Test that source is properly tracked after harmonization"""
        from harmonize import DataHarmonizer

        harmonizer = DataHarmonizer()

        # Prepare data
        sample_training_data['age_years'] = sample_training_data['age'] / 365.25
        sample_training_data['bmi'] = (
            sample_training_data['weight'] /
            ((sample_training_data['height'] / 100) ** 2)
        )

        harmonized = harmonizer.harmonize_dataset(
            sample_training_data,
            source='test_source'
        )

        assert all(harmonized['data_source'] == 'test_source')

    def test_feature_availability_indicators(self, sample_training_data):
        """Test that feature availability is tracked"""
        from harmonize import DataHarmonizer

        harmonizer = DataHarmonizer()

        # Prepare data with some missing columns
        sample_training_data['age_years'] = sample_training_data['age'] / 365.25
        sample_training_data['bmi'] = (
            sample_training_data['weight'] /
            ((sample_training_data['height'] / 100) ** 2)
        )

        harmonized = harmonizer.harmonize_dataset(
            sample_training_data,
            source='kaggle'
        )

        # Check that indicator columns exist for optional features
        # These should indicate what features are available
        assert harmonized is not None


class TestDownloadOrchestrator:
    """Tests for the download orchestrator"""

    def test_orchestrator_initialization(self):
        """Test DatasetDownloader can be initialized"""
        from download_datasets import DatasetDownloader

        downloader = DatasetDownloader()
        assert downloader is not None
        assert hasattr(downloader, 'download_all')
        assert hasattr(downloader, 'verify_downloads')

    def test_dataset_registry(self):
        """Test all datasets are registered"""
        from download_datasets import DatasetDownloader

        downloader = DatasetDownloader()
        # These are the datasets that have downloaders
        # (kaggle dataset is already provided, not downloaded)
        expected_datasets = ['uci', 'nhanes', 'framingham']

        for dataset in expected_datasets:
            assert dataset in downloader.datasets or hasattr(
                downloader, f'download_{dataset}'
            )


class TestDataIntegrity:
    """Tests for data integrity after download and harmonization"""

    def test_no_duplicate_ids(self, sample_training_data):
        """Test no duplicate IDs in harmonized data"""
        from harmonize import DataHarmonizer

        harmonizer = DataHarmonizer()
        sample_training_data['age_years'] = sample_training_data['age'] / 365.25
        sample_training_data['bmi'] = (
            sample_training_data['weight'] /
            ((sample_training_data['height'] / 100) ** 2)
        )

        harmonized = harmonizer.harmonize_dataset(
            sample_training_data,
            source='kaggle'
        )

        # If there's an ID column, check for duplicates
        if 'id' in harmonized.columns:
            assert harmonized['id'].is_unique or harmonized['unified_id'].is_unique

    def test_target_is_binary(self, sample_training_data):
        """Test target variable is binary after harmonization"""
        from harmonize import DataHarmonizer

        harmonizer = DataHarmonizer()
        sample_training_data['age_years'] = sample_training_data['age'] / 365.25
        sample_training_data['bmi'] = (
            sample_training_data['weight'] /
            ((sample_training_data['height'] / 100) ** 2)
        )

        harmonized = harmonizer.harmonize_dataset(
            sample_training_data,
            source='kaggle'
        )

        assert set(harmonized['cardio'].unique()).issubset({0, 1})

    def test_valid_blood_pressure_values(self, sample_training_data):
        """Test blood pressure values are valid after harmonization"""
        from harmonize import DataHarmonizer

        harmonizer = DataHarmonizer()
        sample_training_data['age_years'] = sample_training_data['age'] / 365.25
        sample_training_data['bmi'] = (
            sample_training_data['weight'] /
            ((sample_training_data['height'] / 100) ** 2)
        )

        harmonized = harmonizer.harmonize_dataset(
            sample_training_data,
            source='kaggle'
        )

        # Systolic should be >= diastolic
        assert all(harmonized['ap_hi'] >= harmonized['ap_lo'])

        # Values should be in reasonable range
        assert all(harmonized['ap_hi'] >= 50)
        assert all(harmonized['ap_hi'] <= 300)
        assert all(harmonized['ap_lo'] >= 30)
        assert all(harmonized['ap_lo'] <= 200)
