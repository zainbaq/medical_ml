"""
Demographic Subgroup Validation Module

Validates model performance and fairness across demographic subgroups to ensure
the cardiovascular disease prediction model performs equitably across:
- Gender groups
- Age groups
- BMI categories
- Race/ethnicity (when available)

This module computes:
- Performance metrics by subgroup
- Disparity ratios
- Calibration analysis
- Fairness recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SubgroupMetrics:
    """Metrics for a single subgroup."""
    subgroup_name: str
    subgroup_value: Any
    n_samples: int
    n_positive: int
    prevalence: float
    accuracy: float
    precision: float
    recall: float  # Sensitivity
    specificity: float
    f1_score: float
    roc_auc: float
    brier_score: float


@dataclass
class DisparityMetrics:
    """Disparity metrics comparing subgroups."""
    metric_name: str
    reference_group: str
    reference_value: float
    comparison_group: str
    comparison_value: float
    disparity_ratio: float
    absolute_difference: float
    is_acceptable: bool  # Within acceptable threshold


class SubgroupValidator:
    """
    Validates model fairness across demographic subgroups.

    Ensures that the cardiovascular disease prediction model performs
    consistently across different demographic groups, identifying
    potential biases and recommending mitigations.
    """

    # Subgroup definitions
    SUBGROUPS = {
        'gender': {
            'column': 'gender',
            'values': {1: 'Female', 2: 'Male'},
            'min_samples': 500
        },
        'age_group': {
            'column': 'age_group',
            'values': {0: '18-40', 1: '40-50', 2: '50-60', 3: '60+'},
            'min_samples': 300
        },
        'bmi_category': {
            'column': 'bmi_category',
            'values': {0: 'Normal', 1: 'Underweight', 2: 'Overweight', 3: 'Obese'},
            'min_samples': 200
        },
        'hypertension_stage': {
            'column': 'hypertension_stage',
            'values': {0: 'Normal', 1: 'Elevated', 2: 'Stage 1', 3: 'Stage 2'},
            'min_samples': 200
        }
    }

    # Acceptable disparity thresholds
    DISPARITY_THRESHOLDS = {
        'roc_auc': 0.05,      # Max 5% AUC difference
        'accuracy': 0.05,     # Max 5% accuracy difference
        'recall': 0.10,       # Max 10% sensitivity difference
        'precision': 0.10,    # Max 10% precision difference
        'brier_score': 0.05   # Max 0.05 Brier score difference
    }

    def __init__(
        self,
        disparity_thresholds: Optional[Dict[str, float]] = None,
        min_subgroup_size: int = 100
    ):
        """
        Initialize the subgroup validator.

        Args:
            disparity_thresholds: Custom disparity thresholds
            min_subgroup_size: Minimum samples required for subgroup analysis
        """
        self.disparity_thresholds = disparity_thresholds or self.DISPARITY_THRESHOLDS.copy()
        self.min_subgroup_size = min_subgroup_size
        self.subgroup_results: Dict[str, List[SubgroupMetrics]] = {}
        self.disparity_results: Dict[str, List[DisparityMetrics]] = {}

    def validate_subgroups(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        subgroup_data: pd.DataFrame,
        subgroup_column: str
    ) -> pd.DataFrame:
        """
        Compute metrics for each subgroup within a demographic category.

        Args:
            model: Trained classifier with predict and predict_proba methods
            X_test: Test feature matrix
            y_test: Test target vector
            subgroup_data: DataFrame containing subgroup column
            subgroup_column: Name of the subgroup column

        Returns:
            DataFrame with metrics for each subgroup
        """
        if subgroup_column not in subgroup_data.columns:
            logger.warning(f"Subgroup column '{subgroup_column}' not found")
            return pd.DataFrame()

        subgroup_values = subgroup_data[subgroup_column].unique()
        results = []

        for value in subgroup_values:
            mask = subgroup_data[subgroup_column] == value
            n_samples = mask.sum()

            if n_samples < self.min_subgroup_size:
                logger.warning(
                    f"Subgroup {subgroup_column}={value} has only {n_samples} samples, "
                    f"below minimum {self.min_subgroup_size}"
                )
                continue

            X_sub = X_test[mask]
            y_sub = y_test[mask]

            metrics = self._compute_metrics(model, X_sub, y_sub)
            metrics['subgroup'] = subgroup_column
            metrics['value'] = value
            metrics['n_samples'] = n_samples

            results.append(metrics)

        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        self.subgroup_results[subgroup_column] = results_df

        return results_df

    def _compute_metrics(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all metrics for a subset of data.

        Args:
            model: Trained classifier
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else np.nan,
            'brier_score': brier_score_loss(y, y_proba),
            'prevalence': y.mean()
        }

        # Confusion matrix metrics
        cm = confusion_matrix(y, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['true_positives'] = tp
            metrics['false_positives'] = fp
            metrics['true_negatives'] = tn
            metrics['false_negatives'] = fn

        return metrics

    def compute_disparity_metrics(
        self,
        results_df: pd.DataFrame,
        reference_value: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Compute disparity ratios between subgroups.

        Args:
            results_df: DataFrame from validate_subgroups
            reference_value: Reference subgroup value (uses largest if None)

        Returns:
            DataFrame with disparity metrics
        """
        if results_df.empty:
            return pd.DataFrame()

        # Use largest group as reference if not specified
        if reference_value is None:
            reference_value = results_df.loc[results_df['n_samples'].idxmax(), 'value']

        reference_row = results_df[results_df['value'] == reference_value].iloc[0]
        comparison_rows = results_df[results_df['value'] != reference_value]

        disparity_results = []

        for _, comp_row in comparison_rows.iterrows():
            for metric in ['roc_auc', 'accuracy', 'recall', 'precision', 'brier_score']:
                ref_val = reference_row[metric]
                comp_val = comp_row[metric]

                if pd.isna(ref_val) or pd.isna(comp_val) or ref_val == 0:
                    continue

                disparity_ratio = comp_val / ref_val
                abs_diff = abs(comp_val - ref_val)
                threshold = self.disparity_thresholds.get(metric, 0.1)

                disparity_results.append({
                    'subgroup': reference_row['subgroup'],
                    'metric': metric,
                    'reference_group': str(reference_value),
                    'reference_value': ref_val,
                    'comparison_group': str(comp_row['value']),
                    'comparison_value': comp_val,
                    'disparity_ratio': disparity_ratio,
                    'absolute_difference': abs_diff,
                    'threshold': threshold,
                    'is_acceptable': abs_diff <= threshold
                })

        return pd.DataFrame(disparity_results)

    def validate_all_subgroups(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Validate model across all defined subgroups.

        Args:
            model: Trained classifier
            X_test: Test feature matrix
            y_test: Test target vector
            feature_df: DataFrame with subgroup columns

        Returns:
            Dictionary mapping subgroup names to results DataFrames
        """
        all_results = {}

        for subgroup_name, config in self.SUBGROUPS.items():
            column = config['column']

            if column not in feature_df.columns:
                logger.info(f"Skipping {subgroup_name}: column not in data")
                continue

            logger.info(f"Validating subgroup: {subgroup_name}")

            results = self.validate_subgroups(
                model, X_test, y_test, feature_df, column
            )

            if not results.empty:
                all_results[subgroup_name] = results

                # Compute disparities
                disparities = self.compute_disparity_metrics(results)
                if not disparities.empty:
                    self.disparity_results[subgroup_name] = disparities

        return all_results

    def compute_calibration_by_subgroup(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        subgroup_data: pd.DataFrame,
        subgroup_column: str,
        n_bins: int = 10
    ) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute calibration curves for each subgroup.

        Args:
            model: Trained classifier
            X_test: Test feature matrix
            y_test: Test target vector
            subgroup_data: DataFrame with subgroup column
            subgroup_column: Subgroup column name
            n_bins: Number of calibration bins

        Returns:
            Dictionary mapping subgroup values to (mean_predicted, fraction_positives)
        """
        calibration_results = {}

        y_proba = model.predict_proba(X_test)[:, 1]

        for value in subgroup_data[subgroup_column].unique():
            mask = subgroup_data[subgroup_column] == value
            n_samples = mask.sum()

            if n_samples < self.min_subgroup_size:
                continue

            y_sub = y_test[mask]
            prob_sub = y_proba[mask]

            try:
                fraction_pos, mean_predicted = calibration_curve(
                    y_sub, prob_sub, n_bins=n_bins, strategy='uniform'
                )
                calibration_results[value] = (mean_predicted, fraction_pos)
            except Exception as e:
                logger.warning(f"Calibration failed for {subgroup_column}={value}: {e}")

        return calibration_results

    def generate_fairness_report(self) -> str:
        """
        Generate a comprehensive fairness report.

        Returns:
            Formatted string report
        """
        lines = [
            "=" * 70,
            "CARDIOVASCULAR DISEASE MODEL - FAIRNESS REPORT",
            "=" * 70,
            ""
        ]

        # Summary of subgroup performance
        for subgroup_name, results in self.subgroup_results.items():
            lines.append(f"\n{subgroup_name.upper()}")
            lines.append("-" * 50)

            if isinstance(results, pd.DataFrame):
                for _, row in results.iterrows():
                    lines.append(
                        f"  {row['value']}: "
                        f"n={row['n_samples']}, "
                        f"AUC={row['roc_auc']:.3f}, "
                        f"Acc={row['accuracy']:.3f}, "
                        f"Recall={row['recall']:.3f}"
                    )

        # Disparity analysis
        lines.append("\n" + "=" * 70)
        lines.append("DISPARITY ANALYSIS")
        lines.append("=" * 70)

        issues_found = False
        for subgroup_name, disparities in self.disparity_results.items():
            if isinstance(disparities, pd.DataFrame):
                unacceptable = disparities[~disparities['is_acceptable']]

                if not unacceptable.empty:
                    issues_found = True
                    lines.append(f"\n{subgroup_name}: Issues Detected")
                    for _, row in unacceptable.iterrows():
                        lines.append(
                            f"  - {row['metric']}: {row['reference_group']} ({row['reference_value']:.3f}) vs "
                            f"{row['comparison_group']} ({row['comparison_value']:.3f}) - "
                            f"diff={row['absolute_difference']:.3f}, threshold={row['threshold']}"
                        )

        if not issues_found:
            lines.append("\nAll disparity metrics within acceptable thresholds.")

        # Recommendations
        lines.append("\n" + "=" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 70)

        if issues_found:
            lines.append("\nTo address identified disparities, consider:")
            lines.append("  1. Collect more data from underrepresented groups")
            lines.append("  2. Use sample weighting to balance subgroups")
            lines.append("  3. Train separate models for significantly different groups")
            lines.append("  4. Apply post-hoc calibration by subgroup")
        else:
            lines.append("\nModel performs equitably across tested subgroups.")
            lines.append("Continue monitoring with new data.")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics from validation.

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'subgroups_tested': list(self.subgroup_results.keys()),
            'total_subgroups': sum(
                len(df) if isinstance(df, pd.DataFrame) else 0
                for df in self.subgroup_results.values()
            ),
            'disparity_issues': 0,
            'worst_disparity': None,
            'overall_fair': True
        }

        for subgroup_name, disparities in self.disparity_results.items():
            if isinstance(disparities, pd.DataFrame):
                unacceptable = disparities[~disparities['is_acceptable']]
                summary['disparity_issues'] += len(unacceptable)

                if not unacceptable.empty:
                    summary['overall_fair'] = False
                    worst = unacceptable.loc[unacceptable['absolute_difference'].idxmax()]
                    if (summary['worst_disparity'] is None or
                            worst['absolute_difference'] > summary['worst_disparity']['absolute_difference']):
                        summary['worst_disparity'] = worst.to_dict()

        return summary


def main():
    """Test the subgroup validator."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("SUBGROUP VALIDATOR TEST")
    print("=" * 60)

    # Generate synthetic data with subgroups
    np.random.seed(42)
    n_samples = 5000

    # Create features
    age = np.random.uniform(30, 80, n_samples)
    gender = np.random.choice([1, 2], n_samples)
    bmi = np.random.uniform(18, 40, n_samples)

    # Create age groups
    age_group = pd.cut(age, bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3]).astype(int)

    # Create BMI categories
    bmi_category = np.zeros(n_samples, dtype=int)
    bmi_category[bmi < 18.5] = 1
    bmi_category[(bmi >= 25) & (bmi < 30)] = 2
    bmi_category[bmi >= 30] = 3

    # Create target with some subgroup bias
    base_risk = 0.3 + (age - 30) / 100 + (bmi - 18) / 100
    y = (np.random.random(n_samples) < base_risk).astype(int)

    # Create feature matrix
    X = np.column_stack([age, gender, bmi])

    # Create subgroup DataFrame
    subgroup_df = pd.DataFrame({
        'age_group': age_group,
        'gender': gender,
        'bmi_category': bmi_category
    })

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    subgroup_train, subgroup_test = train_test_split(
        subgroup_df, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Validate subgroups
    validator = SubgroupValidator()
    results = validator.validate_all_subgroups(
        model, X_test, y_test, subgroup_test
    )

    # Print report
    print(validator.generate_fairness_report())

    # Print summary
    summary = validator.get_summary_statistics()
    print(f"\nSummary:")
    print(f"  Subgroups tested: {summary['subgroups_tested']}")
    print(f"  Total subgroups: {summary['total_subgroups']}")
    print(f"  Disparity issues: {summary['disparity_issues']}")
    print(f"  Overall fair: {summary['overall_fair']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
