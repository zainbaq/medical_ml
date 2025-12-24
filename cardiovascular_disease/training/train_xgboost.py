"""
XGBoost Training with Optuna Hyperparameter Optimization

This module provides advanced training capabilities for cardiovascular disease
prediction using XGBoost with:
- Bayesian hyperparameter optimization via Optuna
- SHAP explainability integration
- Cross-validation with early stopping
- Model calibration for reliable probability estimates
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("XGBoost is required. Install with: pip install xgboost")

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError("Optuna is required. Install with: pip install optuna")

try:
    import shap
except ImportError:
    shap = None
    logging.warning("SHAP not installed. Explainability features disabled.")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """
    XGBoost trainer with Optuna hyperparameter optimization and SHAP explainability.

    This class provides:
    - Bayesian hyperparameter optimization with Optuna
    - Stratified k-fold cross-validation
    - Early stopping to prevent overfitting
    - Model calibration for reliable probabilities
    - SHAP value computation for explainability
    """

    # Default hyperparameter search space
    DEFAULT_PARAM_SPACE = {
        'max_depth': {'type': 'int', 'low': 3, 'high': 12},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'n_estimators': {'type': 'int', 'low': 100, 'high': 1000, 'step': 50},
        'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'reg_alpha': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
        'reg_lambda': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
        'gamma': {'type': 'float', 'low': 0, 'high': 0.5}
    }

    # Performance targets
    PERFORMANCE_TARGETS = {
        'roc_auc': 0.82,
        'accuracy': 0.78,
        'sensitivity': 0.75
    }

    def __init__(
        self,
        n_trials: int = 100,
        cv_folds: int = 5,
        timeout: Optional[int] = 3600,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        enable_shap: bool = True
    ):
        """
        Initialize the XGBoost trainer.

        Args:
            n_trials: Number of Optuna trials for hyperparameter search
            cv_folds: Number of cross-validation folds
            timeout: Maximum optimization time in seconds (None for no limit)
            early_stopping_rounds: Early stopping patience
            random_state: Random seed for reproducibility
            enable_shap: Whether to compute SHAP values
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.timeout = timeout
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.enable_shap = enable_shap and shap is not None

        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.shap_explainer = None
        self.feature_names = None
        self.study = None
        self.metrics = {}

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for an Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        for name, space in self.DEFAULT_PARAM_SPACE.items():
            if space['type'] == 'int':
                params[name] = trial.suggest_int(
                    name,
                    space['low'],
                    space['high'],
                    step=space.get('step', 1)
                )
            elif space['type'] == 'float':
                params[name] = trial.suggest_float(
                    name,
                    space['low'],
                    space['high'],
                    log=space.get('log', False)
                )
            elif space['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, space['choices'])

        return params

    def objective(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial
            X: Feature matrix
            y: Target vector

        Returns:
            Mean cross-validation ROC-AUC score
        """
        params = self._suggest_params(trial)

        # Add fixed parameters
        params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': self.random_state,
            'n_jobs': -1
        })

        # Handle class imbalance
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        params['scale_pos_weight'] = trial.suggest_float(
            'scale_pos_weight',
            0.5 * scale_pos_weight,
            2.0 * scale_pos_weight
        )

        # Create model
        model = xgb.XGBClassifier(**params)

        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )

        return scores.mean()

    def train_with_optimization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_weights: Optional[np.ndarray] = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model with Optuna hyperparameter optimization.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional list of feature names
            sample_weights: Optional sample weights

        Returns:
            Trained XGBoost classifier with best hyperparameters
        """
        self.feature_names = feature_names

        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")

        # Create Optuna study
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='cvd_xgboost_optimization'
        )

        # Optimize
        self.study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
            n_jobs=1  # Optuna handles parallelism internally
        )

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info(f"Best trial ROC-AUC: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        # Train final model with best parameters
        final_params = self.best_params.copy()
        final_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': self.random_state,
            'n_jobs': -1
        })

        self.best_model = xgb.XGBClassifier(**final_params)
        self.best_model.fit(
            X, y,
            sample_weight=sample_weights,
            verbose=False
        )

        # Compute SHAP values
        if self.enable_shap:
            self._compute_shap_explainer(X)

        return self.best_model

    def train_simple(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model with provided or default parameters (no optimization).

        Args:
            X: Feature matrix
            y: Target vector
            params: Optional hyperparameters
            feature_names: Optional feature names

        Returns:
            Trained XGBoost classifier
        """
        self.feature_names = feature_names

        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'gamma': 0.1
            }

        params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': self.random_state,
            'n_jobs': -1
        })

        self.best_params = params
        self.best_model = xgb.XGBClassifier(**params)
        self.best_model.fit(X, y)

        if self.enable_shap:
            self._compute_shap_explainer(X)

        return self.best_model

    def _compute_shap_explainer(self, X: np.ndarray) -> None:
        """
        Compute SHAP explainer for the trained model.

        Args:
            X: Feature matrix used for training
        """
        if not self.enable_shap or self.best_model is None:
            return

        logger.info("Computing SHAP explainer...")
        try:
            self.shap_explainer = shap.TreeExplainer(self.best_model)
            # Compute base SHAP values on a sample for caching
            sample_size = min(1000, len(X))
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            _ = self.shap_explainer.shap_values(X[sample_idx])
            logger.info("SHAP explainer ready")
        except Exception as e:
            logger.warning(f"Failed to create SHAP explainer: {e}")
            self.shap_explainer = None

    def compute_shap_values(
        self,
        X: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values for given features.

        Args:
            X: Feature matrix

        Returns:
            SHAP values array or None if SHAP is disabled
        """
        if self.shap_explainer is None:
            return None

        return self.shap_explainer.shap_values(X)

    def explain_prediction(
        self,
        features: np.ndarray,
        feature_values: Optional[Dict] = None,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction.

        Args:
            features: Feature array for one sample
            feature_values: Optional dict of feature name to value
            top_n: Number of top factors to return

        Returns:
            Dictionary with explanation details
        """
        if self.best_model is None:
            raise ValueError("Model not trained")

        # Get prediction
        features_2d = features.reshape(1, -1) if features.ndim == 1 else features
        prediction = self.best_model.predict(features_2d)[0]
        probability = self.best_model.predict_proba(features_2d)[0][1]

        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_factors': [],
            'protective_factors': []
        }

        # Get SHAP values if available
        if self.shap_explainer is not None:
            shap_values = self.compute_shap_values(features_2d)[0]

            # Pair with feature names
            if self.feature_names is not None:
                feature_importance = list(zip(self.feature_names, shap_values))
            else:
                feature_importance = [
                    (f'feature_{i}', v) for i, v in enumerate(shap_values)
                ]

            # Sort by absolute importance
            sorted_features = sorted(
                feature_importance,
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Split into risk and protective factors
            for name, value in sorted_features[:top_n]:
                factor = {
                    'name': name,
                    'shap_value': float(value),
                    'contribution': 'increases' if value > 0 else 'decreases'
                }
                if feature_values and name in feature_values:
                    factor['value'] = feature_values[name]

                if value > 0:
                    result['risk_factors'].append(factor)
                else:
                    result['protective_factors'].append(factor)

        return result

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.

        Args:
            X_test: Test feature matrix
            y_test: Test target vector

        Returns:
            Dictionary of evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("Model not trained")

        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test)[:, 1]

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),  # Sensitivity
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'brier_score': brier_score_loss(y_test, y_proba)
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        self.metrics['specificity'] = tn / (tn + fp)
        self.metrics['sensitivity'] = tp / (tp + fn)

        # Check against targets
        self.metrics['meets_targets'] = all([
            self.metrics['roc_auc'] >= self.PERFORMANCE_TARGETS['roc_auc'],
            self.metrics['accuracy'] >= self.PERFORMANCE_TARGETS['accuracy'],
            self.metrics['sensitivity'] >= self.PERFORMANCE_TARGETS['sensitivity']
        ])

        return self.metrics

    def calibrate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'isotonic'
    ) -> CalibratedClassifierCV:
        """
        Calibrate the model for reliable probability estimates.

        Args:
            X: Calibration feature matrix
            y: Calibration target vector
            method: Calibration method ('isotonic' or 'sigmoid')

        Returns:
            Calibrated classifier
        """
        if self.best_model is None:
            raise ValueError("Model not trained")

        calibrated = CalibratedClassifierCV(
            self.best_model,
            method=method,
            cv='prefit'
        )
        calibrated.fit(X, y)

        return calibrated

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from both XGBoost and SHAP.

        Returns:
            DataFrame with feature importance scores
        """
        if self.best_model is None:
            raise ValueError("Model not trained")

        importance_df = pd.DataFrame({
            'feature': self.feature_names or [f'f{i}' for i in range(len(self.best_model.feature_importances_))],
            'xgb_importance': self.best_model.feature_importances_
        })

        # Add SHAP importance if available
        if self.shap_explainer is not None and hasattr(self, '_shap_values_cache'):
            shap_importance = np.abs(self._shap_values_cache).mean(axis=0)
            importance_df['shap_importance'] = shap_importance

        importance_df = importance_df.sort_values('xgb_importance', ascending=False)
        return importance_df

    def save_model(
        self,
        output_dir: Path,
        model_name: str = 'xgboost_cvd'
    ) -> Dict[str, Path]:
        """
        Save the trained model, scaler, and metadata.

        Args:
            output_dir: Output directory
            model_name: Base name for saved files

        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {}

        # Save model
        model_path = output_dir / f'{model_name}_{timestamp}.pkl'
        joblib.dump(self.best_model, model_path)
        paths['model'] = model_path

        # Save SHAP explainer if available
        if self.shap_explainer is not None:
            shap_path = output_dir / f'shap_explainer_{timestamp}.pkl'
            joblib.dump(self.shap_explainer, shap_path)
            paths['shap_explainer'] = shap_path

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_type': 'XGBoost',
            'best_params': self.best_params,
            'best_cv_score': self.best_score,
            'metrics': self.metrics,
            'feature_names': self.feature_names,
            'n_trials': self.n_trials,
            'cv_folds': self.cv_folds,
            'performance_targets': self.PERFORMANCE_TARGETS,
            'meets_targets': self.metrics.get('meets_targets', False)
        }

        metadata_path = output_dir / f'metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        paths['metadata'] = metadata_path

        # Update latest model info
        latest_info = {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'shap_path': str(paths.get('shap_explainer', '')),
            'timestamp': timestamp
        }

        latest_path = output_dir / 'latest_model_info.json'
        with open(latest_path, 'w') as f:
            json.dump(latest_info, f, indent=2)
        paths['latest_info'] = latest_path

        logger.info(f"Model saved to {model_path}")
        return paths


def main():
    """Test the XGBoost trainer."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("XGBOOST TRAINER TEST")
    print("=" * 60)

    # Generate synthetic data
    X, y = make_classification(
        n_samples=5000,
        n_features=18,
        n_informative=10,
        n_redundant=3,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train with limited trials for testing
    trainer = XGBoostTrainer(
        n_trials=10,  # Use fewer trials for testing
        cv_folds=3,
        timeout=120
    )

    print("\nTraining XGBoost model with Optuna optimization...")
    model = trainer.train_with_optimization(X_train, y_train, feature_names)

    print("\nEvaluating model...")
    metrics = trainer.evaluate(X_test, y_test)

    print("\nMetrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    print("\nFeature importance:")
    importance = trainer.get_feature_importance()
    print(importance.head(10))

    # Test explanation
    print("\nSample prediction explanation:")
    explanation = trainer.explain_prediction(X_test[0], top_n=3)
    print(f"  Prediction: {explanation['prediction']}")
    print(f"  Probability: {explanation['probability']:.4f}")
    print(f"  Top risk factors: {len(explanation['risk_factors'])}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
