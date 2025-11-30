"""
Robust training pipeline for cardiovascular disease prediction
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_FILE, MODELS_DIR, TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    MODELS_CONFIG, OUTLIER_THRESHOLDS, FEATURE_COLUMNS, TARGET_COLUMN,
    USE_IMPROVED_FEATURES, INCLUDE_EXPERIMENTAL_FEATURES, REMOVE_WEIGHT
)
from improved_features import ImprovedFeatureEngineer


class CardiovascularModelTrainer:
    """
    Comprehensive training pipeline for cardiovascular disease prediction
    """

    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=RANDOM_STATE),
            'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'logistic_regression': LogisticRegression(random_state=RANDOM_STATE)
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        # Feature engineering configuration
        self.use_improved_features = USE_IMPROVED_FEATURES
        self.include_experimental = INCLUDE_EXPERIMENTAL_FEATURES
        self.remove_weight = REMOVE_WEIGHT

    def load_data(self):
        """Load the cardiovascular dataset"""
        print(f"Loading data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE, delimiter=';')
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df

    def preprocess_data(self, df):
        """
        Comprehensive data preprocessing including:
        - Converting age from days to years
        - Handling outliers
        - Feature engineering (BMI)
        - Removing invalid records
        """
        print("\nPreprocessing data...")
        df = df.copy()

        # Convert age from days to years
        df['age_years'] = (df['age'] / 365.25).round(1)
        df = df.drop('age', axis=1)

        # Calculate BMI
        df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)

        # Remove outliers
        initial_count = len(df)
        for feature, (min_val, max_val) in OUTLIER_THRESHOLDS.items():
            df = df[(df[feature] >= min_val) & (df[feature] <= max_val)]

        print(f"Removed {initial_count - len(df)} outlier records")

        # Remove records where diastolic > systolic
        df = df[df['ap_lo'] < df['ap_hi']]

        # Remove invalid values
        df = df[df['height'] > 0]
        df = df[df['weight'] > 0]

        # Drop ID column
        df = df.drop('id', axis=1)

        print(f"Final dataset: {len(df)} records")

        # Apply improved feature engineering if enabled
        if self.use_improved_features:
            print("\nApplying improved feature engineering...")
            df = ImprovedFeatureEngineer.engineer_all_features(
                df,
                include_experimental=self.include_experimental
            )
            print(f"Feature engineering complete. Total features: {len(df.columns) - 1}")

        return df

    def prepare_features(self, df):
        """Separate features and target"""
        # Handle weight removal if configured (redundant with BMI)
        feature_cols = list(FEATURE_COLUMNS)
        if self.remove_weight and 'weight' in feature_cols:
            feature_cols.remove('weight')
            print(f"Removed 'weight' feature (redundant with BMI)")

        # Display feature engineering configuration
        print(f"\nFeature Configuration:")
        print(f"  Using improved features: {self.use_improved_features}")
        if self.use_improved_features:
            print(f"  Including experimental features: {self.include_experimental}")
        print(f"  Total features: {len(feature_cols)}")

        X = df[feature_cols]
        y = df[TARGET_COLUMN]
        self.feature_names = feature_cols
        return X, y

    def train_model(self, X_train, y_train, model_name='random_forest', use_grid_search=True):
        """
        Train a model with optional hyperparameter tuning
        """
        print(f"\nTraining {model_name}...")
        model = self.models[model_name]

        if use_grid_search and model_name in MODELS_CONFIG:
            print(f"Performing grid search with {CV_FOLDS}-fold cross-validation...")
            grid_search = GridSearchCV(
                model,
                MODELS_CONFIG[model_name],
                cv=CV_FOLDS,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            return best_model, grid_search.best_params_, grid_search.best_score_
        else:
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='roc_auc')
            print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            return model, {}, cv_scores.mean()

    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Comprehensive model evaluation
        """
        print(f"\nEvaluating {model_name}...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return metrics

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate all models, select the best one
        """
        results = {}

        for model_name in self.models.keys():
            trained_model, best_params, cv_score = self.train_model(
                X_train, y_train, model_name, use_grid_search=True
            )
            metrics = self.evaluate_model(trained_model, X_test, y_test, model_name)

            results[model_name] = {
                'model': trained_model,
                'best_params': best_params,
                'cv_score': cv_score,
                'test_metrics': metrics
            }

        # Select best model based on ROC-AUC
        best_model_name = max(results.items(), key=lambda x: x[1]['test_metrics']['roc_auc'])[0]
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        self.metrics = results[best_model_name]['test_metrics']

        print(f"\n{'='*60}")
        print(f"Best model: {best_model_name}")
        print(f"ROC-AUC: {self.metrics['roc_auc']:.4f}")
        print(f"{'='*60}")

        return results

    def save_model(self, results):
        """
        Save the best model, scaler, and metadata
        """
        MODELS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = MODELS_DIR / f"cardiovascular_model_{timestamp}.pkl"
        joblib.dump(self.best_model, model_path)
        print(f"\nModel saved to: {model_path}")

        # Save scaler
        scaler_path = MODELS_DIR / f"scaler_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'best_params': results[self.best_model_name]['best_params'],
            'cv_score': results[self.best_model_name]['cv_score'],
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'feature_engineering': {
                'use_improved_features': self.use_improved_features,
                'include_experimental': self.include_experimental,
                'remove_weight': self.remove_weight,
                'num_features': len(self.feature_names)
            }
        }

        metadata_path = MODELS_DIR / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {metadata_path}")

        # Save pointer to latest model
        latest_path = MODELS_DIR / "latest_model_info.json"
        with open(latest_path, 'w') as f:
            json.dump({
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'metadata_path': str(metadata_path),
                'timestamp': timestamp
            }, f, indent=4)

        print(f"\nLatest model info saved to: {latest_path}")

        return model_path, scaler_path, metadata_path

    def run_pipeline(self):
        """
        Execute the complete training pipeline
        """
        print("="*60)
        print("CARDIOVASCULAR DISEASE PREDICTION - TRAINING PIPELINE")
        print("="*60)

        # Load data
        df = self.load_data()

        # Preprocess
        df = self.preprocess_data(df)

        # Prepare features
        X, y = self.prepare_features(df)

        # Split data
        print(f"\nSplitting data: {int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train all models
        results = self.train_all_models(X_train_scaled, y_train, X_test_scaled, y_test)

        # Save the best model
        model_path, scaler_path, metadata_path = self.save_model(results)

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)

        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'metadata_path': metadata_path,
            'results': results
        }


def main():
    """Main entry point for training"""
    trainer = CardiovascularModelTrainer()
    trainer.run_pipeline()


if __name__ == "__main__":
    main()
