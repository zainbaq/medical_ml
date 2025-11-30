"""
Training script for breast cancer prediction model
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_FILE, MODELS_DIR, FEATURE_COLUMNS, TARGET_COLUMN,
    DROP_COLUMNS, TEST_SIZE, RANDOM_STATE, CV_FOLDS, MODELS_CONFIG
)


def load_and_preprocess_data():
    """Load and preprocess the breast cancer dataset"""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    print(f"Dataset shape: {df.shape}")

    # Drop unnamed columns (from trailing commas in CSV)
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        print(f"Dropping unnamed columns: {unnamed_cols}")
        df = df.drop(unnamed_cols, axis=1)

    # Check diagnosis values before mapping
    print(f"\nDiagnosis value counts before mapping:")
    print(df[TARGET_COLUMN].value_counts())

    # Drop unnecessary columns
    df = df.drop(DROP_COLUMNS, axis=1, errors='ignore')

    # Convert diagnosis to binary: M=1 (Malignant), B=0 (Benign)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'M': 1, 'B': 0})

    # Check for NaN values after mapping
    if df[TARGET_COLUMN].isnull().sum() > 0:
        print(f"Warning: {df[TARGET_COLUMN].isnull().sum()} NaN values in diagnosis after mapping")
        print("Unique values before mapping might have issues")

    # Handle missing values in feature columns only
    rows_before = len(df)
    # Only drop rows where feature columns or target have NaN
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df.dropna(subset=required_cols)
    rows_after = len(df)

    if rows_before != rows_after:
        print(f"Warning: Dropped {rows_before - rows_after} rows with NaN values")

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Class distribution:\n{df[TARGET_COLUMN].value_counts()}")
    print(f"Malignant (1): {(df[TARGET_COLUMN] == 1).sum()} samples")
    print(f"Benign (0): {(df[TARGET_COLUMN] == 0).sum()} samples")

    return df


def prepare_features(df):
    """Prepare features and target"""
    # Check if all feature columns exist
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y


def train_and_evaluate_model(model_name, model_class, param_grid, X_train, X_test, y_train, y_test):
    """Train and evaluate a model using GridSearchCV"""
    print(f"\n{'='*70}")
    print(f"Training {model_name}...")
    print(f"{'='*70}")

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        model_class,
        param_grid,
        cv=CV_FOLDS,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_

    print(f"\nBest parameters: {best_params}")
    print(f"Best CV ROC-AUC: {cv_score:.4f}")

    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nTest Set Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")

    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }

    return best_model, best_params, cv_score, metrics


def save_model(model, scaler, model_name, best_params, cv_score, metrics):
    """Save model, scaler, and metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / f"breast_cancer_model_{timestamp}.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save scaler
    scaler_path = MODELS_DIR / f"scaler_{timestamp}.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_name': model_name,
        'feature_names': FEATURE_COLUMNS,
        'metrics': metrics,
        'best_params': best_params,
        'cv_score': cv_score,
        'model_path': str(model_path),
        'scaler_path': str(scaler_path)
    }

    metadata_path = MODELS_DIR / f"metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    # Update latest model info
    latest_info = {
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'metadata_path': str(metadata_path),
        'timestamp': timestamp
    }

    latest_info_path = MODELS_DIR / "latest_model_info.json"
    with open(latest_info_path, 'w') as f:
        json.dump(latest_info, f, indent=2)
    print(f"Latest model info updated: {latest_info_path}")

    return model_path, scaler_path, metadata_path


def main():
    """Main training pipeline"""
    print("="*70)
    print("BREAST CANCER PREDICTION MODEL - TRAINING PIPELINE")
    print("="*70)

    # Load and preprocess data
    df = load_and_preprocess_data()

    # Prepare features
    X, y = prepare_features(df)

    # Split data
    print(f"\nSplitting data (test size: {TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train models
    models = {
        'svm': (SVC(probability=True, random_state=RANDOM_STATE), MODELS_CONFIG['svm']),
        'random_forest': (RandomForestClassifier(random_state=RANDOM_STATE), MODELS_CONFIG['random_forest']),
        'gradient_boosting': (GradientBoostingClassifier(random_state=RANDOM_STATE), MODELS_CONFIG['gradient_boosting']),
        'logistic_regression': (LogisticRegression(random_state=RANDOM_STATE), MODELS_CONFIG['logistic_regression'])
    }

    results = {}
    for model_name, (model_class, param_grid) in models.items():
        model, params, cv_score, metrics = train_and_evaluate_model(
            model_name, model_class, param_grid,
            X_train, X_test, y_train, y_test
        )
        results[model_name] = {
            'model': model,
            'params': params,
            'cv_score': cv_score,
            'metrics': metrics
        }

    # Select best model based on ROC-AUC
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    for model_name, result in results.items():
        print(f"{model_name:20s} - ROC-AUC: {result['metrics']['roc_auc']:.4f}, "
              f"Accuracy: {result['metrics']['accuracy']:.4f}")

    best_model_name = max(results, key=lambda x: results[x]['metrics']['roc_auc'])
    best_result = results[best_model_name]

    print(f"\nBest model: {best_model_name}")
    print(f"ROC-AUC: {best_result['metrics']['roc_auc']:.4f}")

    # Save best model
    print("\n" + "="*70)
    print("SAVING BEST MODEL")
    print("="*70)

    model_path, scaler_path, metadata_path = save_model(
        best_result['model'],
        scaler,
        best_model_name,
        best_result['params'],
        best_result['cv_score'],
        best_result['metrics']
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best model: {best_model_name}")
    print(f"Model saved to: {model_path}")
    print(f"You can now start the API with: ./start_api.sh")


if __name__ == "__main__":
    main()
