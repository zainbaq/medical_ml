"""
Training script for Alzheimer's disease prediction model
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
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
    DROP_COLUMNS, TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    MODELS_CONFIG, GENDER_ENCODING, TARGET_ENCODING
)


def load_and_preprocess_data():
    """Load and preprocess the OASIS longitudinal dataset"""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    print(f"Initial dataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")

    # Drop unnecessary columns
    print(f"\nDropping columns: {DROP_COLUMNS}")
    df = df.drop(DROP_COLUMNS, axis=1, errors='ignore')

    # Create Gender column from M/F column
    if 'M/F' in df.columns:
        print("Creating Gender column from M/F...")
        df['Gender'] = df['M/F'].map(GENDER_ENCODING)
        df = df.drop('M/F', axis=1)

    # Encode target
    print(f"\nTarget value counts before encoding:")
    print(df[TARGET_COLUMN].value_counts())
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_ENCODING)

    # Handle missing values
    print(f"\nMissing values before handling:")
    print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum())

    # For SES: use median imputation
    if 'SES' in df.columns and df['SES'].isnull().sum() > 0:
        ses_median = df['SES'].median()
        print(f"\n*** IMPORTANT: SES median value = {ses_median} ***")
        print("*** Update MEDIAN_SES in backend/utils/preprocessing.py with this value ***")
        df['SES'].fillna(ses_median, inplace=True)

    # For MMSE: drop rows (critical feature)
    rows_before = len(df)
    df = df.dropna(subset=['MMSE'])
    rows_after = len(df)
    if rows_before != rows_after:
        print(f"Dropped {rows_before - rows_after} rows with missing MMSE")

    # Drop any remaining rows with NaN in required columns
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df.dropna(subset=required_cols)

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts())
    print(f"Demented (1): {(df[TARGET_COLUMN] == 1).sum()} samples")
    print(f"Non-demented (0): {(df[TARGET_COLUMN] == 0).sum()} samples")

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

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns (in order): {FEATURE_COLUMNS}")

    return X, y


def train_and_evaluate_model(model_name, model_class, param_grid, X_train, X_test, y_train, y_test, scaler):
    """Train and evaluate a model using GridSearchCV"""
    print(f"\n{'='*70}")
    print(f"Training {model_name}...")
    print(f"{'='*70}")

    # Scale data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        model_class,
        param_grid,
        cv=CV_FOLDS,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # Fit model
    grid_search.fit(X_train_scaled, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ROC-AUC
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = 0.0

    # Results
    results = {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    }

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"CV Score (ROC-AUC): {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    return results


def save_model(model, scaler, metadata, timestamp):
    """Save model, scaler, and metadata"""
    # Create timestamped filenames
    model_filename = f"alzheimer_model_{timestamp}.pkl"
    scaler_filename = f"scaler_{timestamp}.pkl"
    metadata_filename = f"metadata_{timestamp}.json"

    model_path = MODELS_DIR / model_filename
    scaler_path = MODELS_DIR / scaler_filename
    metadata_path = MODELS_DIR / metadata_filename

    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Add paths to metadata
    metadata['model_path'] = str(model_path)
    metadata['scaler_path'] = str(scaler_path)

    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Update latest_model_info.json
    latest_info = {
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'metadata_path': str(metadata_path),
        'timestamp': timestamp
    }

    latest_info_path = MODELS_DIR / 'latest_model_info.json'
    with open(latest_info_path, 'w') as f:
        json.dump(latest_info, f, indent=2)

    print(f"\n{'='*70}")
    print("Model saved successfully!")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Latest info: {latest_info_path}")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("ALZHEIMER'S DISEASE PREDICTION MODEL TRAINING")
    print("="*70)

    # Load and preprocess data
    df = load_and_preprocess_data()

    # Prepare features
    X, y = prepare_features(df)

    # Train/test split with stratification
    print(f"\nSplitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Fit scaler on training data
    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Train all models
    model_results = {}

    # SVM
    model_results['svm'] = train_and_evaluate_model(
        'SVM',
        SVC(probability=True, random_state=RANDOM_STATE),
        MODELS_CONFIG['svm'],
        X_train, X_test, y_train, y_test,
        scaler
    )

    # Random Forest
    model_results['random_forest'] = train_and_evaluate_model(
        'Random Forest',
        RandomForestClassifier(random_state=RANDOM_STATE),
        MODELS_CONFIG['random_forest'],
        X_train, X_test, y_train, y_test,
        scaler
    )

    # Gradient Boosting
    model_results['gradient_boosting'] = train_and_evaluate_model(
        'Gradient Boosting',
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        MODELS_CONFIG['gradient_boosting'],
        X_train, X_test, y_train, y_test,
        scaler
    )

    # Logistic Regression
    model_results['logistic_regression'] = train_and_evaluate_model(
        'Logistic Regression',
        LogisticRegression(random_state=RANDOM_STATE),
        MODELS_CONFIG['logistic_regression'],
        X_train, X_test, y_train, y_test,
        scaler
    )

    # Compare models
    print(f"\n{'='*70}")
    print("MODEL COMPARISON (by ROC-AUC)")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-"*70)

    for name, results in model_results.items():
        metrics = results['metrics']
        print(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} {metrics['roc_auc']:<12.4f}")

    # Select best model by ROC-AUC
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['metrics']['roc_auc'])
    best_results = model_results[best_model_name]

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"ROC-AUC: {best_results['metrics']['roc_auc']:.4f}")
    print(f"{'='*70}")

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create metadata
    metadata = {
        'timestamp': timestamp,
        'model_name': best_model_name,
        'feature_names': FEATURE_COLUMNS,
        'metrics': best_results['metrics'],
        'best_params': best_results['best_params'],
        'cv_score': best_results['cv_score']
    }

    # Save best model
    save_model(
        best_results['model'],
        scaler,
        metadata,
        timestamp
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
