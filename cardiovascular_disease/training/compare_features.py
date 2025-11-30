"""
Quick comparison of original vs improved feature engineering
Run this after training to evaluate feature improvements
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

from config import DATA_FILE, FEATURE_COLUMNS, TARGET_COLUMN, OUTLIER_THRESHOLDS, RANDOM_STATE
from improved_features import ImprovedFeatureEngineer


def load_and_preprocess():
    """Load and preprocess data"""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, delimiter=';')

    # Apply preprocessing
    df['age_years'] = (df['age'] / 365.25).round(1)
    df = df.drop('age', axis=1)
    df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)

    # Remove outliers
    for feature, (min_val, max_val) in OUTLIER_THRESHOLDS.items():
        df = df[(df[feature] >= min_val) & (df[feature] <= max_val)]
    df = df[df['ap_lo'] < df['ap_hi']]
    df = df[df['height'] > 0]
    df = df[df['weight'] > 0]
    df = df.drop('id', axis=1)

    print(f"Preprocessed {len(df)} records")
    return df


def evaluate_features(X, y, feature_set_name):
    """Evaluate a feature set with cross-validation"""
    print(f"\nEvaluating {feature_set_name}...")
    print(f"  Features: {X.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test with multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    }

    results = {}
    for model_name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc', n_jobs=-1)
        results[model_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"  {model_name:20s}: ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    return results


def main():
    print("="*80)
    print("FEATURE ENGINEERING COMPARISON")
    print("="*80)

    # Load data
    df = load_and_preprocess()
    y = df[TARGET_COLUMN]

    # Evaluate original features
    print("\n" + "="*80)
    print("1. ORIGINAL FEATURES")
    print("="*80)
    X_original = df[FEATURE_COLUMNS]
    results_original = evaluate_features(X_original, y, "Original Features")

    # Evaluate improved features (high priority only)
    print("\n" + "="*80)
    print("2. IMPROVED FEATURES (High Priority)")
    print("="*80)
    df_improved = ImprovedFeatureEngineer.engineer_all_features(df, include_experimental=False)
    feature_cols_improved = ImprovedFeatureEngineer.get_feature_columns(include_experimental=False, remove_weight=True)
    X_improved = df_improved[feature_cols_improved]
    results_improved = evaluate_features(X_improved, y, "Improved Features (High Priority)")

    # Evaluate improved features (with experimental)
    print("\n" + "="*80)
    print("3. IMPROVED FEATURES (With Experimental)")
    print("="*80)
    df_experimental = ImprovedFeatureEngineer.engineer_all_features(df, include_experimental=True)
    feature_cols_experimental = ImprovedFeatureEngineer.get_feature_columns(include_experimental=True, remove_weight=True)
    X_experimental = df_experimental[feature_cols_experimental]
    results_experimental = evaluate_features(X_experimental, y, "Improved Features (With Experimental)")

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)

    print("\nLogistic Regression:")
    print(f"  Original:             {results_original['Logistic Regression']['mean']:.4f}")
    print(f"  Improved (High Pri):  {results_improved['Logistic Regression']['mean']:.4f} ", end="")
    improvement = (results_improved['Logistic Regression']['mean'] - results_original['Logistic Regression']['mean']) * 100
    print(f"({improvement:+.2f}%)")
    print(f"  Improved (w/ Exp):    {results_experimental['Logistic Regression']['mean']:.4f} ", end="")
    improvement = (results_experimental['Logistic Regression']['mean'] - results_original['Logistic Regression']['mean']) * 100
    print(f"({improvement:+.2f}%)")

    print("\nRandom Forest:")
    print(f"  Original:             {results_original['Random Forest']['mean']:.4f}")
    print(f"  Improved (High Pri):  {results_improved['Random Forest']['mean']:.4f} ", end="")
    improvement = (results_improved['Random Forest']['mean'] - results_original['Random Forest']['mean']) * 100
    print(f"({improvement:+.2f}%)")
    print(f"  Improved (w/ Exp):    {results_experimental['Random Forest']['mean']:.4f} ", end="")
    improvement = (results_experimental['Random Forest']['mean'] - results_original['Random Forest']['mean']) * 100
    print(f"({improvement:+.2f}%)")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    best_lr = max([
        ('Original', results_original['Logistic Regression']['mean']),
        ('Improved (High Pri)', results_improved['Logistic Regression']['mean']),
        ('Improved (w/ Exp)', results_experimental['Logistic Regression']['mean'])
    ], key=lambda x: x[1])

    best_rf = max([
        ('Original', results_original['Random Forest']['mean']),
        ('Improved (High Pri)', results_improved['Random Forest']['mean']),
        ('Improved (w/ Exp)', results_experimental['Random Forest']['mean'])
    ], key=lambda x: x[1])

    print(f"\nBest performing features:")
    print(f"  Logistic Regression: {best_lr[0]} (ROC-AUC: {best_lr[1]:.4f})")
    print(f"  Random Forest:       {best_rf[0]} (ROC-AUC: {best_rf[1]:.4f})")

    if best_lr[0] == 'Improved (High Pri)' or best_rf[0] == 'Improved (High Pri)':
        print("\n✓ RECOMMENDATION: Use Improved Features (High Priority)")
        print("  These features show significant improvement without adding too much complexity")
    elif best_lr[0] == 'Improved (w/ Exp)' or best_rf[0] == 'Improved (w/ Exp)':
        print("\n✓ RECOMMENDATION: Use Improved Features (With Experimental)")
        print("  Experimental features provide additional benefit - proceed with caution for overfitting")
    else:
        print("\n✓ RECOMMENDATION: Stick with Original Features")
        print("  Improved features did not provide significant benefit")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
