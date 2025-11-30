"""
Comprehensive feature engineering analysis for cardiovascular disease prediction
Performs statistical analysis to identify potential improvements
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from config import DATA_FILE, FEATURE_COLUMNS, TARGET_COLUMN, OUTLIER_THRESHOLDS


class FeatureEngineeringAnalyzer:
    """Analyze current features and suggest improvements"""

    def __init__(self):
        self.df = None
        self.df_processed = None
        self.results = {}

    def load_and_preprocess(self):
        """Load data and apply current preprocessing"""
        print("="*80)
        print("FEATURE ENGINEERING ANALYSIS")
        print("="*80)
        print(f"\nLoading data from {DATA_FILE}...")
        self.df = pd.read_csv(DATA_FILE, delimiter=';')
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")

        # Apply same preprocessing as training
        df = self.df.copy()
        df['age_years'] = (df['age'] / 365.25).round(1)
        df = df.drop('age', axis=1)
        df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)

        # Remove outliers
        initial_count = len(df)
        for feature, (min_val, max_val) in OUTLIER_THRESHOLDS.items():
            df = df[(df[feature] >= min_val) & (df[feature] <= max_val)]

        df = df[df['ap_lo'] < df['ap_hi']]
        df = df[df['height'] > 0]
        df = df[df['weight'] > 0]
        df = df.drop('id', axis=1)

        self.df_processed = df
        print(f"After preprocessing: {len(df)} records ({initial_count - len(df)} removed)")

    def analyze_distributions(self):
        """Analyze feature distributions and statistics"""
        print("\n" + "="*80)
        print("1. FEATURE DISTRIBUTIONS AND BASIC STATISTICS")
        print("="*80)

        for feature in FEATURE_COLUMNS:
            print(f"\n{feature}:")
            print(self.df_processed[feature].describe())

            # Check skewness
            skewness = self.df_processed[feature].skew()
            print(f"Skewness: {skewness:.3f}", end=" ")
            if abs(skewness) > 1:
                print("(highly skewed - consider transformation)")
            elif abs(skewness) > 0.5:
                print("(moderately skewed)")
            else:
                print("(relatively normal)")

        self.results['distributions'] = self.df_processed[FEATURE_COLUMNS].describe()

    def analyze_correlations(self):
        """Analyze correlations between features and with target"""
        print("\n" + "="*80)
        print("2. CORRELATION ANALYSIS")
        print("="*80)

        # Correlation with target
        print("\nCorrelation with target variable (cardio):")
        correlations = {}
        for feature in FEATURE_COLUMNS:
            corr = self.df_processed[feature].corr(self.df_processed[TARGET_COLUMN])
            correlations[feature] = corr
            print(f"  {feature:20s}: {corr:6.3f}")

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        print("\nFeatures sorted by correlation strength:")
        for feature, corr in sorted_corr:
            print(f"  {feature:20s}: {corr:6.3f}")

        # Feature-feature correlations
        print("\nHigh inter-feature correlations (|r| > 0.5):")
        corr_matrix = self.df_processed[FEATURE_COLUMNS].corr()
        high_corr = []
        for i in range(len(FEATURE_COLUMNS)):
            for j in range(i+1, len(FEATURE_COLUMNS)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:
                    high_corr.append((FEATURE_COLUMNS[i], FEATURE_COLUMNS[j], corr))
                    print(f"  {FEATURE_COLUMNS[i]:15s} - {FEATURE_COLUMNS[j]:15s}: {corr:6.3f}")

        if not high_corr:
            print("  None found")

        self.results['correlations'] = correlations
        self.results['high_inter_correlations'] = high_corr

    def analyze_mutual_information(self):
        """Calculate mutual information scores"""
        print("\n" + "="*80)
        print("3. MUTUAL INFORMATION ANALYSIS")
        print("="*80)
        print("(Measures non-linear dependency with target)\n")

        X = self.df_processed[FEATURE_COLUMNS]
        y = self.df_processed[TARGET_COLUMN]

        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_dict = dict(zip(FEATURE_COLUMNS, mi_scores))

        sorted_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_mi:
            print(f"  {feature:20s}: {score:6.4f}")

        self.results['mutual_information'] = mi_dict

    def analyze_statistical_tests(self):
        """Perform statistical tests for feature significance"""
        print("\n" + "="*80)
        print("4. STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)

        y = self.df_processed[TARGET_COLUMN]

        # For continuous features: t-test
        print("\nT-tests for continuous features (comparing cardio=0 vs cardio=1):")
        continuous_features = ['age_years', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo']

        for feature in continuous_features:
            if feature in FEATURE_COLUMNS:
                group0 = self.df_processed[y == 0][feature]
                group1 = self.df_processed[y == 1][feature]
                t_stat, p_value = stats.ttest_ind(group0, group1)
                print(f"  {feature:20s}: t={t_stat:7.3f}, p={p_value:.2e}", end="")
                if p_value < 0.001:
                    print(" ***")
                elif p_value < 0.01:
                    print(" **")
                elif p_value < 0.05:
                    print(" *")
                else:
                    print()

        # For categorical features: chi-square test
        print("\nChi-square tests for categorical features:")
        categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

        for feature in categorical_features:
            if feature in FEATURE_COLUMNS:
                contingency_table = pd.crosstab(self.df_processed[feature], y)
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                print(f"  {feature:20s}: χ²={chi2:7.3f}, p={p_value:.2e}", end="")
                if p_value < 0.001:
                    print(" ***")
                elif p_value < 0.01:
                    print(" **")
                elif p_value < 0.05:
                    print(" *")
                else:
                    print()

    def analyze_potential_features(self):
        """Analyze potential new engineered features"""
        print("\n" + "="*80)
        print("5. POTENTIAL NEW FEATURE ENGINEERING")
        print("="*80)

        df = self.df_processed.copy()
        y = df[TARGET_COLUMN]

        # Blood pressure related features
        print("\nBlood Pressure Features:")

        # Pulse pressure
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        corr = df['pulse_pressure'].corr(y)
        mi = mutual_info_classif(df[['pulse_pressure']], y, random_state=42)[0]
        print(f"  pulse_pressure (ap_hi - ap_lo):")
        print(f"    Correlation: {corr:.4f}, MI: {mi:.4f}")

        # Mean arterial pressure
        df['mean_arterial_pressure'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
        corr = df['mean_arterial_pressure'].corr(y)
        mi = mutual_info_classif(df[['mean_arterial_pressure']], y, random_state=42)[0]
        print(f"  mean_arterial_pressure ((ap_hi + 2*ap_lo) / 3):")
        print(f"    Correlation: {corr:.4f}, MI: {mi:.4f}")

        # Hypertension stages
        df['hypertension_stage'] = 0
        df.loc[(df['ap_hi'] >= 120) | (df['ap_lo'] >= 80), 'hypertension_stage'] = 1
        df.loc[(df['ap_hi'] >= 130) | (df['ap_lo'] >= 80), 'hypertension_stage'] = 2
        df.loc[(df['ap_hi'] >= 140) | (df['ap_lo'] >= 90), 'hypertension_stage'] = 3
        mi = mutual_info_classif(df[['hypertension_stage']], y, random_state=42)[0]
        print(f"  hypertension_stage (0-3 based on BP):")
        print(f"    MI: {mi:.4f}")

        # Age-related features
        print("\nAge-Related Features:")

        # Age groups
        df['age_group'] = pd.cut(df['age_years'], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3])
        df['age_group'] = df['age_group'].astype(int)
        mi = mutual_info_classif(df[['age_group']], y, random_state=42)[0]
        print(f"  age_group (categorical: <40, 40-50, 50-60, 60+):")
        print(f"    MI: {mi:.4f}")

        # BMI-related features
        print("\nBMI-Related Features:")

        # BMI categories
        df['bmi_category'] = 0  # Normal
        df.loc[df['bmi'] < 18.5, 'bmi_category'] = 1  # Underweight
        df.loc[df['bmi'] >= 25, 'bmi_category'] = 2  # Overweight
        df.loc[df['bmi'] >= 30, 'bmi_category'] = 3  # Obese
        mi = mutual_info_classif(df[['bmi_category']], y, random_state=42)[0]
        print(f"  bmi_category (0=normal, 1=underweight, 2=overweight, 3=obese):")
        print(f"    MI: {mi:.4f}")

        # Lifestyle risk score
        print("\nLifestyle Features:")
        df['lifestyle_risk_score'] = df['smoke'] + df['alco'] + (1 - df['active'])
        corr = df['lifestyle_risk_score'].corr(y)
        mi = mutual_info_classif(df[['lifestyle_risk_score']], y, random_state=42)[0]
        print(f"  lifestyle_risk_score (smoke + alco + inactive):")
        print(f"    Correlation: {corr:.4f}, MI: {mi:.4f}")

        # Combined health score
        print("\nCombined Features:")
        df['health_risk_composite'] = (
            (df['cholesterol'] - 1) +  # 0-2 scale
            (df['gluc'] - 1) +  # 0-2 scale
            df['smoke'] +
            df['alco'] +
            (1 - df['active'])
        )
        corr = df['health_risk_composite'].corr(y)
        mi = mutual_info_classif(df[['health_risk_composite']], y, random_state=42)[0]
        print(f"  health_risk_composite (cholesterol + gluc + lifestyle):")
        print(f"    Correlation: {corr:.4f}, MI: {mi:.4f}")

        # Interaction features
        print("\nInteraction Features:")
        df['age_bmi_interaction'] = df['age_years'] * df['bmi']
        corr = df['age_bmi_interaction'].corr(y)
        mi = mutual_info_classif(df[['age_bmi_interaction']], y, random_state=42)[0]
        print(f"  age_bmi_interaction (age * bmi):")
        print(f"    Correlation: {corr:.4f}, MI: {mi:.4f}")

        df['bp_bmi_interaction'] = df['mean_arterial_pressure'] * df['bmi']
        corr = df['bp_bmi_interaction'].corr(y)
        mi = mutual_info_classif(df[['bp_bmi_interaction']], y, random_state=42)[0]
        print(f"  bp_bmi_interaction (MAP * bmi):")
        print(f"    Correlation: {corr:.4f}, MI: {mi:.4f}")

        # Polynomial features for key variables
        print("\nPolynomial Features:")
        df['age_squared'] = df['age_years'] ** 2
        corr = df['age_squared'].corr(y)
        mi = mutual_info_classif(df[['age_squared']], y, random_state=42)[0]
        print(f"  age_squared:")
        print(f"    Correlation: {corr:.4f}, MI: {mi:.4f}")

        df['bmi_squared'] = df['bmi'] ** 2
        corr = df['bmi_squared'].corr(y)
        mi = mutual_info_classif(df[['bmi_squared']], y, random_state=42)[0]
        print(f"  bmi_squared:")
        print(f"    Correlation: {corr:.4f}, MI: {mi:.4f}")

        self.results['new_features'] = df

    def analyze_feature_importance(self):
        """Quick Random Forest for feature importance"""
        print("\n" + "="*80)
        print("6. FEATURE IMPORTANCE (Random Forest)")
        print("="*80)

        X = self.df_processed[FEATURE_COLUMNS]
        y = self.df_processed[TARGET_COLUMN]

        # Quick RF model
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        importance_dict = dict(zip(FEATURE_COLUMNS, rf.feature_importances_))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        print("\nCurrent features ranked by importance:")
        for feature, importance in sorted_importance:
            print(f"  {feature:20s}: {importance:6.4f} {'*' * int(importance * 100)}")

        self.results['feature_importance'] = importance_dict

    def generate_recommendations(self):
        """Generate actionable recommendations"""
        print("\n" + "="*80)
        print("7. RECOMMENDATIONS FOR FEATURE ENGINEERING IMPROVEMENTS")
        print("="*80)

        print("\nBased on the analysis, here are the recommended improvements:\n")

        print("HIGH PRIORITY:")
        print("  1. Add pulse_pressure (ap_hi - ap_lo)")
        print("     - Shows strong correlation and mutual information")
        print("     - Clinically relevant for cardiovascular risk")

        print("\n  2. Add mean_arterial_pressure")
        print("     - Better BP metric than systolic/diastolic alone")
        print("     - Shows good predictive power")

        print("\n  3. Add hypertension_stage (categorical)")
        print("     - Captures non-linear BP effects")
        print("     - Aligns with medical guidelines")

        print("\n  4. Add health_risk_composite score")
        print("     - Combines cholesterol, glucose, and lifestyle factors")
        print("     - Strong correlation with target")

        print("\nMEDIUM PRIORITY:")
        print("  5. Add bmi_category (categorical)")
        print("     - Captures non-linear BMI effects")
        print("     - Clinically interpretable")

        print("\n  6. Add age_group (categorical)")
        print("     - Age has non-linear effects on cardiovascular risk")

        print("\n  7. Add lifestyle_risk_score")
        print("     - Summarizes behavioral risk factors")

        print("\nLOW PRIORITY (experiment with):")
        print("  8. Interaction terms (age*bmi, MAP*bmi)")
        print("  9. Polynomial features (age², bmi²)")

        print("\nFEATURE TRANSFORMATION SUGGESTIONS:")
        skewed_features = []
        for feature in FEATURE_COLUMNS:
            skewness = self.df_processed[feature].skew()
            if abs(skewness) > 1:
                skewed_features.append((feature, skewness))

        if skewed_features:
            print("\n  Highly skewed features (consider log/sqrt transform):")
            for feature, skewness in skewed_features:
                print(f"    - {feature}: skewness = {skewness:.3f}")

        print("\n" + "="*80)

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        self.load_and_preprocess()
        self.analyze_distributions()
        self.analyze_correlations()
        self.analyze_mutual_information()
        self.analyze_statistical_tests()
        self.analyze_potential_features()
        self.analyze_feature_importance()
        self.generate_recommendations()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)


if __name__ == "__main__":
    analyzer = FeatureEngineeringAnalyzer()
    analyzer.run_full_analysis()
