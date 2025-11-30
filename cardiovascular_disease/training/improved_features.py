"""
Improved feature engineering based on statistical analysis
Implements high-priority feature recommendations
"""
import pandas as pd
import numpy as np


class ImprovedFeatureEngineer:
    """
    Enhanced feature engineering for cardiovascular disease prediction
    Based on comprehensive statistical analysis
    """

    @staticmethod
    def add_blood_pressure_features(df):
        """
        Add derived blood pressure features

        Features:
        - pulse_pressure: Difference between systolic and diastolic (arterial stiffness indicator)
        - mean_arterial_pressure: Average arterial pressure during cardiac cycle
        - hypertension_stage: Clinical hypertension classification (0-3)
        """
        df = df.copy()

        # Pulse Pressure (Correlation: 0.339, MI: 0.0722)
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

        # Mean Arterial Pressure (Correlation: 0.413, MI: 0.1082)
        # MAP = (SBP + 2*DBP) / 3
        df['mean_arterial_pressure'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3

        # Hypertension Stage (MI: 0.0992)
        # Based on ACC/AHA guidelines
        df['hypertension_stage'] = 0  # Normal: <120/<80
        df.loc[(df['ap_hi'] >= 120) | (df['ap_lo'] >= 80), 'hypertension_stage'] = 1  # Elevated
        df.loc[(df['ap_hi'] >= 130) | (df['ap_lo'] >= 80), 'hypertension_stage'] = 2  # Stage 1
        df.loc[(df['ap_hi'] >= 140) | (df['ap_lo'] >= 90), 'hypertension_stage'] = 3  # Stage 2

        return df

    @staticmethod
    def add_bmi_features(df):
        """
        Add BMI-related categorical features

        Features:
        - bmi_category: WHO BMI classification (0=normal, 1=underweight, 2=overweight, 3=obese)
        """
        df = df.copy()

        # BMI Category (MI: 0.0206)
        df['bmi_category'] = 0  # Normal (18.5-25)
        df.loc[df['bmi'] < 18.5, 'bmi_category'] = 1  # Underweight
        df.loc[df['bmi'] >= 25, 'bmi_category'] = 2  # Overweight
        df.loc[df['bmi'] >= 30, 'bmi_category'] = 3  # Obese

        return df

    @staticmethod
    def add_age_features(df):
        """
        Add age-related categorical features

        Features:
        - age_group: Age risk categories (0=<40, 1=40-50, 2=50-60, 3=60+)
        """
        df = df.copy()

        # Age Group (MI: 0.0272)
        df['age_group'] = pd.cut(
            df['age_years'],
            bins=[0, 40, 50, 60, 100],
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(int)

        return df

    @staticmethod
    def add_composite_scores(df):
        """
        Add composite health risk scores

        Features:
        - health_risk_composite: Combined cholesterol, glucose, and lifestyle score
        - lifestyle_risk_score: Combined behavioral risk factors
        """
        df = df.copy()

        # Health Risk Composite (Correlation: 0.173, MI: 0.0123)
        # Combines metabolic and lifestyle factors
        df['health_risk_composite'] = (
            (df['cholesterol'] - 1) +  # Scale to 0-2
            (df['gluc'] - 1) +  # Scale to 0-2
            df['smoke'] +
            df['alco'] +
            (1 - df['active'])  # Invert so 1 = inactive (risk factor)
        )

        # Lifestyle Risk Score (Correlation: 0.015, MI: 0.0002)
        df['lifestyle_risk_score'] = df['smoke'] + df['alco'] + (1 - df['active'])

        return df

    @staticmethod
    def add_interaction_features(df):
        """
        Add interaction terms (experimental)

        Features:
        - age_bmi_interaction: Captures combined effect of age and obesity
        - bp_bmi_interaction: Captures combined effect of BP and obesity
        """
        df = df.copy()

        # Age-BMI Interaction (Correlation: 0.274, MI: 0.0471)
        df['age_bmi_interaction'] = df['age_years'] * df['bmi']

        # BP-BMI Interaction (Correlation: 0.331, MI: 0.0963)
        # Use MAP if available, otherwise calculate it
        if 'mean_arterial_pressure' not in df.columns:
            df['mean_arterial_pressure'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
        df['bp_bmi_interaction'] = df['mean_arterial_pressure'] * df['bmi']

        return df

    @staticmethod
    def add_polynomial_features(df):
        """
        Add polynomial features (experimental)

        Features:
        - age_squared: Captures non-linear age effects
        - bmi_squared: Captures non-linear BMI effects
        """
        df = df.copy()

        # Polynomial features
        df['age_squared'] = df['age_years'] ** 2
        df['bmi_squared'] = df['bmi'] ** 2

        return df

    @staticmethod
    def engineer_all_features(df, include_experimental=False):
        """
        Apply all feature engineering steps

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with basic features
        include_experimental : bool
            Whether to include experimental features (interactions, polynomials)

        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered features
        """
        print("Applying improved feature engineering...")

        # High priority features
        df = ImprovedFeatureEngineer.add_blood_pressure_features(df)
        print("  ✓ Blood pressure features added")

        df = ImprovedFeatureEngineer.add_bmi_features(df)
        print("  ✓ BMI features added")

        df = ImprovedFeatureEngineer.add_age_features(df)
        print("  ✓ Age features added")

        df = ImprovedFeatureEngineer.add_composite_scores(df)
        print("  ✓ Composite scores added")

        # Experimental features (optional)
        if include_experimental:
            df = ImprovedFeatureEngineer.add_interaction_features(df)
            print("  ✓ Interaction features added")

            df = ImprovedFeatureEngineer.add_polynomial_features(df)
            print("  ✓ Polynomial features added")

        return df

    @staticmethod
    def get_feature_columns(include_experimental=False, remove_weight=True):
        """
        Get list of feature columns after engineering

        Parameters:
        -----------
        include_experimental : bool
            Whether experimental features are included
        remove_weight : bool
            Whether to remove weight (redundant with BMI)

        Returns:
        --------
        list
            List of feature column names
        """
        # Base features
        features = [
            'age_years', 'gender', 'height', 'bmi',
            'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active'
        ]

        # Add weight if not removing
        if not remove_weight:
            features.insert(3, 'weight')

        # High priority engineered features
        features.extend([
            'pulse_pressure',
            'mean_arterial_pressure',
            'hypertension_stage',
            'bmi_category',
            'age_group',
            'health_risk_composite',
            'lifestyle_risk_score'
        ])

        # Experimental features
        if include_experimental:
            features.extend([
                'age_bmi_interaction',
                'bp_bmi_interaction',
                'age_squared',
                'bmi_squared'
            ])

        return features


# Example usage
if __name__ == "__main__":
    from config import DATA_FILE, OUTLIER_THRESHOLDS

    print("="*80)
    print("IMPROVED FEATURE ENGINEERING - DEMO")
    print("="*80)

    # Load data
    print(f"\nLoading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE, delimiter=';')
    print(f"Loaded {len(df)} records")

    # Apply basic preprocessing
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
    print(f"After preprocessing: {len(df)} records")

    # Apply improved feature engineering
    df_engineered = ImprovedFeatureEngineer.engineer_all_features(df, include_experimental=True)

    print(f"\nOriginal features: {len(OUTLIER_THRESHOLDS) + 8}")
    print(f"Engineered features: {len(ImprovedFeatureEngineer.get_feature_columns(include_experimental=True))}")

    print("\nNew features added:")
    original_cols = set(df.columns)
    new_cols = set(df_engineered.columns) - original_cols
    for col in sorted(new_cols):
        print(f"  - {col}")

    print("\nSample of engineered features:")
    feature_cols = ImprovedFeatureEngineer.get_feature_columns(include_experimental=True)
    print(df_engineered[feature_cols].head())

    print("\n" + "="*80)
    print("Ready to integrate into training pipeline!")
    print("="*80)
