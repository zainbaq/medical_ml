"""
Improved feature engineering based on statistical analysis
Implements high-priority feature recommendations

Extended in v2 to support:
- Metabolic features (when lab values available from NHANES/Framingham)
- Clinical features (from UCI dataset)
- Extended BP features (medication-aware)
- Modifiable risk tracking
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


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

    # =========================================================================
    # V2 Extended Features - For multi-dataset training
    # =========================================================================

    @staticmethod
    def add_metabolic_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add metabolic syndrome indicators (requires lab values from NHANES/Framingham).

        Features:
        - total_hdl_ratio: Total cholesterol / HDL ratio (strong CVD predictor)
        - metabolic_syndrome_score: Count of metabolic syndrome criteria met (0-5)
        - diabetes_risk_indicator: Risk based on glucose/HbA1c

        These features are only computed when lab values are available.
        Missing values are left as NaN for imputation or tiered modeling.
        """
        df = df.copy()

        # Total/HDL Cholesterol Ratio
        # Strong predictor: ratio > 5 indicates elevated CVD risk
        if 'total_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
            mask = (df['hdl_cholesterol'].notna()) & (df['hdl_cholesterol'] > 0)
            df.loc[mask, 'total_hdl_ratio'] = (
                df.loc[mask, 'total_cholesterol'] / df.loc[mask, 'hdl_cholesterol']
            )
        else:
            df['total_hdl_ratio'] = np.nan

        # Metabolic Syndrome Score (ATP III criteria)
        # Components: abdominal obesity, high triglycerides, low HDL, high BP, high fasting glucose
        df['metabolic_syndrome_score'] = 0

        # 1. Abdominal obesity (using BMI >= 30 as proxy)
        if 'bmi' in df.columns:
            df['metabolic_syndrome_score'] += (df['bmi'] >= 30).astype(int)

        # 2. High triglycerides (>= 150 mg/dL)
        if 'triglycerides' in df.columns:
            df['metabolic_syndrome_score'] += (
                df['triglycerides'].fillna(0) >= 150
            ).astype(int)

        # 3. Low HDL (< 40 for men, < 50 for women)
        if 'hdl_cholesterol' in df.columns and 'gender' in df.columns:
            male_low_hdl = (df['gender'] == 2) & (df['hdl_cholesterol'] < 40)
            female_low_hdl = (df['gender'] == 1) & (df['hdl_cholesterol'] < 50)
            df['metabolic_syndrome_score'] += (male_low_hdl | female_low_hdl).astype(int)

        # 4. High blood pressure (>= 130/85)
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            df['metabolic_syndrome_score'] += (
                (df['ap_hi'] >= 130) | (df['ap_lo'] >= 85)
            ).astype(int)

        # 5. High fasting glucose (>= 100 mg/dL)
        if 'fasting_glucose' in df.columns:
            df['metabolic_syndrome_score'] += (
                df['fasting_glucose'].fillna(0) >= 100
            ).astype(int)

        # Diabetes Risk Indicator
        # Based on HbA1c (>= 5.7 pre-diabetes, >= 6.5 diabetes) or fasting glucose
        df['diabetes_risk_indicator'] = 0

        if 'hba1c' in df.columns:
            df.loc[df['hba1c'] >= 5.7, 'diabetes_risk_indicator'] = 1
            df.loc[df['hba1c'] >= 6.5, 'diabetes_risk_indicator'] = 2
        elif 'fasting_glucose' in df.columns:
            df.loc[df['fasting_glucose'] >= 100, 'diabetes_risk_indicator'] = 1
            df.loc[df['fasting_glucose'] >= 126, 'diabetes_risk_indicator'] = 2
        elif 'gluc' in df.columns:
            # Use categorical glucose as fallback
            df.loc[df['gluc'] >= 2, 'diabetes_risk_indicator'] = 1
            df.loc[df['gluc'] >= 3, 'diabetes_risk_indicator'] = 2

        return df

    @staticmethod
    def add_extended_bp_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add extended blood pressure features with medication awareness.

        Features:
        - bp_index: Normalized BP considering both systolic and diastolic
        - bp_control_status: For patients on medication, indicates control level
        - hypertension_crisis: Flag for dangerously high BP values
        """
        df = df.copy()

        # BP Index (multiplicative risk indicator)
        # Normalized to 1.0 for ideal BP (120/80)
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            df['bp_index'] = (df['ap_hi'] / 120) * (df['ap_lo'] / 80)

            # Hypertensive crisis flag (>180/>120)
            df['hypertension_crisis'] = (
                (df['ap_hi'] > 180) | (df['ap_lo'] > 120)
            ).astype(int)

        # BP Control Status (for patients on medication)
        # 0 = not on meds, 1 = on meds & controlled, 2 = on meds & uncontrolled
        if 'bp_medication' in df.columns:
            df['bp_control_status'] = 0  # Not on medication

            on_meds = df['bp_medication'] == 1
            controlled = (df['ap_hi'] < 140) & (df['ap_lo'] < 90)

            df.loc[on_meds & controlled, 'bp_control_status'] = 1
            df.loc[on_meds & ~controlled, 'bp_control_status'] = 2
        else:
            df['bp_control_status'] = np.nan

        return df

    @staticmethod
    def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add clinical features from UCI dataset (exercise tests, ECG, etc.).

        Features:
        - exercise_capacity_score: Combined exercise test indicators
        - ecg_abnormality_indicator: Flag for abnormal resting ECG
        - ischemia_indicator: Evidence of myocardial ischemia
        """
        df = df.copy()

        # Exercise Capacity Score
        # Combines max heart rate achieved and exercise-induced angina
        if 'max_heart_rate' in df.columns:
            # Age-predicted max HR: 220 - age
            if 'age_years' in df.columns:
                predicted_max_hr = 220 - df['age_years']
                df['hr_reserve_pct'] = (
                    df['max_heart_rate'] / predicted_max_hr * 100
                ).clip(0, 150)
            else:
                df['hr_reserve_pct'] = np.nan

        if 'exercise_angina' in df.columns:
            df['exercise_capacity_score'] = 0
            # Good capacity: high HR achieved, no angina
            if 'hr_reserve_pct' in df.columns:
                good_hr = df['hr_reserve_pct'] >= 85
                no_angina = df['exercise_angina'] == 0
                df.loc[good_hr & no_angina, 'exercise_capacity_score'] = 2
                df.loc[good_hr & ~no_angina, 'exercise_capacity_score'] = 1
                df.loc[~good_hr & no_angina, 'exercise_capacity_score'] = 1
                # Poor capacity: low HR and angina = 0 (already set)
        else:
            df['exercise_capacity_score'] = np.nan

        # ECG Abnormality Indicator
        # resting_ecg: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy
        if 'resting_ecg' in df.columns:
            df['ecg_abnormality'] = (df['resting_ecg'] > 0).astype(int)
        else:
            df['ecg_abnormality'] = np.nan

        # Ischemia Indicator
        # Based on ST depression during exercise
        if 'st_depression' in df.columns:
            df['ischemia_indicator'] = 0
            df.loc[df['st_depression'] > 0, 'ischemia_indicator'] = 1
            df.loc[df['st_depression'] > 2, 'ischemia_indicator'] = 2  # Significant
        else:
            df['ischemia_indicator'] = np.nan

        return df

    @staticmethod
    def add_modifiable_risk_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features tracking modifiable risk factors.

        Features:
        - modifiable_risk_count: Count of modifiable risk factors present
        - modifiable_risk_score: Weighted score of modifiable risks
        - improvement_potential: Estimated potential for risk reduction
        """
        df = df.copy()

        # Modifiable Risk Count
        # Factors that can be changed through lifestyle or medication
        df['modifiable_risk_count'] = 0

        # High blood pressure (modifiable with medication/lifestyle)
        if 'ap_hi' in df.columns:
            df['modifiable_risk_count'] += (df['ap_hi'] >= 130).astype(int)

        # High cholesterol (modifiable with diet/statins)
        if 'cholesterol' in df.columns:
            df['modifiable_risk_count'] += (df['cholesterol'] >= 2).astype(int)

        # Smoking (cessation possible)
        if 'smoke' in df.columns:
            df['modifiable_risk_count'] += df['smoke'].fillna(0).astype(int)

        # Physical inactivity (can increase activity)
        if 'active' in df.columns:
            df['modifiable_risk_count'] += (1 - df['active'].fillna(1)).astype(int)

        # Obesity (weight loss possible)
        if 'bmi' in df.columns:
            df['modifiable_risk_count'] += (df['bmi'] >= 30).astype(int)

        # Alcohol (reduction possible)
        if 'alco' in df.columns:
            df['modifiable_risk_count'] += df['alco'].fillna(0).astype(int)

        # Modifiable Risk Score (weighted)
        # Weights based on relative CVD risk contribution
        df['modifiable_risk_score'] = 0

        weights = {
            'ap_hi': (130, 0.25),      # BP has high impact
            'cholesterol': (2, 0.20),   # Cholesterol significant
            'smoke': (1, 0.25),         # Smoking very high impact
            'active': (0, 0.15),        # Inactivity moderate
            'bmi': (30, 0.10),          # Obesity moderate
            'alco': (1, 0.05)           # Alcohol lower impact
        }

        for col, (threshold, weight) in weights.items():
            if col in df.columns:
                if col == 'active':
                    # Invert: active=0 is risk
                    risk = (df[col].fillna(1) == 0).astype(float)
                elif col in ['smoke', 'alco']:
                    risk = (df[col].fillna(0) >= threshold).astype(float)
                elif col == 'cholesterol':
                    risk = (df[col] >= threshold).astype(float)
                else:
                    risk = (df[col] >= threshold).astype(float)
                df['modifiable_risk_score'] += risk * weight

        # Improvement Potential (inverse of current risk)
        # Higher score = more room for improvement
        df['improvement_potential'] = df['modifiable_risk_count'] / 6 * 100

        return df

    @staticmethod
    def add_demographic_risk_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add demographic-adjusted risk features.

        Features:
        - cardiovascular_age_group: Fine-grained age risk brackets (5 categories)
        - gender_adjusted_age_risk: Age risk adjusted for gender differences
        """
        df = df.copy()

        # Cardiovascular Age Groups (5-year risk brackets)
        # Based on clinical risk stratification
        if 'age_years' in df.columns:
            df['cardiovascular_age_group'] = pd.cut(
                df['age_years'],
                bins=[0, 40, 55, 65, 75, 100],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True
            ).astype(float)  # Use float to allow NaN

        # Gender-Adjusted Age Risk
        # Women develop CVD ~10 years later than men on average
        if 'age_years' in df.columns and 'gender' in df.columns:
            df['gender_adjusted_age_risk'] = df['age_years'].copy()
            # Add 10 years to female age for risk equivalence
            df.loc[df['gender'] == 1, 'gender_adjusted_age_risk'] += 10

        return df

    @staticmethod
    def create_features(
        df: pd.DataFrame,
        include_experimental: bool = False,
        include_extended: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Alias for engineer_all_features for backward compatibility.
        """
        return ImprovedFeatureEngineer.engineer_all_features(
            df,
            include_experimental=include_experimental,
            include_extended=include_extended,
            verbose=verbose
        )

    @staticmethod
    def engineer_all_features(
        df: pd.DataFrame,
        include_experimental: bool = False,
        include_extended: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with basic features
        include_experimental : bool
            Whether to include experimental features (interactions, polynomials)
        include_extended : bool
            Whether to include v2 extended features (metabolic, clinical, etc.)
        verbose : bool
            Whether to print progress messages

        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered features
        """
        def log(msg):
            if verbose:
                print(msg)

        log("Applying improved feature engineering...")

        # High priority features (v1)
        df = ImprovedFeatureEngineer.add_blood_pressure_features(df)
        log("  [v1] Blood pressure features added")

        df = ImprovedFeatureEngineer.add_bmi_features(df)
        log("  [v1] BMI features added")

        df = ImprovedFeatureEngineer.add_age_features(df)
        log("  [v1] Age features added")

        df = ImprovedFeatureEngineer.add_composite_scores(df)
        log("  [v1] Composite scores added")

        # Experimental features (optional)
        if include_experimental:
            df = ImprovedFeatureEngineer.add_interaction_features(df)
            log("  [exp] Interaction features added")

            df = ImprovedFeatureEngineer.add_polynomial_features(df)
            log("  [exp] Polynomial features added")

        # Extended v2 features (for multi-dataset training)
        if include_extended:
            df = ImprovedFeatureEngineer.add_metabolic_features(df)
            log("  [v2] Metabolic features added")

            df = ImprovedFeatureEngineer.add_extended_bp_features(df)
            log("  [v2] Extended BP features added")

            df = ImprovedFeatureEngineer.add_clinical_features(df)
            log("  [v2] Clinical features added")

            df = ImprovedFeatureEngineer.add_modifiable_risk_features(df)
            log("  [v2] Modifiable risk features added")

            df = ImprovedFeatureEngineer.add_demographic_risk_features(df)
            log("  [v2] Demographic risk features added")

        return df

    @staticmethod
    def get_feature_columns(
        include_experimental: bool = False,
        include_extended: bool = False,
        remove_weight: bool = True,
        tier: str = 'basic'
    ) -> List[str]:
        """
        Get list of feature columns after engineering.

        Parameters:
        -----------
        include_experimental : bool
            Whether experimental features are included
        include_extended : bool
            Whether v2 extended features are included
        remove_weight : bool
            Whether to remove weight (redundant with BMI)
        tier : str
            Feature tier: 'basic' (original 18), 'extended' (with lab values),
            or 'full' (all available)

        Returns:
        --------
        list
            List of feature column names
        """
        # Base features (v1)
        features = [
            'age_years', 'gender', 'height', 'bmi',
            'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active'
        ]

        # Add weight if not removing
        if not remove_weight:
            features.insert(3, 'weight')

        # High priority engineered features (v1)
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

        # Extended v2 features
        if include_extended or tier in ['extended', 'full']:
            # Metabolic features
            features.extend([
                'total_hdl_ratio',
                'metabolic_syndrome_score',
                'diabetes_risk_indicator'
            ])

            # Extended BP features
            features.extend([
                'bp_index',
                'hypertension_crisis',
                'bp_control_status'
            ])

            # Modifiable risk features
            features.extend([
                'modifiable_risk_count',
                'modifiable_risk_score',
                'improvement_potential'
            ])

            # Demographic risk features
            features.extend([
                'cardiovascular_age_group',
                'gender_adjusted_age_risk'
            ])

        # Full tier includes clinical features (UCI-specific)
        if tier == 'full':
            features.extend([
                'hr_reserve_pct',
                'exercise_capacity_score',
                'ecg_abnormality',
                'ischemia_indicator'
            ])

        return features

    @staticmethod
    def get_feature_tiers() -> Dict[str, Dict]:
        """
        Get feature tier definitions for tiered prediction.

        Returns:
        --------
        dict
            Dictionary with tier definitions including required and optional features
        """
        return {
            'basic': {
                'description': 'Core features available from user input',
                'required': [
                    'age_years', 'gender', 'height', 'weight',
                    'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
                    'smoke', 'alco', 'active'
                ],
                'engineered': [
                    'bmi', 'pulse_pressure', 'mean_arterial_pressure',
                    'hypertension_stage', 'bmi_category', 'age_group',
                    'health_risk_composite', 'lifestyle_risk_score'
                ],
                'model_suffix': '_basic'
            },
            'extended': {
                'description': 'Enhanced features with lab values',
                'optional': [
                    'total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol',
                    'triglycerides', 'fasting_glucose', 'hba1c',
                    'bp_medication', 'diabetes', 'family_history_cvd'
                ],
                'engineered': [
                    'total_hdl_ratio', 'metabolic_syndrome_score',
                    'diabetes_risk_indicator', 'bp_index', 'bp_control_status',
                    'modifiable_risk_count', 'modifiable_risk_score'
                ],
                'model_suffix': '_extended'
            },
            'clinical': {
                'description': 'Clinical features from medical examinations',
                'optional': [
                    'chest_pain_type', 'resting_ecg', 'max_heart_rate',
                    'exercise_angina', 'st_depression', 'thalassemia'
                ],
                'engineered': [
                    'hr_reserve_pct', 'exercise_capacity_score',
                    'ecg_abnormality', 'ischemia_indicator'
                ],
                'model_suffix': '_clinical'
            }
        }


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
