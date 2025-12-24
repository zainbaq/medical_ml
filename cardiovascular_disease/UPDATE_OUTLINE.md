# Cardiovascular Disease Prediction Model - Comprehensive Retraining Plan

## Executive Summary

Your current model achieves 72-75% accuracy with ROC-AUC of 0.78-0.82 using the Kaggle Cardiovascular Disease dataset. This plan will guide you through a complete overhaul targeting **80-85% accuracy and ROC-AUC > 0.82** through:

1. **Data Quality Improvements** - Cleaning the existing dataset and removing ~15-20% of invalid records
2. **Dataset Augmentation** - Adding UCI Heart Disease, NHANES, and Framingham teaching datasets
3. **Enhanced Feature Engineering** - Adding clinically-validated risk factors
4. **Model Architecture Upgrade** - Migrating from Random Forest to XGBoost with SHAP explainability
5. **Validation Framework** - Implementing robust cross-validation and demographic subgroup testing
6. **Ethical Safeguards** - Adding appropriate disclaimers and risk communication

---

## Phase 1: Data Acquisition and Integration (Week 1-2)

### 1.1 Current Dataset Issues

**Critical Problems Identified in cardio_train.csv:**

| Issue | Examples Found | Impact |
|-------|---------------|--------|
| Impossible BP values | ap_lo = 1000 (rows 69874, 69880, 69887, 69969) | Model learns from noise |
| Negative/zero values | Potential height/weight zeros | Division errors in BMI |
| BP logical errors | ap_lo > ap_hi in some records | Clinically impossible |
| Extreme outliers | BP > 300, weight > 250 | Skews distributions |

**Estimated Clean Dataset Size:** ~57,000-59,000 records (from 70,000)

### 1.2 Additional Datasets to Acquire

#### Dataset 1: UCI Heart Disease (PRIORITY: HIGH)
- **Size:** 920 records (combined) / 303 (Cleveland only)
- **Access:** Free, immediate download
- **License:** CC BY 4.0
- **Key Features Added:**
  - Chest pain type (4 categories)
  - Resting ECG results
  - Maximum heart rate achieved
  - Exercise-induced angina
  - ST depression (oldpeak)
  - Slope of peak exercise ST segment
  - Number of major vessels colored by fluoroscopy
  - Thalassemia status

**Download Method:**
```python
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets
```

#### Dataset 2: NHANES Cardiovascular Module (PRIORITY: HIGH)
- **Size:** 5,000-10,000 per cycle (use 2017-2020 pre-pandemic)
- **Access:** Free, no registration
- **URL:** https://wwwn.cdc.gov/nchs/nhanes/
- **Key Features Added:**
  - HDL Cholesterol (actual mg/dL values)
  - LDL Cholesterol
  - Total Cholesterol
  - Triglycerides
  - HbA1c
  - Fasting glucose
  - Blood pressure medication status
  - Race/ethnicity (5 categories)

**Required Files:**
- DEMO_J.XPT (Demographics)
- BPQ_J.XPT (Blood Pressure Questionnaire)
- BPX_J.XPT (Blood Pressure Examination)
- TCHOL_J.XPT (Total Cholesterol)
- HDL_J.XPT (HDL Cholesterol)
- TRIGLY_J.XPT (Triglycerides)
- GLU_J.XPT (Plasma Glucose)
- MCQ_J.XPT (Medical Conditions)
- SMQ_J.XPT (Smoking)

#### Dataset 3: Framingham Teaching Dataset (PRIORITY: MEDIUM)
- **Size:** 4,434 records
- **Access:** Free via Kaggle or MIT OpenCourseWare
- **License:** Educational use
- **Key Features Added:**
  - 10-year CHD outcome (validated longitudinal)
  - Education level
  - Cigarettes per day (not just binary)
  - BP medication status
  - Prevalent stroke history
  - Prevalent hypertension
  - Diabetes status

**Kaggle URL:** https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset

### 1.3 Unified Data Schema

Create a standardized schema that maps features across all datasets:

| Unified Feature | Kaggle | UCI | NHANES | Framingham |
|-----------------|--------|-----|--------|------------|
| age_years | age/365.25 | age | RIDAGEYR | age |
| gender | gender | sex | RIAGENDR | male |
| height_cm | height | — | BMXHT | — |
| weight_kg | weight | — | BMXWT | — |
| bmi | calculated | — | BMXBMI | BMI |
| systolic_bp | ap_hi | trestbps | BPXSY1 | sysBP |
| diastolic_bp | ap_lo | — | BPXDI1 | diaBP |
| total_chol | — | chol | LBXTC | totChol |
| hdl_chol | — | — | LBDHDD | — |
| cholesterol_cat | cholesterol | — | — | — |
| glucose_cat | gluc | fbs>120 | — | glucose |
| smoking | smoke | — | SMQ020 | currentSmoker |
| bp_medication | — | — | BPQ050A | BPMeds |
| diabetes | — | — | DIQ010 | diabetes |
| exercise | active | — | PAQ605 | — |
| chest_pain_type | — | cp | — | — |
| max_heart_rate | — | thalach | — | heartRate |
| cvd_outcome | cardio | num | MCQ160C | TenYearCHD |

---

## Phase 2: Data Cleaning and Preprocessing (Week 2-3)

### 2.1 Enhanced Cleaning Rules

```python
CLEANING_RULES = {
    # Blood Pressure
    'ap_hi': {'min': 70, 'max': 250, 'name': 'Systolic BP'},
    'ap_lo': {'min': 40, 'max': 150, 'name': 'Diastolic BP'},
    
    # Anthropometrics  
    'height': {'min': 100, 'max': 220, 'name': 'Height (cm)'},
    'weight': {'min': 30, 'max': 300, 'name': 'Weight (kg)'},
    'bmi': {'min': 12, 'max': 60, 'name': 'BMI'},
    
    # Age
    'age_years': {'min': 18, 'max': 100, 'name': 'Age'},
    
    # Logical constraints
    'bp_relationship': 'ap_hi > ap_lo + 10',  # Pulse pressure must be positive
    'bmi_consistency': 'weight / (height/100)^2 within 5% of stated BMI'
}
```

### 2.2 Missing Value Strategy

| Feature Type | Strategy | Rationale |
|--------------|----------|-----------|
| Continuous (BP, BMI) | Multiple Imputation (MICE) | Preserves variance, handles MAR |
| Categorical (smoking) | Mode imputation or "Unknown" category | Low missingness expected |
| Lab values (cholesterol) | Create "lab_available" indicator + impute | Missing may be informative |
| Binary (medication) | Assume "No" with uncertainty flag | Conservative approach |

### 2.3 Outlier Detection

Use Isolation Forest for multivariate outlier detection:
- Train on physiologically plausible ranges
- Flag records with contamination score > 0.1
- Manual review for borderline cases

---

## Phase 3: Enhanced Feature Engineering (Week 3-4)

### 3.1 New Clinical Features

#### Blood Pressure Features (PRIORITY: HIGH)
```python
# Pulse Pressure - arterial stiffness indicator
pulse_pressure = ap_hi - ap_lo

# Mean Arterial Pressure - overall perfusion pressure
mean_arterial_pressure = (ap_hi + 2 * ap_lo) / 3

# Hypertension Stage (ACC/AHA 2017 Guidelines)
# 0: Normal (<120/<80)
# 1: Elevated (120-129/<80)
# 2: Stage 1 (130-139/80-89)
# 3: Stage 2 (≥140/≥90)
# 4: Hypertensive Crisis (>180/>120)

# Blood Pressure Index (multiplicative risk)
bp_index = (ap_hi / 120) * (ap_lo / 80)
```

#### Metabolic Risk Features (PRIORITY: HIGH)
```python
# If HDL available (from NHANES/UCI):
total_hdl_ratio = total_chol / hdl_chol  # Strong CVD predictor

# Metabolic Syndrome Score (0-5 components)
metabolic_syndrome_score = sum([
    bmi >= 30,
    ap_hi >= 130 or ap_lo >= 85,
    hdl_chol < 40 if male else hdl_chol < 50,
    triglycerides >= 150,
    fasting_glucose >= 100
])

# Diabetes Risk Indicator
diabetes_risk = (glucose_cat >= 2) | (hba1c >= 5.7 if available)
```

#### Lifestyle Risk Features (PRIORITY: MEDIUM)
```python
# Lifestyle Risk Score (0-4)
lifestyle_risk = sum([
    smoke == 1,
    alco == 1,
    active == 0,
    bmi >= 30
])

# Modifiable Risk Count
modifiable_risk_count = sum([
    ap_hi >= 130,  # Controllable with medication
    cholesterol_cat >= 2,  # Controllable with diet/statins
    smoke == 1,  # Cessation possible
    active == 0  # Can increase activity
])
```

#### Age-Related Risk Features (PRIORITY: MEDIUM)
```python
# Cardiovascular Age Groups (risk stratification)
# 0: <40 (low baseline risk)
# 1: 40-54 (increasing risk)
# 2: 55-64 (moderate-high risk)
# 3: 65-74 (high risk)
# 4: ≥75 (very high risk)

# Gender-Adjusted Age Risk
# Women typically develop CVD 10 years later than men
adjusted_age_risk = age_years if gender == 2 else age_years - 10
```

### 3.2 Feature Selection Strategy

1. **Initial Selection:** Include all engineered features
2. **Correlation Filter:** Remove features with |r| > 0.90 mutual correlation
3. **Importance Ranking:** Use XGBoost feature importance + SHAP values
4. **Final Selection:** Keep top 20-25 features based on:
   - Predictive power (AUC contribution)
   - Clinical interpretability
   - User reportability (can users provide this data?)

### 3.3 Feature Categorization for UI

| Feature Category | Features | User Can Report? |
|------------------|----------|------------------|
| Basic Demographics | age, gender | ✅ Yes |
| Anthropometrics | height, weight, bmi | ✅ Yes |
| Blood Pressure | ap_hi, ap_lo (self-measured) | ⚠️ If they have BP monitor |
| Lifestyle | smoking, alcohol, exercise | ✅ Yes |
| Lab Values | cholesterol, glucose, HDL | ❌ Requires blood test |
| Medical History | diabetes, BP meds | ⚠️ If diagnosed |

---

## Phase 4: Model Architecture Upgrade (Week 4-5)

### 4.1 Model Selection: XGBoost

**Why XGBoost over Random Forest:**
- 2-5% higher AUC in recent benchmarks (94.34% accuracy reported)
- Native handling of missing values
- Built-in regularization (L1/L2)
- Better calibrated probabilities
- Faster training with GPU support

### 4.2 Hyperparameter Search Space

```python
XGBOOST_PARAMS = {
    # Tree Parameters
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    
    # Learning Parameters
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300, 500],
    
    # Regularization
    'reg_alpha': [0, 0.1, 1],  # L1
    'reg_lambda': [1, 2, 5],   # L2
    
    # Sampling
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    
    # Class Imbalance
    'scale_pos_weight': [1, sum(y==0)/sum(y==1)]
}
```

### 4.3 Training Strategy

```python
# Stratified K-Fold with Nested Cross-Validation
# Outer loop: Model evaluation (5 folds)
# Inner loop: Hyperparameter tuning (3 folds)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from optuna import create_study

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Use Optuna for Bayesian hyperparameter optimization
# More efficient than GridSearchCV for large search spaces
```

### 4.4 Ensemble Approach (Optional)

If single XGBoost doesn't meet targets, consider:

```python
# Voting Ensemble
ensemble = VotingClassifier([
    ('xgb', XGBClassifier(**best_xgb_params)),
    ('lgb', LGBMClassifier(**best_lgb_params)),
    ('rf', RandomForestClassifier(**best_rf_params))
], voting='soft', weights=[0.5, 0.3, 0.2])
```

### 4.5 Calibration

Ensure probability estimates are well-calibrated for risk communication:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    base_estimator=best_model,
    method='isotonic',  # or 'sigmoid'
    cv=5
)
```

---

## Phase 5: Explainability with SHAP (Week 5)

### 5.1 SHAP Integration

```python
import shap

# Create explainer
explainer = shap.TreeExplainer(model)

# Global feature importance
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Individual prediction explanation
def explain_prediction(patient_data):
    shap_values = explainer.shap_values(patient_data)
    return {
        'prediction': model.predict_proba(patient_data)[0][1],
        'top_risk_factors': get_top_factors(shap_values, feature_names, n=5),
        'top_protective_factors': get_top_factors(shap_values, feature_names, n=3, ascending=True)
    }
```

### 5.2 User-Facing Explanations

For each prediction, provide:

1. **Risk Level:** Low / Medium / High / Very High
2. **Probability Range:** "Your estimated 10-year CVD risk is 15-20%"
3. **Top 3 Contributing Factors:** "High blood pressure, Age over 55, Smoking"
4. **Modifiable Recommendations:** "Quitting smoking could reduce your risk by ~15%"

---

## Phase 6: Validation Framework (Week 5-6)

### 6.1 Validation Metrics

| Metric | Target | Current | Purpose |
|--------|--------|---------|---------|
| ROC-AUC | > 0.82 | 0.78-0.82 | Discrimination ability |
| Accuracy | > 78% | 72-75% | Overall correctness |
| Sensitivity | > 75% | ~68-72% | Catch true positives |
| Specificity | > 75% | ~70-73% | Avoid false alarms |
| Brier Score | < 0.18 | ~0.20 | Calibration quality |
| Precision | > 72% | ~70-73% | Positive predictive value |

### 6.2 Demographic Subgroup Validation

Test performance separately for:

| Subgroup | Minimum N | Acceptable AUC Drop |
|----------|-----------|---------------------|
| Male | 1000 | ≤ 0.02 |
| Female | 1000 | ≤ 0.02 |
| Age < 45 | 500 | ≤ 0.03 |
| Age 45-65 | 1000 | ≤ 0.02 |
| Age > 65 | 500 | ≤ 0.03 |
| BMI < 25 | 500 | ≤ 0.03 |
| BMI ≥ 30 | 500 | ≤ 0.03 |

### 6.3 External Validation

If possible, validate on a completely held-out dataset:
- Reserve 10% of NHANES data (never seen during training)
- Test on a different time period (e.g., train on 2017-2018, test on 2019-2020)

---

## Phase 7: API and UI Updates (Week 6-7)

### 7.1 Updated API Schema

```python
class PatientDataV2(BaseModel):
    # Required fields (same as current)
    age_years: float = Field(..., ge=18, le=100)
    gender: int = Field(..., ge=1, le=2)
    height: float = Field(..., ge=100, le=220)
    weight: float = Field(..., ge=30, le=300)
    ap_hi: int = Field(..., ge=70, le=250)
    ap_lo: int = Field(..., ge=40, le=150)
    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)
    
    # NEW: Optional enhanced fields
    hdl_cholesterol: Optional[float] = Field(None, ge=20, le=100)
    ldl_cholesterol: Optional[float] = Field(None, ge=40, le=300)
    total_cholesterol: Optional[float] = Field(None, ge=100, le=400)
    bp_medication: Optional[int] = Field(None, ge=0, le=1)
    diabetes: Optional[int] = Field(None, ge=0, le=1)
    family_history_cvd: Optional[int] = Field(None, ge=0, le=1)

class PredictionResponseV2(BaseModel):
    prediction: int
    probability: float
    probability_range: tuple[float, float]  # NEW: Confidence interval
    risk_level: str
    risk_category: str  # NEW: "Low (<5%)", "Borderline (5-7.5%)", etc.
    bmi: float
    
    # NEW: Explainability
    top_risk_factors: list[dict]
    top_protective_factors: list[dict]
    modifiable_recommendations: list[str]
    
    # NEW: Data quality indicators
    prediction_confidence: str  # "high" if all features provided
    missing_features: list[str]
    disclaimer: str
```

### 7.2 Tiered Risk Assessment

Implement two-tier predictions based on available data:

```python
class RiskAssessor:
    def assess(self, patient_data):
        if self.has_lab_values(patient_data):
            return self.enhanced_prediction(patient_data)
        else:
            return self.basic_prediction(patient_data)
    
    def basic_prediction(self, data):
        """Uses non-lab features only"""
        features = self.extract_basic_features(data)
        prob = self.basic_model.predict_proba(features)
        return {
            'probability': prob,
            'confidence': 'moderate',
            'note': 'Accuracy improves with lab values'
        }
    
    def enhanced_prediction(self, data):
        """Uses all available features including labs"""
        features = self.extract_all_features(data)
        prob = self.enhanced_model.predict_proba(features)
        return {
            'probability': prob,
            'confidence': 'high',
            'note': 'Full feature set used'
        }
```

---

## Phase 8: Ethical Safeguards and Disclaimers (Week 7)

### 8.1 Required Disclaimers

```python
DISCLAIMERS = {
    'general': """
        This tool provides an estimated cardiovascular disease risk assessment 
        based on the information you provided. It is NOT a medical diagnosis 
        and should not replace consultation with a healthcare professional.
    """,
    
    'diabetes_warning': """
        If you have diabetes, this tool may underestimate your cardiovascular 
        risk. Please consult your doctor for a comprehensive assessment.
    """,
    
    'emergency': """
        If you are experiencing chest pain, shortness of breath, or other 
        symptoms of a heart attack, call emergency services immediately.
    """,
    
    'data_limitation': """
        This assessment is based on statistical models trained on population 
        data. Individual risk may vary based on factors not captured here.
    """,
    
    'action_recommendation': """
        For moderate or high risk results, we recommend discussing these 
        results with your healthcare provider who can order appropriate 
        laboratory tests and provide personalized guidance.
    """
}
```

### 8.2 Risk Communication Framework

| Estimated Risk | Category | Color | Action Recommendation |
|----------------|----------|-------|----------------------|
| < 5% | Low | Green | Maintain healthy lifestyle |
| 5-7.5% | Borderline | Yellow | Consider lifestyle modifications |
| 7.5-20% | Intermediate | Orange | Discuss with healthcare provider |
| ≥ 20% | High | Red | Seek medical evaluation |

### 8.3 California AB 3030 Compliance

If serving California users, include:
- Clear statement that results are AI-generated
- Instructions for contacting human healthcare providers
- Option to request human review of results

---

## Phase 9: Implementation Timeline

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1 | Data Acquisition | Download UCI, NHANES, Framingham datasets |
| 2 | Data Cleaning | Clean Kaggle dataset, unified schema |
| 3 | Data Integration | Merge datasets, handle missing values |
| 4 | Feature Engineering | Implement all new features |
| 5 | Model Training | XGBoost with Optuna tuning, SHAP integration |
| 6 | Validation | Cross-validation, subgroup testing |
| 7 | API Updates | New endpoints, disclaimers, UI changes |
| 8 | Testing & Deployment | Load testing, A/B testing, production deploy |

---

## Phase 10: Success Metrics

### 10.1 Technical Metrics

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| ROC-AUC | 0.78-0.82 | > 0.82 | > 0.85 |
| Accuracy | 72-75% | > 78% | > 82% |
| Sensitivity | 68-72% | > 75% | > 80% |
| Brier Score | ~0.20 | < 0.18 | < 0.15 |

### 10.2 User Metrics

- Time to complete assessment: < 2 minutes
- User understanding of results: > 80% (survey)
- Follow-up action rate: > 30% for high-risk users

### 10.3 Ethical Metrics

- No demographic group with AUC < 0.75
- Calibration error < 5% across all risk strata
- 100% of predictions include appropriate disclaimers

---

## Appendix A: File Structure

```
cardiovascular_disease/
├── data/
│   ├── raw/
│   │   ├── cardio_train.csv          # Original Kaggle
│   │   ├── uci_heart_disease.csv     # UCI dataset
│   │   ├── nhanes_2017_2020.csv      # NHANES combined
│   │   └── framingham.csv            # Framingham teaching
│   ├── processed/
│   │   ├── cleaned_kaggle.csv
│   │   ├── unified_training.csv      # Combined & cleaned
│   │   └── feature_engineered.csv
│   └── external_validation/
│       └── holdout_nhanes.csv
├── training/
│   ├── data_acquisition/
│   │   ├── download_uci.py
│   │   ├── download_nhanes.py
│   │   └── download_framingham.py
│   ├── preprocessing/
│   │   ├── clean_kaggle.py
│   │   ├── unify_schema.py
│   │   └── impute_missing.py
│   ├── features/
│   │   ├── feature_engineering_v2.py
│   │   └── feature_selection.py
│   ├── models/
│   │   ├── train_xgboost.py
│   │   ├── hyperparameter_tuning.py
│   │   └── calibration.py
│   ├── evaluation/
│   │   ├── cross_validation.py
│   │   ├── subgroup_analysis.py
│   │   └── shap_analysis.py
│   └── config.py
├── backend/
│   ├── models/
│   │   ├── schemas_v2.py             # Updated Pydantic models
│   │   └── ml_model_v2.py
│   ├── routes/
│   │   └── predict_v2.py
│   └── utils/
│       ├── preprocessing_v2.py
│       ├── explainability.py
│       └── disclaimers.py
└── models/
    ├── xgboost_model_v2.pkl
    ├── scaler_v2.pkl
    ├── shap_explainer.pkl
    └── metadata_v2.json
```

---

## Appendix B: Quick Start Commands

```bash
# 1. Create new training environment
cd cardiovascular_disease
python -m venv venv_v2
source venv_v2/bin/activate
pip install -r training/requirements_v2.txt

# 2. Download additional datasets
python training/data_acquisition/download_all.py

# 3. Clean and preprocess
python training/preprocessing/run_pipeline.py

# 4. Train new model
python training/models/train_xgboost.py

# 5. Evaluate
python training/evaluation/full_evaluation.py

# 6. Start updated API
cd backend
uvicorn main:app --reload --port 8003
```

---

## Next Steps

1. **Immediate (This Week):** Run the data acquisition scripts
2. **Review:** Examine data quality report for each dataset
3. **Decision Point:** Choose whether to include lab-value features (requires two-tier model)
4. **Proceed:** Execute remaining phases according to timeline

Would you like me to generate the implementation code for any specific phase?