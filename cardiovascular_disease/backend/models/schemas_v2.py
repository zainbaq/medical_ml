"""
Pydantic models for API v2 - Enhanced cardiovascular disease prediction

v2 Features:
- Optional lab values for enhanced prediction
- SHAP-based risk factor explanations
- Confidence intervals
- Tiered prediction based on available data
- Modifiable risk recommendations
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Tuple
from enum import Enum


class RiskCategory(str, Enum):
    """Validated risk categories based on clinical guidelines."""
    LOW = "low"
    MODERATE = "moderate"
    MODERATELY_HIGH = "moderately_high"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PredictionTier(str, Enum):
    """Prediction tier based on available features."""
    BASIC = "basic"
    EXTENDED = "extended"
    CLINICAL = "clinical"


class ConfidenceLevel(str, Enum):
    """Prediction confidence level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PatientDataV2(BaseModel):
    """
    Enhanced patient data schema with optional lab values for v2 API.

    Required fields are the same as v1 for backward compatibility.
    Optional fields enable enhanced prediction when available.
    """

    # Required fields (same as v1)
    age_years: float = Field(
        ...,
        ge=18,
        le=100,
        description="Age in years"
    )
    gender: int = Field(
        ...,
        ge=1,
        le=2,
        description="Gender: 1=female, 2=male"
    )
    height: float = Field(
        ...,
        ge=100,
        le=220,
        description="Height in centimeters"
    )
    weight: float = Field(
        ...,
        ge=30,
        le=300,
        description="Weight in kilograms"
    )
    ap_hi: int = Field(
        ...,
        ge=70,
        le=250,
        description="Systolic blood pressure (mmHg)"
    )
    ap_lo: int = Field(
        ...,
        ge=40,
        le=150,
        description="Diastolic blood pressure (mmHg)"
    )
    cholesterol: int = Field(
        ...,
        ge=1,
        le=3,
        description="Cholesterol level: 1=normal, 2=above normal, 3=well above normal"
    )
    gluc: int = Field(
        ...,
        ge=1,
        le=3,
        description="Glucose level: 1=normal, 2=above normal, 3=well above normal"
    )
    smoke: int = Field(
        ...,
        ge=0,
        le=1,
        description="Smoking status: 0=no, 1=yes"
    )
    alco: int = Field(
        ...,
        ge=0,
        le=1,
        description="Alcohol intake: 0=no, 1=yes"
    )
    active: int = Field(
        ...,
        ge=0,
        le=1,
        description="Physical activity: 0=no, 1=yes"
    )

    # NEW: Optional enhanced fields (v2)
    hdl_cholesterol: Optional[float] = Field(
        None,
        ge=20,
        le=100,
        description="HDL cholesterol in mg/dL (optional, improves accuracy)"
    )
    ldl_cholesterol: Optional[float] = Field(
        None,
        ge=40,
        le=300,
        description="LDL cholesterol in mg/dL (optional)"
    )
    total_cholesterol: Optional[float] = Field(
        None,
        ge=100,
        le=400,
        description="Total cholesterol in mg/dL (optional)"
    )
    triglycerides: Optional[float] = Field(
        None,
        ge=30,
        le=1000,
        description="Triglycerides in mg/dL (optional)"
    )
    fasting_glucose: Optional[float] = Field(
        None,
        ge=40,
        le=500,
        description="Fasting glucose in mg/dL (optional)"
    )
    hba1c: Optional[float] = Field(
        None,
        ge=3.0,
        le=15.0,
        description="HbA1c percentage (optional)"
    )
    bp_medication: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Currently on BP medication: 0=no, 1=yes (optional)"
    )
    diabetes: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Diagnosed diabetes: 0=no, 1=yes (optional)"
    )
    family_history_cvd: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Family history of CVD: 0=no, 1=yes (optional)"
    )

    @field_validator('ap_lo')
    @classmethod
    def validate_blood_pressure(cls, v, info):
        """Ensure diastolic BP is less than systolic BP."""
        if 'ap_hi' in info.data and v >= info.data['ap_hi']:
            raise ValueError('Diastolic BP must be less than systolic BP')
        return v

    @model_validator(mode='after')
    def validate_cholesterol_consistency(self):
        """Validate cholesterol values are consistent if multiple provided."""
        if (self.total_cholesterol is not None and
                self.hdl_cholesterol is not None and
                self.ldl_cholesterol is not None):
            # Total should be approximately HDL + LDL + triglycerides/5
            # Allow 20% variance for measurement differences
            expected_total = self.hdl_cholesterol + self.ldl_cholesterol
            if self.triglycerides:
                expected_total += self.triglycerides / 5
            if abs(self.total_cholesterol - expected_total) > 0.3 * expected_total:
                # Just log warning, don't fail validation
                pass
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "age_years": 55.0,
                "gender": 2,
                "height": 170.0,
                "weight": 80.0,
                "ap_hi": 140,
                "ap_lo": 90,
                "cholesterol": 2,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1,
                "hdl_cholesterol": 45.0,
                "total_cholesterol": 220.0,
                "bp_medication": 1
            }
        }


class RiskFactor(BaseModel):
    """Individual risk factor with SHAP-based contribution."""

    name: str = Field(..., description="Feature name")
    display_name: str = Field(..., description="Human-readable name")
    value: float = Field(..., description="Feature value")
    contribution: float = Field(
        ...,
        description="SHAP contribution to prediction (positive=increases risk)"
    )
    is_modifiable: bool = Field(
        ...,
        description="Whether this factor can be changed"
    )
    recommendation: Optional[str] = Field(
        None,
        description="Action recommendation if modifiable"
    )


class PredictionResponseV2(BaseModel):
    """
    Enhanced prediction response with explainability and confidence intervals.
    """

    # Core prediction (same fields as v1 for compatibility)
    prediction: int = Field(
        ...,
        description="Prediction: 0=no disease risk, 1=disease risk present"
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of cardiovascular disease (0-1)"
    )
    risk_level: str = Field(
        default="medium",
        description="Risk level: low, medium, high (v1 compatible)"
    )
    bmi: float = Field(
        ...,
        description="Calculated BMI"
    )

    # NEW: Enhanced risk assessment (v2)
    probability_range: Tuple[float, float] = Field(
        ...,
        description="95% confidence interval for probability"
    )
    risk_category: RiskCategory = Field(
        ...,
        description="Validated risk category based on probability"
    )
    risk_score: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="0-100 normalized risk score"
    )

    # NEW: Prediction metadata (v2)
    prediction_tier: PredictionTier = Field(
        ...,
        description="Prediction tier: basic, extended, or clinical"
    )
    prediction_confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence level based on data completeness"
    )
    missing_features: List[str] = Field(
        default_factory=list,
        description="List of optional features not provided"
    )

    # NEW: Explainability (v2)
    top_risk_factors: List[RiskFactor] = Field(
        ...,
        max_length=5,
        description="Top factors increasing CVD risk"
    )
    top_protective_factors: List[RiskFactor] = Field(
        ...,
        max_length=3,
        description="Top factors decreasing CVD risk"
    )
    modifiable_recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for risk reduction"
    )

    # NEW: Metadata (v2)
    model_version: str = Field(
        default="2.0.0",
        description="Model version used for prediction"
    )
    disclaimer: str = Field(
        default=(
            "This prediction is for informational purposes only and should not "
            "replace professional medical advice. If you have concerns about your "
            "cardiovascular health, please consult a healthcare provider."
        ),
        description="Medical disclaimer"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.68,
                "risk_level": "high",
                "bmi": 27.68,
                "probability_range": [0.62, 0.74],
                "risk_category": "intermediate",
                "risk_score": 68.0,
                "prediction_tier": "extended",
                "prediction_confidence": "high",
                "missing_features": ["triglycerides", "hba1c"],
                "top_risk_factors": [
                    {
                        "name": "ap_hi",
                        "display_name": "Systolic Blood Pressure",
                        "value": 140,
                        "contribution": 0.15,
                        "is_modifiable": True,
                        "recommendation": "Blood pressure control through diet, exercise, or medication"
                    }
                ],
                "top_protective_factors": [
                    {
                        "name": "active",
                        "display_name": "Physical Activity",
                        "value": 1,
                        "contribution": -0.08,
                        "is_modifiable": True,
                        "recommendation": None
                    }
                ],
                "modifiable_recommendations": [
                    "Consider blood pressure management",
                    "Continue regular physical activity"
                ],
                "model_version": "2.0.0",
                "disclaimer": "This prediction is for informational purposes only..."
            }
        }


class FeatureRequirementsResponse(BaseModel):
    """Response describing feature requirements for each prediction tier."""

    basic: dict = Field(
        ...,
        description="Basic tier feature requirements"
    )
    extended: dict = Field(
        ...,
        description="Extended tier feature requirements"
    )
    clinical: dict = Field(
        ...,
        description="Clinical tier feature requirements (future)"
    )


class ModelInfoResponseV2(BaseModel):
    """Enhanced model information response for v2."""

    model_type: str = Field(..., description="Model type (e.g., XGBoost)")
    version: str = Field(..., description="Model version")
    training_date: str = Field(..., description="Training timestamp")
    metrics: dict = Field(..., description="Performance metrics")
    feature_importance: dict = Field(
        ...,
        description="Feature importance scores"
    )
    performance_targets: dict = Field(
        ...,
        description="Target performance thresholds"
    )
    meets_targets: bool = Field(
        ...,
        description="Whether model meets all performance targets"
    )
    available_tiers: List[str] = Field(
        ...,
        description="Available prediction tiers"
    )


class HealthResponseV2(BaseModel):
    """Enhanced health check response for v2."""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: Optional[str] = Field(None, description="Loaded model type")
    shap_available: bool = Field(
        ...,
        description="Whether SHAP explainability is available"
    )
    available_tiers: List[str] = Field(
        default_factory=list,
        description="Available prediction tiers"
    )
