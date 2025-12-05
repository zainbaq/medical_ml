"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional


class PatientData(BaseModel):
    """
    Input schema for patient data
    """
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
        ge=140,
        le=210,
        description="Height in centimeters"
    )
    weight: float = Field(
        ...,
        ge=40,
        le=200,
        description="Weight in kilograms"
    )
    ap_hi: int = Field(
        ...,
        ge=80,
        le=200,
        description="Systolic blood pressure"
    )
    ap_lo: int = Field(
        ...,
        ge=60,
        le=130,
        description="Diastolic blood pressure"
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

    @validator('ap_lo')
    def validate_blood_pressure(cls, v, values):
        """Ensure diastolic BP is less than systolic BP"""
        if 'ap_hi' in values and v >= values['ap_hi']:
            raise ValueError('Diastolic BP must be less than systolic BP')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age_years": 55.0,
                "gender": 2,
                "height": 170.0,
                "weight": 80.0,
                "ap_hi": 130,
                "ap_lo": 85,
                "cholesterol": 2,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for prediction
    """
    prediction: int = Field(
        ...,
        description="Prediction: 0=no disease, 1=disease present"
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of cardiovascular disease (0-1)"
    )
    risk_level: str = Field(
        ...,
        description="Risk level: low, medium, high"
    )
    bmi: float = Field(
        ...,
        description="Calculated BMI"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.72,
                "risk_level": "high",
                "bmi": 27.68
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str
    service: str
    model_loaded: bool
    model_name: Optional[str] = None
    version: str


class ErrorResponse(BaseModel):
    """
    Error response schema
    """
    error: str
    detail: Optional[str] = None
