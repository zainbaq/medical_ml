"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class CognitiveAssessment(BaseModel):
    """Input schema for patient cognitive assessment data"""

    age: int = Field(
        ...,
        ge=55,
        le=100,
        description="Patient age in years",
        examples=[75]
    )

    gender: str = Field(
        ...,
        description="Patient gender: M (male) or F (female)",
        examples=["M"]
    )

    education_years: int = Field(
        ...,
        ge=0,
        le=30,
        alias="EDUC",
        description="Years of formal education",
        examples=[14]
    )

    socioeconomic_status: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        alias="SES",
        description="Socioeconomic status (1=highest, 5=lowest). Optional - median will be used if not provided.",
        examples=[2]
    )

    mmse_score: int = Field(
        ...,
        ge=0,
        le=30,
        alias="MMSE",
        description="Mini Mental State Examination score (0-30)",
        examples=[28]
    )

    cdr_score: float = Field(
        ...,
        alias="CDR",
        description="Clinical Dementia Rating (0, 0.5, 1, 2, or 3)",
        examples=[0.0]
    )

    estimated_total_intracranial_volume: float = Field(
        ...,
        ge=900,
        le=2200,
        alias="eTIV",
        description="Estimated total intracranial volume in cm³",
        examples=[1500]
    )

    normalized_whole_brain_volume: float = Field(
        ...,
        ge=0.6,
        le=0.9,
        alias="nWBV",
        description="Normalized whole brain volume (0-1 range)",
        examples=[0.75]
    )

    atlas_scaling_factor: float = Field(
        ...,
        ge=0.8,
        le=1.6,
        alias="ASF",
        description="Atlas scaling factor",
        examples=[1.2]
    )

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v.upper() not in ['M', 'F']:
            raise ValueError("Gender must be 'M' or 'F'")
        return v.upper()

    @field_validator('cdr_score')
    @classmethod
    def validate_cdr(cls, v):
        valid_values = [0.0, 0.5, 1.0, 2.0, 3.0]
        if v not in valid_values:
            raise ValueError(f"CDR must be one of {valid_values}")
        return v

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [{
                "age": 75,
                "gender": "M",
                "EDUC": 14,
                "SES": 2,
                "MMSE": 28,
                "CDR": 0.0,
                "eTIV": 1500,
                "nWBV": 0.75,
                "ASF": 1.2
            }]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for Alzheimer's prediction"""

    prediction: int = Field(
        ...,
        description="Binary prediction: 0=Non-demented, 1=Demented"
    )

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of dementia (0.0 to 1.0)"
    )

    risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Risk score (probability × 100)"
    )

    stage: str = Field(
        ...,
        description="Dementia stage based on CDR: none, questionable, mild, moderate, severe"
    )

    risk_level: str = Field(
        ...,
        description="Risk category: low, medium, high"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "prediction": 0,
                "probability": 0.15,
                "risk_score": 15.0,
                "stage": "none",
                "risk_level": "low"
            }]
        }
    }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    model_loaded: bool
    model_name: Optional[str] = None
    version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    timestamp: str
    metrics: dict
    feature_names: list
