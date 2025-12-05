"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional


class TumorFeatures(BaseModel):
    """
    Input schema for tumor features
    All features are derived from digitized images of fine needle aspirate (FNA) of breast mass
    """
    # Mean features
    radius_mean: float = Field(..., ge=0, description="Mean of distances from center to points on the perimeter")
    texture_mean: float = Field(..., ge=0, description="Standard deviation of gray-scale values")
    perimeter_mean: float = Field(..., ge=0, description="Mean perimeter of the tumor")
    area_mean: float = Field(..., ge=0, description="Mean area of the tumor")
    smoothness_mean: float = Field(..., ge=0, le=1, description="Mean local variation in radius lengths")
    compactness_mean: float = Field(..., ge=0, description="Mean of perimeter^2 / area - 1.0")
    concavity_mean: float = Field(..., ge=0, description="Mean severity of concave portions of the contour")
    concave_points_mean: float = Field(..., ge=0, description="Mean number of concave portions of the contour", alias="concave points_mean")
    symmetry_mean: float = Field(..., ge=0, le=1, description="Mean symmetry")
    fractal_dimension_mean: float = Field(..., ge=0, description="Mean 'coastline approximation' - 1")

    # Standard error features
    radius_se: float = Field(..., ge=0, description="Standard error of radius")
    texture_se: float = Field(..., ge=0, description="Standard error of texture")
    perimeter_se: float = Field(..., ge=0, description="Standard error of perimeter")
    area_se: float = Field(..., ge=0, description="Standard error of area")
    smoothness_se: float = Field(..., ge=0, description="Standard error of smoothness")
    compactness_se: float = Field(..., ge=0, description="Standard error of compactness")
    concavity_se: float = Field(..., ge=0, description="Standard error of concavity")
    concave_points_se: float = Field(..., ge=0, description="Standard error of concave points", alias="concave points_se")
    symmetry_se: float = Field(..., ge=0, description="Standard error of symmetry")
    fractal_dimension_se: float = Field(..., ge=0, description="Standard error of fractal dimension")

    # Worst features (mean of the three largest values)
    radius_worst: float = Field(..., ge=0, description="Worst radius")
    texture_worst: float = Field(..., ge=0, description="Worst texture")
    perimeter_worst: float = Field(..., ge=0, description="Worst perimeter")
    area_worst: float = Field(..., ge=0, description="Worst area")
    smoothness_worst: float = Field(..., ge=0, le=1, description="Worst smoothness")
    compactness_worst: float = Field(..., ge=0, description="Worst compactness")
    concavity_worst: float = Field(..., ge=0, description="Worst concavity")
    concave_points_worst: float = Field(..., ge=0, description="Worst concave points", alias="concave points_worst")
    symmetry_worst: float = Field(..., ge=0, le=1, description="Worst symmetry")
    fractal_dimension_worst: float = Field(..., ge=0, description="Worst fractal dimension")

    class Config:
        populate_by_name = True  # Allow both field name and alias
        json_schema_extra = {
            "example": {
                "radius_mean": 17.99,
                "texture_mean": 10.38,
                "perimeter_mean": 122.8,
                "area_mean": 1001.0,
                "smoothness_mean": 0.1184,
                "compactness_mean": 0.2776,
                "concavity_mean": 0.3001,
                "concave points_mean": 0.1471,
                "symmetry_mean": 0.2419,
                "fractal_dimension_mean": 0.07871,
                "radius_se": 1.095,
                "texture_se": 0.9053,
                "perimeter_se": 8.589,
                "area_se": 153.4,
                "smoothness_se": 0.006399,
                "compactness_se": 0.04904,
                "concavity_se": 0.05373,
                "concave points_se": 0.01587,
                "symmetry_se": 0.03003,
                "fractal_dimension_se": 0.006193,
                "radius_worst": 25.38,
                "texture_worst": 17.33,
                "perimeter_worst": 184.6,
                "area_worst": 2019.0,
                "smoothness_worst": 0.1622,
                "compactness_worst": 0.6656,
                "concavity_worst": 0.7119,
                "concave points_worst": 0.2654,
                "symmetry_worst": 0.4601,
                "fractal_dimension_worst": 0.1189
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for prediction
    """
    prediction: int = Field(
        ...,
        description="Prediction: 0=benign, 1=malignant"
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of malignancy (0-1)"
    )
    diagnosis: str = Field(
        ...,
        description="Diagnosis: benign or malignant"
    )
    confidence: str = Field(
        ...,
        description="Confidence level: low, medium, high"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.92,
                "diagnosis": "malignant",
                "confidence": "high"
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
