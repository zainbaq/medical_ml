# Cardiovascular Disease Prediction API

A production-ready FastAPI backend for predicting cardiovascular disease risk using machine learning.

## Features

- **ML Model Training**: Comprehensive training pipeline with multiple models (Random Forest, Gradient Boosting, Logistic Regression)
- **Automatic Model Selection**: Selects the best performing model based on ROC-AUC score
- **Feature Engineering**: Automatic BMI calculation and data preprocessing
- **Input Validation**: Robust validation using Pydantic models
- **Risk Assessment**: Provides risk levels (low, medium, high) based on prediction probability
- **RESTful API**: FastAPI-based endpoints with automatic documentation

## Training Data Sources

The model is trained on a harmonized combination of cardiovascular disease datasets (~80K total records):

| Dataset | Records | Weight | Description |
|---------|---------|--------|-------------|
| **Kaggle** | ~70,000 | 1.0 | Original cardiovascular dataset |
| **NHANES 2017-2020** | ~5,800 | 1.2 | CDC population health data with lab values |
| **Framingham** | ~4,000 | 1.3 | Validated longitudinal heart study |

### Excluded Datasets

| Dataset | Records | Reason |
|---------|---------|--------|
| **UCI Heart Disease** | 673 | Missing anthropometric features (height, weight, lifestyle data) |

### Data Transparency

- All training data uses **real measured patient values**
- API responses include `training_data_sources` field showing which datasets were used
- Model metadata includes complete data provenance information

### Features Used

**Core Features (from all datasets):**
- Age (years)
- Gender (1=female, 2=male)
- BMI (calculated from height/weight)
- Systolic and Diastolic Blood Pressure
- Cholesterol level (1=normal, 2=above normal, 3=well above normal)
- Glucose level (1=normal, 2=above normal, 3=well above normal)
- Smoking status
- Alcohol consumption
- Physical activity

**Extended Features (v2 API, when available):**
- HDL/LDL cholesterol, triglycerides
- Fasting glucose, HbA1c
- BP medication status, diabetes status

> **Note:** Height and weight are collected from users but converted to BMI for model input. This enables multi-dataset training with consistent features.

## Project Structure

```
cardiovascular_disease/
├── backend/                 # FastAPI application
│   ├── main.py             # Main application
│   ├── config.py           # Configuration
│   ├── models/
│   │   ├── schemas.py      # Pydantic models
│   │   └── ml_model.py     # Model loader
│   ├── routes/
│   │   └── predict.py      # Prediction endpoints
│   ├── utils/
│   │   └── preprocessing.py # Utilities
│   └── requirements.txt
├── training/               # Model training
│   ├── train_model.py      # Training pipeline
│   ├── config.py           # Training config
│   └── requirements.txt
├── models/                 # Saved models
├── data/                   # Dataset
└── README.md
```

## Getting Started

### 1. Train the Model

First, install training dependencies and train the model:

```bash
cd training
pip install -r requirements.txt
python train_model.py
```

This will:
- Load and preprocess the data
- Train multiple ML models with hyperparameter tuning
- Select the best performing model
- Save the model, scaler, and metadata to the `models/` directory

### 2. Start the API Server

Install backend dependencies and start the server:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /health
```

Check API and model status.

### Predict Cardiovascular Disease
```
POST /api/v1/predict
```

Make a prediction for a patient.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.72,
  "risk_level": "high",
  "bmi": 27.68
}
```

### Model Information
```
GET /api/v1/model-info
```

Get information about the loaded model including metrics and parameters.

## API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Example Usage

### Using cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Using Python

```python
import requests

url = "http://localhost:8000/api/v1/predict"
patient_data = {
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

response = requests.post(url, json=patient_data)
result = response.json()
print(result)
```

## Model Performance

The training pipeline evaluates multiple models and selects the best one based on cross-validated ROC-AUC scores. Typical performance metrics:

- **Accuracy**: ~72-75%
- **Precision**: ~70-73%
- **Recall**: ~68-72%
- **ROC-AUC**: ~0.78-0.82

## Input Validation

The API includes comprehensive input validation:

- Age: 18-100 years
- Height: 140-210 cm
- Weight: 40-200 kg
- Blood Pressure: Systolic (80-200), Diastolic (60-130)
- Diastolic BP must be less than Systolic BP
- Categorical features: Valid range checks

## Risk Levels

Predictions include risk level classification:

- **Low Risk**: Probability < 0.33
- **Medium Risk**: Probability 0.33-0.66
- **High Risk**: Probability > 0.66

## Error Handling

The API includes proper error handling for:

- Invalid input data
- Model not loaded
- Server errors
- Validation errors

## License

This project is for educational and research purposes.
