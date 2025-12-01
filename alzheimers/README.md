# Alzheimer's Disease Prediction

Machine learning system for predicting Alzheimer's disease risk based on cognitive assessments and brain imaging metrics.

## Dataset

**OASIS Longitudinal Dataset** - Alzheimer's Disease Detection
- 334 clinical assessments from longitudinal study (after preprocessing)
- Features: cognitive tests (MMSE, CDR), brain volume metrics, demographics
- Source: https://www.kaggle.com/hyunseokc/detecting-early-alzheimer-s

## Model Performance

Achieved on OASIS dataset:
- **Best Model**: SVM (RBF kernel)
- **Accuracy**: 100%
- **ROC-AUC**: 1.0000
- **Training Date**: 2025-11-30

All models (SVM, Random Forest, Logistic Regression) achieved perfect or near-perfect performance on the test set.

## Features

### Input Features (9 total)

1. **Age**: Patient age (55-100 years)
2. **Gender**: M (male) or F (female)
3. **EDUC**: Years of formal education (0-30)
4. **SES**: Socioeconomic status (1-5, optional - median used if not provided)
   - 1 = Highest status
   - 5 = Lowest status
5. **MMSE**: Mini Mental State Examination score (0-30)
   - Cognitive assessment tool
   - Higher scores indicate better cognitive function
6. **CDR**: Clinical Dementia Rating (0, 0.5, 1, 2, 3)
   - 0 = No dementia
   - 0.5 = Questionable dementia
   - 1 = Mild dementia
   - 2 = Moderate dementia
   - 3 = Severe dementia
7. **eTIV**: Estimated Total Intracranial Volume (900-2200 cm³)
8. **nWBV**: Normalized Whole Brain Volume (0.6-0.9)
   - Brain volume normalized by intracranial volume
9. **ASF**: Atlas Scaling Factor (0.8-1.6)

### Output

- **Prediction**: 0 (Non-demented) or 1 (Demented)
- **Probability**: Risk probability (0.0-1.0)
- **Risk Score**: Percentage score (0-100)
- **Stage**: Dementia stage based on CDR input (none/questionable/mild/moderate/severe)
- **Risk Level**: Categorical risk (low/medium/high)
  - Low: probability < 0.33
  - Medium: 0.33 ≤ probability < 0.66
  - High: probability ≥ 0.66

## Setup & Training

### Prerequisites

- Python 3.8+
- Virtual environment support

### Train the Model

```bash
chmod +x train.sh
./train.sh
```

This will:
- Create training virtual environment
- Install dependencies (pandas, numpy, scikit-learn, joblib)
- Load and preprocess OASIS dataset
- Train multiple models (SVM, Random Forest, Gradient Boosting, Logistic Regression)
- Select best model based on ROC-AUC
- Save model with versioning to `models/` directory
- Output median SES value (2.0)

Training takes approximately 2-5 minutes depending on your machine.

## API Usage

### Start the API Server

```bash
chmod +x start_api.sh
./start_api.sh
```

The API will be available at: **http://localhost:8001**

Interactive API docs: **http://localhost:8001/docs**

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-30T19:08:46.123456"
}
```

#### 2. Make Prediction

```bash
curl -X POST http://localhost:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 75,
    "gender": "M",
    "EDUC": 14,
    "SES": 2,
    "MMSE": 28,
    "CDR": 0.0,
    "eTIV": 1500,
    "nWBV": 0.75,
    "ASF": 1.2
  }'
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.05,
  "risk_score": 5.0,
  "stage": "none",
  "risk_level": "low"
}
```

#### 3. Get Model Info

```bash
curl http://localhost:8001/api/v1/model-info
```

Response:
```json
{
  "model_name": "svm",
  "timestamp": "20251130_190846",
  "metrics": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "roc_auc": 1.0
  },
  "feature_names": ["Age", "Gender", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
}
```

### Test the API

```bash
python test_api.py
```

This runs a comprehensive test suite including:
- Health endpoint test
- Prediction tests (non-demented, demented, moderate cases)
- Model info endpoint test
- Validation tests (invalid inputs)

## Project Structure

```
alzheimers/
├── data/
│   └── oasis_longitudinal.csv      # OASIS dataset
├── models/
│   ├── alzheimer_model_*.pkl       # Trained models
│   ├── scaler_*.pkl                # Feature scalers
│   ├── metadata_*.json             # Model metadata
│   └── latest_model_info.json      # Pointer to active model
├── training/
│   ├── config.py                   # Training configuration
│   ├── train_model.py              # Training pipeline
│   ├── requirements.txt            # Training dependencies
│   └── venv/                       # Training virtual env
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── config.py                   # Backend configuration
│   ├── models/
│   │   ├── schemas.py              # Pydantic request/response models
│   │   ├── ml_model.py             # Model loader class
│   │   └── __init__.py
│   ├── routes/
│   │   ├── predict.py              # Prediction endpoints
│   │   └── __init__.py
│   ├── utils/
│   │   ├── preprocessing.py        # Feature preparation
│   │   └── __init__.py
│   ├── requirements.txt            # Backend dependencies
│   └── venv/                       # Backend virtual env
├── train.sh                        # Training automation
├── start_api.sh                    # API startup
├── test_api.py                     # API tests
└── README.md                       # This file
```

## Architecture

### Training Pipeline

1. **Data Loading**: Read OASIS longitudinal CSV
2. **Preprocessing**:
   - Drop identifier columns (Subject ID, MRI ID, etc.)
   - Encode gender (M=1, F=0)
   - Encode target (Demented=1, Nondemented=0)
   - Handle missing SES with median imputation (2.0)
   - Drop rows with missing MMSE (critical feature)
3. **Feature Engineering**: Extract 9 core features
4. **Model Training**: GridSearchCV with 5-fold CV for 4 model types
5. **Model Selection**: Best model by ROC-AUC score
6. **Model Persistence**: Save with timestamp versioning

### Backend API

1. **Startup**: Load latest model via `latest_model_info.json`
2. **Request**: Validate input with Pydantic schemas
3. **Preprocessing**: Convert request to feature array (exact training order)
4. **Prediction**: Scale features → predict → calculate probability
5. **Post-processing**: Compute risk score, stage, risk level
6. **Response**: Return structured JSON response

## Important Notes

### Data Handling

- **SES (Socioeconomic Status)**: Optional in API - median value (2.0) used if missing
- **CDR Validation**: Only accepts 0, 0.5, 1, 2, or 3
- **Gender Encoding**: M=1, F=0 (consistent across training and inference)
- **Feature Order**: CRITICAL - must match training exactly:
  - Age, Gender, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF

### API Configuration

- **Port**: 8001 (different from CVD:8000, breast cancer:8002)
- **CORS**: Enabled for all origins
- **Validation**: Strict field validation with clear error messages
- **Logging**: Comprehensive logging for debugging

### Model Versioning

- Models saved with timestamps: `alzheimer_model_YYYYMMDD_HHMMSS.pkl`
- Metadata includes metrics, hyperparameters, feature names
- `latest_model_info.json` points to active model
- Easy to rollback or compare models

## Clinical Context

### MMSE (Mini Mental State Examination)
- Standard cognitive assessment tool
- 30-point questionnaire
- Scores < 24 suggest cognitive impairment
- Reference: Folstein et al. (1975)

### CDR (Clinical Dementia Rating)
- Assesses dementia severity
- Based on memory, orientation, judgment, community affairs, home/hobbies, personal care
- Reference: Morris (1993)

### Brain Volume Metrics
- **eTIV**: Total intracranial volume (head size)
- **nWBV**: Brain volume relative to head size
- **ASF**: Scaling factor for atlas normalization
- Lower nWBV associated with brain atrophy in Alzheimer's

## Troubleshooting

### Model Not Loading
- Ensure training has been completed: `./train.sh`
- Check `models/latest_model_info.json` exists
- Verify model files exist in `models/` directory

### API Not Starting
- Check port 8001 is not in use: `lsof -i :8001`
- Verify backend dependencies installed
- Check logs for errors

### Prediction Errors
- Validate input against schema in `/docs`
- Ensure CDR is one of: 0, 0.5, 1, 2, 3
- Ensure gender is 'M' or 'F'
- Check feature value ranges

## References

- **Dataset**: OASIS Longitudinal - https://www.kaggle.com/hyunseokc/detecting-early-alzheimer-s
- **MMSE**: Folstein, M. F., et al. (1975). "Mini-mental state". Journal of psychiatric research.
- **CDR**: Morris, J. C. (1993). "The Clinical Dementia Rating (CDR)". Neurology.
- **OASIS Project**: https://www.oasis-brains.org/

## License

This project is for educational and research purposes. Please ensure compliance with data usage policies and medical regulations when deploying in clinical settings.

## Related Projects

This implementation follows the same architecture as:
- **Cardiovascular Disease Prediction** (port 8000)
- **Breast Cancer Prediction** (port 8002)

All three projects share consistent patterns for easy maintenance and deployment.
