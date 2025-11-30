# Breast Cancer Prediction System

Machine learning system for predicting breast cancer diagnosis (benign/malignant) using tumor features from fine needle aspirate (FNA) imaging.

## Dataset

**Wisconsin Breast Cancer Dataset**
- 569 samples
- 30 numerical features derived from digitized images
- Binary classification: Benign (0) or Malignant (1)
- Features include radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension
- Each feature has three measurements: mean, standard error, and worst (largest)

## Project Structure

```
breast_cancer/
├── data/                          # Dataset
│   └── breast_cancer_wisconsin.csv
├── models/                        # Trained models
│   └── [Generated after training]
├── training/                      # Training pipeline
│   ├── config.py                 # Training configuration
│   └── train_model.py            # Model training script
├── backend/                       # FastAPI backend
│   ├── main.py                   # FastAPI application
│   ├── config.py                 # Backend configuration
│   ├── models/                   # Data models
│   │   ├── schemas.py           # Pydantic schemas
│   │   └── ml_model.py          # ML model loader
│   ├── routes/                   # API routes
│   │   └── predict.py           # Prediction endpoints
│   ├── utils/                    # Utilities
│   │   └── preprocessing.py     # Feature preprocessing
│   └── requirements.txt          # Python dependencies
├── train.sh                       # Training script
├── start_api.sh                  # API startup script
├── test_api.py                   # Basic API tests
└── README.md                     # This file
```

## Quick Start

### 1. Train the Model

```bash
cd breast_cancer
./train.sh
```

This will:
- Create a virtual environment
- Install dependencies
- Train multiple models (SVM, Random Forest, Gradient Boosting, Logistic Regression)
- Select the best model based on ROC-AUC
- Save the model, scaler, and metadata

Expected performance: ~96-98% accuracy, ROC-AUC > 0.98

### 2. Start the API

```bash
./start_api.sh
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### 3. Test the API

```bash
python test_api.py
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Prediction
```bash
POST /api/v1/predict
```

**Request Body** (all 30 features required):
```json
{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "perimeter_mean": 122.8,
  "area_mean": 1001.0,
  "smoothness_mean": 0.1184,
  "compactness_mean": 0.2776,
  "concavity_mean": 0.3001,
  "concave_points_mean": 0.1471,
  "symmetry_mean": 0.2419,
  "fractal_dimension_mean": 0.07871,
  "radius_se": 1.095,
  "texture_se": 0.9053,
  "perimeter_se": 8.589,
  "area_se": 153.4,
  "smoothness_se": 0.006399,
  "compactness_se": 0.04904,
  "concavity_se": 0.05373,
  "concave_points_se": 0.01587,
  "symmetry_se": 0.03003,
  "fractal_dimension_se": 0.006193,
  "radius_worst": 25.38,
  "texture_worst": 17.33,
  "perimeter_worst": 184.6,
  "area_worst": 2019.0,
  "smoothness_worst": 0.1622,
  "compactness_worst": 0.6656,
  "concavity_worst": 0.7119,
  "concave_points_worst": 0.2654,
  "symmetry_worst": 0.4601,
  "fractal_dimension_worst": 0.1189
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.9523,
  "diagnosis": "malignant",
  "confidence": "high"
}
```

### Model Info
```bash
GET /api/v1/model-info
```

## Features

### Input Features (30 total)

#### Mean Features (10)
- `radius_mean`: Mean of distances from center to points on perimeter
- `texture_mean`: Standard deviation of gray-scale values
- `perimeter_mean`: Mean perimeter
- `area_mean`: Mean area
- `smoothness_mean`: Mean local variation in radius lengths
- `compactness_mean`: Mean of perimeter² / area - 1.0
- `concavity_mean`: Mean severity of concave portions of contour
- `concave_points_mean`: Mean number of concave portions
- `symmetry_mean`: Mean symmetry
- `fractal_dimension_mean`: Mean "coastline approximation" - 1

#### Standard Error Features (10)
- SE of each of the above 10 measurements
- Named as: `radius_se`, `texture_se`, etc.

#### Worst Features (10)
- Worst (largest) values of each of the 10 measurements
- Named as: `radius_worst`, `texture_worst`, etc.

## Model Training

The training pipeline:
1. Loads and preprocesses data from CSV
2. Converts diagnosis to binary (M=1, B=0)
3. Splits data (80% train, 20% test)
4. Scales features using StandardScaler
5. Trains 4 different models with GridSearchCV:
   - Support Vector Machine (SVM)
   - Random Forest
   - Gradient Boosting
   - Logistic Regression
6. Evaluates using ROC-AUC, accuracy, precision, recall, F1
7. Selects and saves the best model

## API Response

### Prediction Values
- `prediction`: 0 (benign) or 1 (malignant)
- `probability`: Probability of malignancy (0-1)
- `diagnosis`: "benign" or "malignant"
- `confidence`: "low", "medium", or "high"

### Confidence Levels
- **High**: Probability < 0.1 or > 0.9 (very certain)
- **Medium**: Probability 0.3-0.7 or 0.7-0.3
- **Low**: Probability close to 0.5 (uncertain)

## Development

### Manual Training
```bash
cd training
source venv/bin/activate
python train_model.py
```

### Manual API Start
```bash
cd breast_cancer
source backend/venv/bin/activate
uvicorn backend.main:app --reload
```

## Requirements

- Python 3.8+
- Dependencies (automatically installed):
  - fastapi
  - uvicorn
  - scikit-learn
  - pandas
  - numpy
  - pydantic
  - joblib

## Performance Metrics

Expected model performance:
- **Accuracy**: 96-98%
- **ROC-AUC**: > 0.98
- **Precision**: > 0.95
- **Recall**: > 0.95

## Notes

- The model is trained on the Wisconsin Breast Cancer dataset
- All 30 features must be provided for prediction
- Features are automatically scaled using the saved scaler
- The API loads the latest trained model on startup
- Model performance is excellent due to the well-separated nature of the dataset

## Citation

If using the Wisconsin Breast Cancer dataset, please cite:
- W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993.
