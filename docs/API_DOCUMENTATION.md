# Medical ML Service Registry - API Documentation

Complete API reference for building frontend applications that interact with the Medical ML prediction services.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Service Registry API](#service-registry-api)
3. [Cardiovascular Disease API](#cardiovascular-disease-api)
4. [Breast Cancer API](#breast-cancer-api)
5. [Alzheimer's Disease API](#alzheimers-disease-api)
6. [Frontend Integration Guide](#frontend-integration-guide)
7. [Error Handling](#error-handling)

---

## Architecture Overview

### Base URLs

| Service | Base URL | Port |
|---------|----------|------|
| Registry | `http://localhost:9000` | 9000 |
| CVD Service | `http://localhost:8000` | 8000 |
| Breast Cancer | `http://localhost:8001` | 8001 |
| Alzheimer's | `http://localhost:8002` | 8002 |

### API Versioning

All prediction services use the API prefix: `/api/v1`

### Content Type

All requests must include:
```
Content-Type: application/json
```

---

## Service Registry API

The Registry Service provides service discovery, health monitoring, and metadata retrieval.

### 1. List All Services

Discover all available prediction services.

**Endpoint:** `GET /api/v1/services`

**Request:**
```bash
curl http://localhost:9000/api/v1/services
```

**Response:** `200 OK`
```json
[
  {
    "service_id": "cardiovascular_disease",
    "service_name": "Cardiovascular Disease Prediction API",
    "version": "1.0.0",
    "description": "Predicts cardiovascular disease risk from patient data",
    "base_url": "http://localhost:8000",
    "port": 8000,
    "endpoints": {
      "predict": "/api/v1/predict",
      "health": "/health",
      "model_info": "/api/v1/model-info"
    },
    "tags": ["cardiovascular", "disease", "classification", "health"],
    "input_schema": { ... },
    "output_schema": { ... }
  },
  {
    "service_id": "breast_cancer",
    "service_name": "Breast Cancer Prediction API",
    "version": "1.0.0",
    "description": "Predicts breast cancer from tumor features",
    "base_url": "http://localhost:8001",
    "port": 8001,
    "endpoints": {
      "predict": "/api/v1/predict",
      "health": "/health",
      "model_info": "/api/v1/model-info"
    },
    "tags": ["cancer", "breast", "classification"],
    "input_schema": { ... },
    "output_schema": { ... }
  },
  {
    "service_id": "alzheimers",
    "service_name": "Alzheimer's Disease Prediction API",
    "version": "1.0.0",
    "description": "Predicts Alzheimer's disease risk from cognitive assessments",
    "base_url": "http://localhost:8002",
    "port": 8002,
    "endpoints": {
      "predict": "/api/v1/predict",
      "health": "/health",
      "model_info": "/api/v1/model-info"
    },
    "tags": ["alzheimers", "dementia", "classification", "neurology"],
    "input_schema": { ... },
    "output_schema": { ... }
  }
]
```

**Frontend Usage:**
```javascript
// Discover all services
async function discoverServices() {
  const response = await fetch('http://localhost:9000/api/v1/services');
  const services = await response.json();

  // Build service catalog
  services.forEach(service => {
    console.log(`${service.service_name} available at ${service.base_url}`);
  });

  return services;
}
```

### 2. Get Specific Service

Get detailed information about a single service including full schemas.

**Endpoint:** `GET /api/v1/services/{service_id}`

**Request:**
```bash
curl http://localhost:9000/api/v1/services/cardiovascular_disease
```

**Response:** `200 OK`
```json
{
  "service_id": "cardiovascular_disease",
  "service_name": "Cardiovascular Disease Prediction API",
  "version": "1.0.0",
  "description": "Predicts cardiovascular disease risk from patient data",
  "base_url": "http://localhost:8000",
  "port": 8000,
  "endpoints": {
    "predict": "/api/v1/predict",
    "health": "/health",
    "model_info": "/api/v1/model-info"
  },
  "input_schema": {
    "type": "object",
    "properties": {
      "age_years": {
        "type": "number",
        "minimum": 18,
        "maximum": 100,
        "description": "Age in years"
      },
      "gender": {
        "type": "integer",
        "minimum": 1,
        "maximum": 2,
        "description": "Gender: 1=female, 2=male"
      },
      "height": {
        "type": "number",
        "minimum": 140,
        "maximum": 210,
        "description": "Height in centimeters"
      },
      "weight": {
        "type": "number",
        "minimum": 40,
        "maximum": 200,
        "description": "Weight in kilograms"
      },
      "ap_hi": {
        "type": "integer",
        "minimum": 80,
        "maximum": 200,
        "description": "Systolic blood pressure"
      },
      "ap_lo": {
        "type": "integer",
        "minimum": 60,
        "maximum": 130,
        "description": "Diastolic blood pressure"
      },
      "cholesterol": {
        "type": "integer",
        "minimum": 1,
        "maximum": 3,
        "description": "Cholesterol level: 1=normal, 2=above normal, 3=well above normal"
      },
      "gluc": {
        "type": "integer",
        "minimum": 1,
        "maximum": 3,
        "description": "Glucose level: 1=normal, 2=above normal, 3=well above normal"
      },
      "smoke": {
        "type": "integer",
        "minimum": 0,
        "maximum": 1,
        "description": "Smoking status: 0=no, 1=yes"
      },
      "alco": {
        "type": "integer",
        "minimum": 0,
        "maximum": 1,
        "description": "Alcohol intake: 0=no, 1=yes"
      },
      "active": {
        "type": "integer",
        "minimum": 0,
        "maximum": 1,
        "description": "Physical activity: 0=no, 1=yes"
      }
    },
    "required": ["age_years", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "prediction": {
        "type": "integer",
        "description": "Prediction: 0=no disease, 1=disease present"
      },
      "probability": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Probability of cardiovascular disease (0-1)"
      },
      "risk_level": {
        "type": "string",
        "description": "Risk level: low, medium, high"
      },
      "bmi": {
        "type": "number",
        "description": "Calculated BMI"
      }
    },
    "required": ["prediction", "probability", "risk_level", "bmi"]
  },
  "capabilities": {
    "model_name": "gradient_boosting",
    "metrics": {
      "accuracy": 0.7372,
      "roc_auc": 0.8029
    }
  },
  "tags": ["cardiovascular", "disease", "classification", "health"]
}
```

**Frontend Usage:**
```javascript
// Get service details and build dynamic form
async function getServiceSchema(serviceId) {
  const response = await fetch(`http://localhost:9000/api/v1/services/${serviceId}`);
  const service = await response.json();

  // Use input_schema to generate form fields
  const formFields = Object.entries(service.input_schema.properties).map(([fieldName, fieldSchema]) => ({
    name: fieldName,
    type: fieldSchema.type,
    min: fieldSchema.minimum,
    max: fieldSchema.maximum,
    description: fieldSchema.description,
    required: service.input_schema.required.includes(fieldName)
  }));

  return { service, formFields };
}
```

### 3. Search Services by Tags

Find services by medical condition or classification type.

**Endpoint:** `GET /api/v1/services/search/by-tags?tags={tags}`

**Request:**
```bash
curl "http://localhost:9000/api/v1/services/search/by-tags?tags=cardiovascular,classification"
```

**Response:** `200 OK`
```json
[
  {
    "service_id": "cardiovascular_disease",
    "service_name": "Cardiovascular Disease Prediction API",
    ...
  }
]
```

**Frontend Usage:**
```javascript
// Search for cancer-related services
async function searchServices(tags) {
  const tagString = tags.join(',');
  const response = await fetch(
    `http://localhost:9000/api/v1/services/search/by-tags?tags=${tagString}`
  );
  return await response.json();
}

// Usage
const cancerServices = await searchServices(['cancer', 'classification']);
```

### 4. Aggregate Health Check

Get health status of all services at once.

**Endpoint:** `GET /api/v1/health/all`

**Request:**
```bash
curl http://localhost:9000/api/v1/health/all
```

**Response:** `200 OK`
```json
{
  "cardiovascular_disease": {
    "service_name": "Cardiovascular Disease Prediction API",
    "status": "healthy",
    "base_url": "http://localhost:8000",
    "last_heartbeat": null
  },
  "breast_cancer": {
    "service_name": "Breast Cancer Prediction API",
    "status": "healthy",
    "base_url": "http://localhost:8001",
    "last_heartbeat": null
  },
  "alzheimers": {
    "service_name": "Alzheimer's Disease Prediction API",
    "status": "healthy",
    "base_url": "http://localhost:8002",
    "last_heartbeat": null
  }
}
```

**Frontend Usage:**
```javascript
// Monitor system health
async function checkSystemHealth() {
  const response = await fetch('http://localhost:9000/api/v1/health/all');
  const healthStatus = await response.json();

  const allHealthy = Object.values(healthStatus).every(
    service => service.status === 'healthy'
  );

  return { healthStatus, allHealthy };
}
```

### 5. Registry Health

Check if the registry itself is operational.

**Endpoint:** `GET /health`

**Request:**
```bash
curl http://localhost:9000/health
```

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "service": "Medical ML Service Registry",
  "registered_services": 3,
  "healthy_services": 3,
  "version": "1.0.0"
}
```

---

## Cardiovascular Disease API

Predicts cardiovascular disease risk from patient vitals and lifestyle factors.

### Base Information

- **Base URL:** `http://localhost:8000`
- **Service ID:** `cardiovascular_disease`
- **Model:** Gradient Boosting Classifier
- **Accuracy:** 73.7% | ROC-AUC: 0.80

### 1. Make Prediction

**Endpoint:** `POST /api/v1/predict`

**Request Headers:**
```
Content-Type: application/json
```

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

**Field Descriptions:**
- `age_years` (number, 18-100): Patient age in years
- `gender` (integer, 1-2): 1=female, 2=male
- `height` (number, 140-210): Height in centimeters
- `weight` (number, 40-200): Weight in kilograms
- `ap_hi` (integer, 80-200): Systolic blood pressure (mmHg)
- `ap_lo` (integer, 60-130): Diastolic blood pressure (mmHg)
- `cholesterol` (integer, 1-3): 1=normal, 2=above normal, 3=well above normal
- `gluc` (integer, 1-3): Glucose level: 1=normal, 2=above normal, 3=well above normal
- `smoke` (integer, 0-1): Smoking status: 0=no, 1=yes
- `alco` (integer, 0-1): Alcohol intake: 0=no, 1=yes
- `active` (integer, 0-1): Physical activity: 0=no, 1=yes

**Response:** `200 OK`
```json
{
  "prediction": 1,
  "probability": 0.6253,
  "risk_level": "medium",
  "bmi": 27.68
}
```

**Response Fields:**
- `prediction` (integer): 0=no disease, 1=disease present
- `probability` (float, 0-1): Probability of cardiovascular disease
- `risk_level` (string): "low", "medium", or "high"
- `bmi` (float): Calculated Body Mass Index

**Risk Level Thresholds:**
- Low: probability < 0.3
- Medium: 0.3 ≤ probability < 0.7
- High: probability ≥ 0.7

**Frontend Usage:**
```javascript
async function predictCVD(patientData) {
  const response = await fetch('http://localhost:8000/api/v1/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patientData)
  });

  if (!response.ok) {
    throw new Error('Prediction failed');
  }

  return await response.json();
}

// Example usage
const result = await predictCVD({
  age_years: 55,
  gender: 2,
  height: 170,
  weight: 80,
  ap_hi: 130,
  ap_lo: 85,
  cholesterol: 2,
  gluc: 1,
  smoke: 0,
  alco: 0,
  active: 1
});

console.log(`Risk Level: ${result.risk_level}`);
console.log(`Probability: ${(result.probability * 100).toFixed(1)}%`);
console.log(`BMI: ${result.bmi.toFixed(1)}`);
```

### 2. Get Model Info

**Endpoint:** `GET /api/v1/model-info`

**Response:** `200 OK`
```json
{
  "model_name": "gradient_boosting",
  "timestamp": "20251203_232317",
  "feature_names": [
    "age_years", "gender", "height", "bmi", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
    "pulse_pressure", "mean_arterial_pressure", "hypertension_stage",
    "bmi_category", "age_group", "health_risk_composite", "lifestyle_risk_score"
  ],
  "metrics": {
    "accuracy": 0.7372,
    "precision": 0.7563,
    "recall": 0.6915,
    "f1_score": 0.7225,
    "roc_auc": 0.8029
  }
}
```

### 3. Health Check

**Endpoint:** `GET /health`

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "service": "Cardiovascular Disease Prediction API",
  "model_loaded": true,
  "model_name": "gradient_boosting",
  "version": "1.0.0"
}
```

---

## Breast Cancer API

Predicts benign vs malignant tumors from Fine Needle Aspirate (FNA) imaging features.

### Base Information

- **Base URL:** `http://localhost:8001`
- **Service ID:** `breast_cancer`
- **Model:** Random Forest Classifier
- **Accuracy:** 97.4% | ROC-AUC: 0.99

### 1. Make Prediction

**Endpoint:** `POST /api/v1/predict`

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
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

**Field Categories (30 total features):**

**Mean Values (10 features):**
- `radius_mean`: Mean of distances from center to perimeter points
- `texture_mean`: Standard deviation of gray-scale values
- `perimeter_mean`: Mean perimeter of tumor
- `area_mean`: Mean area of tumor
- `smoothness_mean`: Mean local variation in radius lengths (0-1)
- `compactness_mean`: Mean of perimeter² / area - 1.0
- `concavity_mean`: Mean severity of concave portions
- `concave_points_mean`: Mean number of concave portions
- `symmetry_mean`: Mean symmetry (0-1)
- `fractal_dimension_mean`: Mean "coastline approximation" - 1

**Standard Error Values (10 features):**
- `radius_se`, `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`,
  `compactness_se`, `concavity_se`, `concave_points_se`, `symmetry_se`, `fractal_dimension_se`

**Worst/Largest Values (10 features):**
- `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`,
  `compactness_worst`, `concavity_worst`, `concave_points_worst`, `symmetry_worst`, `fractal_dimension_worst`

**Response:** `200 OK`
```json
{
  "prediction": 1,
  "probability": 0.9284,
  "diagnosis": "malignant",
  "confidence": "high"
}
```

**Response Fields:**
- `prediction` (integer): 0=benign, 1=malignant
- `probability` (float, 0-1): Probability of malignancy
- `diagnosis` (string): "benign" or "malignant"
- `confidence` (string): "low", "medium", or "high"

**Confidence Thresholds:**
- Low: 0.5 ≤ probability < 0.7
- Medium: 0.7 ≤ probability < 0.9
- High: probability ≥ 0.9 or probability < 0.1

**Frontend Usage:**
```javascript
async function predictBreastCancer(tumorFeatures) {
  const response = await fetch('http://localhost:8001/api/v1/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(tumorFeatures)
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Prediction failed');
  }

  return await response.json();
}

// Display results
const result = await predictBreastCancer(features);
console.log(`Diagnosis: ${result.diagnosis}`);
console.log(`Confidence: ${result.confidence} (${(result.probability * 100).toFixed(1)}%)`);
```

### 2. Get Model Info

**Endpoint:** `GET /api/v1/model-info`

**Response:** `200 OK`
```json
{
  "model_name": "random_forest",
  "timestamp": "20251130_172828",
  "feature_names": [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
  ],
  "metrics": {
    "accuracy": 0.9736,
    "precision": 1.0,
    "recall": 0.9285,
    "f1_score": 0.9629,
    "roc_auc": 0.9947
  }
}
```

### 3. Health Check

**Endpoint:** `GET /health`

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "service": "Breast Cancer Prediction API",
  "model_loaded": true,
  "model_name": "random_forest",
  "version": "1.0.0"
}
```

---

## Alzheimer's Disease API

Predicts dementia risk from cognitive assessments and brain imaging metrics.

### Base Information

- **Base URL:** `http://localhost:8002`
- **Service ID:** `alzheimers`
- **Model:** Support Vector Machine (RBF kernel)
- **Accuracy:** 100% | ROC-AUC: 1.0

### 1. Make Prediction

**Endpoint:** `POST /api/v1/predict`

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "age": 75,
  "gender": "M",
  "EDUC": 14,
  "SES": 2,
  "MMSE": 28,
  "CDR": 0.0,
  "eTIV": 1500,
  "nWBV": 0.75,
  "ASF": 1.2
}
```

**Field Descriptions:**
- `age` (integer, 55-100): Patient age in years
- `gender` (string): "M" for male or "F" for female
- `EDUC` (integer, 0-30): Years of formal education
- `SES` (integer, 1-5, optional): Socioeconomic status (1=highest, 5=lowest)
- `MMSE` (integer, 0-30): Mini Mental State Examination score
- `CDR` (float): Clinical Dementia Rating - **Must be one of: 0.0, 0.5, 1.0, 2.0, 3.0**
  - 0.0 = No dementia
  - 0.5 = Questionable/very mild dementia
  - 1.0 = Mild dementia
  - 2.0 = Moderate dementia
  - 3.0 = Severe dementia
- `eTIV` (float, 900-2200): Estimated Total Intracranial Volume in cm³
- `nWBV` (float, 0.6-0.9): Normalized Whole Brain Volume
- `ASF` (float, 0.8-1.6): Atlas Scaling Factor

**Response:** `200 OK`
```json
{
  "prediction": 0,
  "probability": 0.0152,
  "risk_score": 1.52,
  "stage": "none",
  "risk_level": "low"
}
```

**Response Fields:**
- `prediction` (integer): 0=Non-demented, 1=Demented
- `probability` (float, 0-1): Probability of dementia
- `risk_score` (float, 0-100): Probability × 100
- `stage` (string): "none", "questionable", "mild", "moderate", "severe" (based on CDR)
- `risk_level` (string): "low", "medium", "high"

**Risk Level Thresholds:**
- Low: risk_score < 30
- Medium: 30 ≤ risk_score < 70
- High: risk_score ≥ 70

**CDR to Stage Mapping:**
- CDR 0.0 → "none"
- CDR 0.5 → "questionable"
- CDR 1.0 → "mild"
- CDR 2.0 → "moderate"
- CDR 3.0 → "severe"

**Frontend Usage:**
```javascript
async function predictAlzheimers(assessmentData) {
  const response = await fetch('http://localhost:8002/api/v1/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(assessmentData)
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Prediction failed');
  }

  return await response.json();
}

// Example usage with form validation
function validateCDR(value) {
  const validCDR = [0.0, 0.5, 1.0, 2.0, 3.0];
  if (!validCDR.includes(parseFloat(value))) {
    throw new Error('CDR must be 0.0, 0.5, 1.0, 2.0, or 3.0');
  }
  return parseFloat(value);
}

const result = await predictAlzheimers({
  age: 75,
  gender: "M",
  EDUC: 14,
  SES: 2,
  MMSE: 28,
  CDR: validateCDR(0.0),
  eTIV: 1500,
  nWBV: 0.75,
  ASF: 1.2
});

console.log(`Risk Level: ${result.risk_level}`);
console.log(`Risk Score: ${result.risk_score}%`);
console.log(`Stage: ${result.stage}`);
```

### 2. Get Model Info

**Endpoint:** `GET /api/v1/model-info`

**Response:** `200 OK`
```json
{
  "model_name": "svm",
  "timestamp": "20251130_190846",
  "feature_names": [
    "Age", "Gender", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"
  ],
  "metrics": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "roc_auc": 1.0
  }
}
```

### 3. Health Check

**Endpoint:** `GET /health`

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "service": "Alzheimer's Disease Prediction API",
  "model_loaded": true,
  "model_name": "svm",
  "version": "1.0.0",
  "timestamp": "2025-12-04T00:44:30.851932"
}
```

---

## Frontend Integration Guide

### Complete Workflow Example

Here's a complete example of how a frontend application should interact with the system:

```javascript
class MedicalMLClient {
  constructor(registryUrl = 'http://localhost:9000') {
    this.registryUrl = registryUrl;
    this.services = {};
  }

  // 1. Initialize: Discover all services
  async initialize() {
    const response = await fetch(`${this.registryUrl}/api/v1/services`);
    const services = await response.json();

    // Cache services by ID
    services.forEach(service => {
      this.services[service.service_id] = service;
    });

    return services;
  }

  // 2. Get service details for form generation
  async getServiceSchema(serviceId) {
    if (!this.services[serviceId]) {
      const response = await fetch(
        `${this.registryUrl}/api/v1/services/${serviceId}`
      );
      this.services[serviceId] = await response.json();
    }
    return this.services[serviceId];
  }

  // 3. Generate form fields from schema
  generateFormFields(inputSchema) {
    return Object.entries(inputSchema.properties).map(([fieldName, field]) => ({
      name: fieldName,
      label: fieldName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      type: this.getInputType(field),
      min: field.minimum,
      max: field.maximum,
      description: field.description,
      required: inputSchema.required?.includes(fieldName) || false,
      // For select fields (enum values)
      options: field.enum || null
    }));
  }

  getInputType(field) {
    if (field.type === 'integer') return 'number';
    if (field.type === 'number') return 'number';
    if (field.type === 'string') return 'text';
    if (field.enum) return 'select';
    return 'text';
  }

  // 4. Make prediction
  async predict(serviceId, data) {
    const service = this.services[serviceId];
    if (!service) {
      throw new Error(`Service ${serviceId} not found`);
    }

    const predictUrl = `${service.base_url}${service.endpoints.predict}`;
    const response = await fetch(predictUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Prediction failed');
    }

    return await response.json();
  }

  // 5. Check system health
  async checkHealth() {
    const response = await fetch(`${this.registryUrl}/api/v1/health/all`);
    return await response.json();
  }

  // 6. Search services
  async searchByTags(tags) {
    const tagString = tags.join(',');
    const response = await fetch(
      `${this.registryUrl}/api/v1/services/search/by-tags?tags=${tagString}`
    );
    return await response.json();
  }
}

// Usage Example
async function main() {
  const client = new MedicalMLClient();

  // 1. Discover services
  console.log('Discovering services...');
  const services = await client.initialize();
  console.log(`Found ${services.length} services`);

  // 2. Get CVD service schema
  console.log('\nGetting CVD service schema...');
  const cvdService = await client.getServiceSchema('cardiovascular_disease');
  const formFields = client.generateFormFields(cvdService.input_schema);

  // 3. Display form fields (for UI generation)
  console.log('\nForm fields for CVD prediction:');
  formFields.forEach(field => {
    console.log(`- ${field.label}: ${field.description}`);
  });

  // 4. Make a prediction
  console.log('\nMaking CVD prediction...');
  const result = await client.predict('cardiovascular_disease', {
    age_years: 55,
    gender: 2,
    height: 170,
    weight: 80,
    ap_hi: 130,
    ap_lo: 85,
    cholesterol: 2,
    gluc: 1,
    smoke: 0,
    alco: 0,
    active: 1
  });

  console.log('Prediction result:', result);
  console.log(`Risk Level: ${result.risk_level}`);
  console.log(`Probability: ${(result.probability * 100).toFixed(1)}%`);

  // 5. Check system health
  console.log('\nChecking system health...');
  const health = await client.checkHealth();
  const healthyCount = Object.values(health).filter(
    s => s.status === 'healthy'
  ).length;
  console.log(`${healthyCount}/${Object.keys(health).length} services healthy`);

  // 6. Search for cancer services
  console.log('\nSearching for cancer services...');
  const cancerServices = await client.searchByTags(['cancer']);
  console.log(`Found ${cancerServices.length} cancer-related service(s)`);
}

main().catch(console.error);
```

### React Component Example

```jsx
import React, { useState, useEffect } from 'react';

function CVDPrediction() {
  const [formData, setFormData] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [schema, setSchema] = useState(null);

  useEffect(() => {
    // Fetch service schema on mount
    async function fetchSchema() {
      try {
        const response = await fetch(
          'http://localhost:9000/api/v1/services/cardiovascular_disease'
        );
        const service = await response.json();
        setSchema(service.input_schema);
      } catch (err) {
        setError('Failed to load form schema');
      }
    }
    fetchSchema();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: parseFloat(value) || value }));
  };

  if (!schema) return <div>Loading form...</div>;

  return (
    <div className="cvd-prediction">
      <h2>Cardiovascular Disease Risk Assessment</h2>

      <form onSubmit={handleSubmit}>
        {Object.entries(schema.properties).map(([field, config]) => (
          <div key={field} className="form-group">
            <label>
              {field.replace(/_/g, ' ')}
              {schema.required.includes(field) && <span className="required">*</span>}
            </label>
            <input
              type="number"
              min={config.minimum}
              max={config.maximum}
              step={config.type === 'integer' ? 1 : 0.1}
              onChange={(e) => handleChange(field, e.target.value)}
              required={schema.required.includes(field)}
            />
            <small>{config.description}</small>
          </div>
        ))}

        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Predict Risk'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className={`result risk-${result.risk_level}`}>
          <h3>Prediction Results</h3>
          <div className="risk-level">
            Risk Level: <strong>{result.risk_level.toUpperCase()}</strong>
          </div>
          <div className="probability">
            Probability: {(result.probability * 100).toFixed(1)}%
          </div>
          <div className="bmi">
            BMI: {result.bmi.toFixed(1)}
          </div>
        </div>
      )}
    </div>
  );
}

export default CVDPrediction;
```

---

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | When It Occurs |
|------|---------|---------------|
| 200 | OK | Successful request |
| 404 | Not Found | Service ID not found in registry |
| 422 | Unprocessable Content | Validation error in request data |
| 500 | Internal Server Error | Model loading failed or prediction error |

### Error Response Format

All services return errors in this format:

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "CDR"],
      "msg": "Value error, CDR must be one of [0.0, 0.5, 1.0, 2.0, 3.0]",
      "input": 1.5,
      "ctx": {"error": {}}
    }
  ]
}
```

### Frontend Error Handling

```javascript
async function makePrediction(serviceUrl, data) {
  try {
    const response = await fetch(serviceUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      const error = await response.json();

      // Handle validation errors (422)
      if (response.status === 422 && error.detail) {
        const validationErrors = error.detail.map(err => ({
          field: err.loc[1],
          message: err.msg,
          value: err.input
        }));
        throw new ValidationError('Invalid input data', validationErrors);
      }

      // Handle other errors
      throw new Error(error.detail || 'Prediction failed');
    }

    return await response.json();
  } catch (err) {
    console.error('Prediction error:', err);
    throw err;
  }
}

class ValidationError extends Error {
  constructor(message, errors) {
    super(message);
    this.name = 'ValidationError';
    this.errors = errors;
  }
}
```

### Field Validation Examples

```javascript
// CVD Service Validators
const cvdValidators = {
  age_years: (val) => val >= 18 && val <= 100,
  gender: (val) => [1, 2].includes(val),
  height: (val) => val >= 140 && val <= 210,
  weight: (val) => val >= 40 && val <= 200,
  ap_hi: (val) => val >= 80 && val <= 200,
  ap_lo: (val) => val >= 60 && val <= 130,
  cholesterol: (val) => [1, 2, 3].includes(val),
  gluc: (val) => [1, 2, 3].includes(val),
  smoke: (val) => [0, 1].includes(val),
  alco: (val) => [0, 1].includes(val),
  active: (val) => [0, 1].includes(val)
};

// Alzheimer's Service Validators
const alzhValidators = {
  age: (val) => val >= 55 && val <= 100,
  gender: (val) => ['M', 'F'].includes(val),
  EDUC: (val) => val >= 0 && val <= 30,
  SES: (val) => val === null || (val >= 1 && val <= 5),
  MMSE: (val) => val >= 0 && val <= 30,
  CDR: (val) => [0.0, 0.5, 1.0, 2.0, 3.0].includes(val),
  eTIV: (val) => val >= 900 && val <= 2200,
  nWBV: (val) => val >= 0.6 && val <= 0.9,
  ASF: (val) => val >= 0.8 && val <= 1.6
};
```

---

## Interactive API Documentation

When services are running, visit these URLs for interactive Swagger documentation:

- **Registry**: http://localhost:9000/docs
- **CVD Service**: http://localhost:8000/docs
- **Breast Cancer**: http://localhost:8001/docs
- **Alzheimer's**: http://localhost:8002/docs

These provide:
- Interactive API testing
- Request/response examples
- Schema definitions
- Try-it-out functionality

---

## Summary

### Quick Reference

| Service | Endpoint | Input Fields | Output Fields |
|---------|----------|--------------|---------------|
| **CVD** | POST /api/v1/predict | 11 fields (vitals + lifestyle) | prediction, probability, risk_level, bmi |
| **Breast Cancer** | POST /api/v1/predict | 30 fields (tumor features) | prediction, probability, diagnosis, confidence |
| **Alzheimer's** | POST /api/v1/predict | 9 fields (cognitive + imaging) | prediction, probability, risk_score, stage, risk_level |

### Workflow Summary

1. **Discover Services**: `GET /api/v1/services`
2. **Get Schema**: `GET /api/v1/services/{service_id}`
3. **Generate Form**: Use `input_schema` to create dynamic forms
4. **Make Prediction**: `POST {base_url}/api/v1/predict`
5. **Monitor Health**: `GET /api/v1/health/all`

All services follow the same pattern, making it easy to build a unified frontend interface.

---

**Last Updated**: December 4, 2025
**API Version**: 1.0.0
**Status**: Production Ready
