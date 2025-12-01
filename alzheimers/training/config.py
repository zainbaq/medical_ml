"""
Configuration for Alzheimer's disease prediction model training
"""
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "oasis_longitudinal.csv"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Features (9 core features from OASIS dataset)
FEATURE_COLUMNS = [
    'Age',      # Age in years
    'Gender',   # Gender: 1=Male, 0=Female
    'EDUC',     # Years of education
    'SES',      # Socioeconomic status (1-5, lower is higher)
    'MMSE',     # Mini Mental State Examination (0-30)
    'CDR',      # Clinical Dementia Rating (0, 0.5, 1, 2, 3)
    'eTIV',     # Estimated Total Intracranial Volume
    'nWBV',     # Normalized Whole Brain Volume (0-1)
    'ASF'       # Atlas Scaling Factor
]

TARGET_COLUMN = 'Group'  # Demented/Nondemented

# Columns to drop from raw data
DROP_COLUMNS = ['Subject ID', 'MRI ID', 'Visit', 'MR Delay', 'Hand']

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Gender encoding
GENDER_ENCODING = {'M': 1, 'F': 0}

# Target encoding
TARGET_ENCODING = {'Demented': 1, 'Nondemented': 0}

# CDR to Stage mapping (for API output)
CDR_STAGE_MAP = {
    0.0: 'none',
    0.5: 'questionable',
    1.0: 'mild',
    2.0: 'moderate',
    3.0: 'severe'
}

# Model configurations for GridSearchCV
MODELS_CONFIG = {
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4]  # for poly kernel
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000]
    }
}
