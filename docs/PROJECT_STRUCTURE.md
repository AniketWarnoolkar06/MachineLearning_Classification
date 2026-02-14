# Project Structure

This document provides an overview of the project organization and the purpose of each directory and file.

```
MachineLearning_Classification/
│
├── README.md                   # Project overview and results summary
├── requirements.txt            # Python dependencies
│
├── app/                        # Streamlit web application
│   └── app.py                  # Main Streamlit app for model evaluation
│
├── data/                       # Data directory
│   ├── raw/                    # Original, unprocessed data
│   │   └── bank-full.csv       # Full bank marketing dataset
│   └── test/                   # Test dataset
│       └── bank_test.csv       # Held-out test set for evaluation
│
├── models/                     # Saved models and configuration
│   ├── decision_tree_threshold.txt     # Optimal threshold for Decision Tree
│   ├── knn_threshold.txt               # Optimal threshold for KNN
│   ├── logistic_threshold.txt          # Optimal threshold for Logistic Regression
│   ├── naive_bayes_threshold.txt       # Optimal threshold for Naive Bayes
│   ├── random_forest_threshold.txt     # Optimal threshold for Random Forest
│   └── xgboost_threshold.txt           # Optimal threshold for XGBoost
│
├── notebooks/                  # Jupyter notebooks for analysis and modeling
│   ├── 01_data_inspection.ipynb        # Data exploration and EDA
│   ├── 02_logistic_regression.ipynb    # Logistic Regression implementation
│   ├── 03_decision_tree.ipynb          # Decision Tree implementation
│   ├── 04_knn.ipynb                    # K-Nearest Neighbors implementation
│   ├── 05_naive_bayes.ipynb            # Naive Bayes implementation
│   ├── 06_random_forest.ipynb          # Random Forest implementation
│   ├── 07_xgboost.ipynb                # XGBoost implementation
│   └── 08_model_comparison.ipynb       # Model comparison and final results
│
├── src/                        # Source code modules
│   ├── __init__.py             # Makes src a Python package
│   │
│   ├── data/                   # Data processing modules
│   │   ├── __init__.py
│   │   ├── load_data.py        # Functions for loading data
│   │   └── preprocess.py       # Data preprocessing and feature engineering
│   │
│   └── models/                 # Model implementation modules
│       ├── __init__.py
│       ├── decision_tree.py    # Decision Tree model code
│       ├── knn.py              # KNN model code
│       ├── logistic.py         # Logistic Regression model code
│       ├── naive_bayes.py      # Naive Bayes model code
│       ├── random_forest.py    # Random Forest model code
│       └── xgboost_model.py    # XGBoost model code
│
├── docs/                       # Documentation
│   ├── INSTALLATION.md         # Installation instructions
│   ├── USAGE.md                # Usage guide
│   └── PROJECT_STRUCTURE.md    # This file
│
└── venv/                       # Virtual environment (not tracked in git)
```

---

## Directory Descriptions

### `app/`
Contains the Streamlit web application for interactive model evaluation.

**Key File:**
- `app.py` - Main application that allows users to upload test data and evaluate trained models

### `data/`
Stores all datasets used in the project.

**Subdirectories:**
- `raw/` - Original, unmodified datasets
- `test/` - Test datasets for model evaluation

**Key Files:**
- `bank-full.csv` - Complete bank marketing dataset from UCI repository
- `bank_test.csv` - Held-out test set (not used during training)

### `models/`
Contains saved model artifacts and configuration files.

**Threshold Files:**
- Store optimal probability thresholds for each model
- Used to improve performance on imbalanced data
- Thresholds were tuned to maximize F1-score on validation data

**Note:** Trained model `.pkl` files are typically stored here but may be excluded from version control due to size.

### `notebooks/`
Jupyter notebooks for experimentation, analysis, and documentation.

**Workflow:**
1. **01_data_inspection.ipynb** - Explore data, check for missing values, visualize distributions
2. **02-07** - Individual model notebooks with training, tuning, and evaluation
3. **08_model_comparison.ipynb** - Compare all models and select the best performer

Each notebook includes:
- Data loading and preprocessing
- Model training
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Threshold optimization
- Evaluation metrics and visualizations

### `src/`
Reusable Python modules for data processing and modeling.

**Structure:**
- `data/` - Data loading and preprocessing utilities
  - `load_data.py` - Functions to load CSV files
  - `preprocess.py` - Feature engineering, encoding, scaling

- `models/` - Model-specific code
  - Each `.py` file contains model training and prediction functions
  - Allows code reuse across notebooks and the Streamlit app

**Benefits:**
- DRY (Don't Repeat Yourself) principle
- Easy to maintain and update
- Enables testing and modularity

### `docs/`
Project documentation for setup, usage, and structure.

**Files:**
- `INSTALLATION.md` - How to set up the project environment
- `USAGE.md` - How to run notebooks and the Streamlit app
- `PROJECT_STRUCTURE.md` - Overview of project organization (this file)

---

## Key Files

### `README.md`
High-level project overview including:
- Problem statement
- Dataset description
- Models and evaluation metrics
- Results table
- Model performance observations

### `requirements.txt`
Lists all Python dependencies required for the project:
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
streamlit
joblib
```

Install with: `pip install -r requirements.txt`

### `.gitignore`
Specifies files and directories to exclude from version control:
- Virtual environments (`venv/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Jupyter checkpoints (`.ipynb_checkpoints/`)
- Large model files (`.pkl`, `.joblib`)

---

## Workflow Summary

### 1. Data Preparation
- Load data from `data/raw/`
- Preprocess using `src/data/preprocess.py`
- Split into train/validation/test sets

### 2. Model Development
- Experiment in notebooks (`notebooks/01-07`)
- Use functions from `src/models/`
- Tune hyperparameters
- Optimize probability thresholds
- Save thresholds to `models/`

### 3. Model Evaluation
- Run `08_model_comparison.ipynb`
- Compare all models
- Select best performer (XGBoost)

### 4. Deployment
- Use Streamlit app (`app/app.py`)
- Upload test data
- Evaluate models interactively
- View metrics and visualizations

---

## Best Practices

### Code Organization
- Keep notebooks for experimentation
- Move reusable code to `src/` modules
- Document functions and classes

### Data Management
- Never modify raw data
- Store processed data separately
- Version control small datasets only

### Model Management
- Save models with version numbers
- Track hyperparameters
- Document threshold values

---

For installation instructions, see [INSTALLATION.md](INSTALLATION.md).  
For usage instructions, see [USAGE.md](USAGE.md).
