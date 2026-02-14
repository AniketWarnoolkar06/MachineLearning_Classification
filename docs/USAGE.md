# Usage Guide

## Table of Contents
1. [Running Jupyter Notebooks](#running-jupyter-notebooks)
2. [Running the Streamlit App](#running-the-streamlit-app)
3. [Deployment](#deployment)
3. [Understanding Model Threshold Files](#understanding-model-threshold-files)
4. [Working with Test Data](#working-with-test-data)

---

## Running Jupyter Notebooks

The project includes several notebooks for different models and analyses:

### Available Notebooks:
- `01_data_inspection.ipynb` - Initial data exploration and analysis
- `02_logistic_regression.ipynb` - Logistic Regression model
- `03_decision_tree.ipynb` - Decision Tree model
- `04_knn.ipynb` - K-Nearest Neighbors model
- `05_naive_bayes.ipynb` - Naive Bayes model
- `06_random_forest.ipynb` - Random Forest model
- `07_xgboost.ipynb` - XGBoost model
- `08_model_comparison.ipynb` - Comparison of all models

### Steps to Run:

1. **Activate your virtual environment** (if not already activated):
   ```bash
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   This will open Jupyter in your default browser.

3. **Navigate to the `notebooks/` folder** and open any notebook.

4. **Run cells sequentially** using `Shift + Enter` or the "Run" button.

### Recommended Order:
1. Start with `01_data_inspection.ipynb` to understand the dataset
2. Run individual model notebooks (`02` through `07`) to see each model's implementation
3. Finally, review `08_model_comparison.ipynb` for the overall comparison

---

## Running the Streamlit App

The Streamlit app provides an interactive dashboard for evaluating trained models.

### Steps to Launch:

1. **Navigate to the project root directory**:
   ```bash
   cd /path/to/MachineLearning_Classification
   ```

2. **Activate your virtual environment**:
   ```bash
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Access the app**: The app will automatically open in your browser at `http://localhost:8501`

### Using the App:

0. **(Optional) Download sample test data**
   - The app provides a **Download Sample Test File** button (from `data/test/bank_test.csv`) when available.

1. **Upload Test Data**: 
   - Click "Browse files" and upload your test CSV file
   - The file must contain a `y` column with ground truth labels

2. **Select a Model**:
   - Choose from the dropdown menu:
     - Logistic Regression
     - Decision Tree
     - K-Nearest Neighbors (KNN)
     - Naive Bayes
     - Random Forest
     - XGBoost

3. **View Results**:
   - The app displays:
     - Evaluation metrics (Accuracy, Precision, Recall, F1-score, MCC, AUC-ROC)
     - Confusion matrix
     - Classification report
   - ROC curve

4. **Compare Models**: You can re-run with different models to compare performance.

### App Features:
- Interactive model selection
- Real-time evaluation on uploaded test data
- Visual performance metrics
- Confusion matrix display
- Detailed classification reports

---

## Deployment

The Streamlit application is deployed here:

https://machinelearningclassification-kpsaehclcg2cmsospdgwat.streamlit.app

---

## Understanding Model Threshold Files

The `models/` directory contains both trained models and threshold files:

- `*_model.pkl` (example: `xgboost_model.pkl`)
- `*_threshold.txt` (example: `xgboost_threshold.txt`)

Threshold files for each model:
- `logistic_threshold.txt`
- `decision_tree_threshold.txt`
- `knn_threshold.txt`
- `naive_bayes_threshold.txt`
- `random_forest_threshold.txt`
- `xgboost_threshold.txt`

### What Are Threshold Files?

These files store the **optimal probability threshold** for each model. 

- **Default threshold**: 0.5 (predict class 1 if probability > 0.5)
- **Optimized threshold**: Tuned to maximize F1-score on validation data

### Why Threshold Tuning?

Due to class imbalance in the dataset:
- Majority class: `no` (customer did not subscribe)
- Minority class: `yes` (customer subscribed)

Default thresholds often perform poorly on imbalanced data. By tuning the threshold, we can:
- Improve recall for the minority class
- Balance precision and recall
- Maximize F1-score and MCC

### How They're Used:

When making predictions:
1. Model outputs probability scores
2. Threshold file is loaded
3. Predictions are made using: `prediction = 1 if probability > threshold else 0`

---

## Working with Test Data

### Test Data Location:
- `data/test/bank_test.csv` - Pre-split test set

### Required Format:

Your test CSV file should have:
- All feature columns used during training
- A `y` column with ground truth labels (`yes` or `no`)

If the feature columns do not match what the selected model expects (for example, missing columns or extra columns), the model may fail during prediction.

### Example Structure:
```
age,job,marital,education,balance,...,y
30,technician,married,secondary,1500,...,no
45,management,single,tertiary,2500,...,yes
```

### Using Custom Test Data:

1. Ensure your data has the same features as the training data
2. Include the `y` column for evaluation
3. Upload via the Streamlit app
4. The app will automatically preprocess and evaluate

---

## Common Commands Summary

```bash
# Activate environment
source venv/bin/activate

# Run notebooks
jupyter notebook

# Run Streamlit app
streamlit run app.py

# Deactivate environment
deactivate
```

---

For installation instructions, see [INSTALLATION.md](INSTALLATION.md).  
For project structure details, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).
