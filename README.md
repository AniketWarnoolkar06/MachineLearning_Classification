# Bank Marketing Prediction — Machine Learning Project

## Overview

The goal of this project is to predict whether a customer will subscribe to a term deposit based on their demographic details and past interactions with the bank.

This is a binary classification problem:
- "yes" → customer subscribed
- "no" → customer did not subscribe

The dataset is highly imbalanced, so evaluation is focused more on metrics like F1-score, ROC-AUC, and MCC instead of just accuracy.

---

## Dataset

The dataset used is the Bank Marketing dataset from the UCI Machine Learning Repository.

Key details:
- Around 45,000 records
- Combination of numerical and categorical features
- Target variable is imbalanced (~88% "no", ~12% "yes")

Because of the imbalance, predicting only the majority class can give high accuracy but poor real performance. Hence, better metrics are used.

---

## Data Preprocessing

The following preprocessing steps were applied:

- Removed the `duration` column to avoid data leakage
- Converted target variable to binary (0 and 1)
- Performed train-test split using stratification
- Encoded categorical variables using OneHotEncoder
- Applied scaling for models that require it (Logistic, KNN, Naive Bayes)

---

## Feature Engineering

Additional features were created to improve model performance:

- `prev_contacted`  
  Indicates whether the customer was previously contacted (based on pdays)

- `campaign_bin`  
  Groups the number of contacts into categories (once, few, many)

- `age_group`  
  Segments customers into age categories (young, mid, senior, old)

- `balance_log`  
  Log transformation of balance to handle skewed distribution

Also:
- Extreme values were clipped using quantiles to reduce the effect of outliers

---

## Models Implemented

The following models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

Each model was trained using:
- Pipeline (preprocessing + model)
- GridSearchCV for hyperparameter tuning
- F1-score as the primary metric

---

## Handling Class Imbalance

Different strategies were used depending on the model:

- Logistic Regression → class_weight="balanced"
- Decision Tree / Random Forest → class_weight="balanced"
- XGBoost → scale_pos_weight
- Threshold tuning applied for all models

---

## Threshold Tuning

Instead of using the default threshold (0.5), the optimal threshold was found for each model by:

- Testing multiple thresholds
- Selecting the one that maximizes F1-score

This significantly improved model performance on the minority class.

---

## Model Performance

| Model               | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|--------------------|---------|-------|-----------|--------|--------|--------|
| XGBoost            | 0.8797  | 0.797 | 0.486     | 0.500  | 0.493  | 0.425 |
| Random Forest      | 0.8717  | 0.799 | 0.458     | 0.519  | 0.486  | 0.414 |
| Decision Tree      | 0.8703  | 0.757 | 0.452     | 0.509  | 0.478  | 0.406 |
| Logistic Regression| 0.8664  | 0.777 | 0.436     | 0.480  | 0.457  | 0.381 |
| Naive Bayes        | 0.8545  | 0.753 | 0.393     | 0.445  | 0.417  | 0.335 |
| KNN                | 0.8566  | 0.735 | 0.395     | 0.423  | 0.409  | 0.327 |

---

## Model Observations

### Logistic Regression
Logistic Regression provides a strong baseline model. It achieves balanced precision (~0.44) and recall (~0.48), showing that it can reasonably identify both classes. However, being a linear model, it cannot capture complex non-linear relationships in the data, which limits its overall performance.

### Decision Tree
The Decision Tree model captures non-linear patterns effectively, leading to improved recall (~0.51). However, it tends to overfit the training data, even with pruning. This results in slightly lower generalization compared to ensemble methods.

### K-Nearest Neighbors (KNN)
KNN shows the weakest performance among all models. It is sensitive to high dimensionality and class imbalance. Since it relies on distance calculations, the presence of many one-hot encoded features negatively affects its performance.

### Naive Bayes
Naive Bayes performs moderately well despite its simplicity. However, it assumes independence between features, which is not realistic for this dataset. This limits its ability to capture relationships between variables, resulting in lower performance.

### Random Forest
Random Forest improves significantly over a single decision tree by combining multiple trees. It reduces overfitting and achieves higher recall (~0.52) and strong AUC (~0.80). It provides a good balance between bias and variance.

### XGBoost
XGBoost is the best-performing model in this project. It effectively handles class imbalance and captures complex feature interactions. It achieves the highest F1-score (~0.49) and MCC (~0.42), making it the most reliable model for this problem.

---

## Final Model Selection

Based on evaluation metrics (especially F1-score and MCC), **XGBoost** was selected as the final model because:

- It provides the best balance between precision and recall
- It handles class imbalance effectively
- It captures complex relationships in the data
- It achieves the highest overall performance

---

## Model Comparison

A separate notebook (`08_model_comparison.ipynb`) was created to:

- Load all trained models
- Evaluate them on the same test dataset
- Compare metrics
- Rank models based on F1-score
- Visualize results

---

## Streamlit Application

A Streamlit app was developed to allow interactive model evaluation.

### Features:
- Upload test dataset (CSV)
- Select a trained model
- Compute predictions
- Display evaluation metrics
- Show confusion matrix
- Display ROC curve
- Show classification report
- Download predictions

---

## How to Run the App

```bash
streamlit run app.py

---

## Deployment

The Streamlit application is deployed and available here:

https://machinelearningclassification-kpsaehclcg2cmsospdgwat.streamlit.app
