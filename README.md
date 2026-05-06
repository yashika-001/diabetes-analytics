# 🏥 Predictive Analytics & Readmission Analysis for Diabetes Management

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diabetes-analytics-iojbmgfbvxsat7qyihxktm.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Scikit--learn-orange.svg)](https://scikit-learn.org/)

**Live Dashboard / App:** [Click here to view the Streamlit App](https://diabetes-analytics-iojbmgfbvxsat7qyihxktm.streamlit.app/)

## 📌 Project Overview
This project is a comprehensive machine learning pipeline designed to improve diabetes management and hospital resource efficiency. It is divided into two core components:
1. **Predictive Modeling:** A diagnostic tool utilizing advanced data imputation and classification algorithms to assess diabetes risk based on patient vitals.
2. **Clinical Readmission Analysis:** An in-depth exploratory data analysis (EDA) of a secondary hospital dataset to uncover the relationship between HbA1c monitoring, medication adjustments, and 30-day readmission rates.

## 🚀 Key Features & Methodology

### Part 1: Predictive Modeling Pipeline
* **Advanced Data Imputation:** Implemented **K-Nearest Neighbors (KNN)** imputation to handle missing values in critical clinical features (e.g., Glucose, BMI, Insulin), replacing naive mean/median approaches for a more scientifically robust dataset.
* **Model Training & Optimization:** Standardized input features and trained a **Logistic Regression** model. Configured class weights to address dataset imbalance, prioritizing high-recall for the positive (diabetic) class to minimize critical false-negative diagnoses.
* **Evaluation:** Achieved ~76% accuracy, evaluated using a comprehensive suite of metrics including Precision, Recall, Confusion Matrices, and ROC-AUC curves. 

### Part 2: Readmission Analytics
* **Bivariate & Trivariate Analysis:** Analyzed over 100,000 hospital records to correlate HbA1c measurement frequency, medication changes, primary diagnoses, and hospital readmission.
* **Data Visualization:** Developed comprehensive heatmaps and clustered bar plots using `Seaborn` and `Matplotlib` to communicate findings clearly to clinical stakeholders.

## 💡 Clinical Insights & Impact
* **Proactive Intervention:** Data proved that measuring HbA1c triggers proactive medication changes 56% of the time, jumping to 65% for dangerously high results (>8).
* **Readmission Reduction:** Regular HbA1c monitoring correlates with a significant decrease in critical <30-day hospital readmissions.
* **Stratified Impact:** The reduction in readmission rates is highly concentrated in primary diabetic cohorts, providing evidence-based recommendations for targeted hospital monitoring protocols.

## 🛠️ Technical Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Logistic Regression, KNNImputer, StandardScaler)
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit, Joblib

## 📂 Repository Structure
```text
├── app.py                   # Main Streamlit application script
├── requirements.txt         # Python dependencies
├── diabetes_model.pkl       # Serialized Logistic Regression model
├── scaler.pkl               # Serialized StandardScaler for input normalization
├── heatmap.png              # Exported visualization from Part 2
├── barchart.png             # Exported visualization from Part 2
└── README.md                # Project documentation

## 📚 Data Sources & References
This project utilizes publicly available clinical datasets and replicates findings from peer-reviewed medical research:
* **Dataset 1 (Predictive Modeling):** [Pima Indians Diabetes Database] 
* **Dataset 2 (Readmission Analysis):** [UCI Machine Learning Repository - Diabetes 130-US hospitals for years 1999-2008](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
* **Academic Reference:** Replicated readmission and HbA1c monitoring methodologies based on the accompanying clinical research paper provided in the UCI repository.
