# ❤️ Heart Disease Prediction System

## Overview

The Heart Disease Prediction System is a Streamlit-based machine learning application designed to assess the likelihood of heart disease using patient clinical information.

The application trains and compares multiple machine learning classification algorithms and automatically selects the model with the highest test accuracy.

## Machine Learning Models

The system compares the following classification models:

* Logistic Regression
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Gradient Boosting
* Naive Bayes
* Extra Trees
* Multi-Layer Perceptron (MLP)

## Main Features

### 1. Model Comparison

The application trains the available classification models using the heart disease dataset and calculates their classification accuracy.

### 2. Automatic Model Selection

The model with the highest test accuracy is automatically selected for patient prediction.

### 3. Existing Patient Prediction

Users can select an existing patient from the dataset by entering the patient's index number and generate a prediction.

### 4. New Patient Prediction

Users can enter clinical information for a new patient, including:

* Sex
* Dataset
* Chest Pain (CP)
* Fasting Blood Sugar (FBS)
* Resting ECG
* Exercise-Induced Angina (Exang)
* Slope
* CA
* Thal
* Age
* Resting Blood Pressure
* Cholesterol
* Maximum Heart Rate
* Oldpeak

### 5. Patient Status

The system displays the prediction as either:

* NO HEART DISEASE
* HEART DISEASE

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn

## Project Files

```text
01_Binary_Logistic/
│
├── app.py
├── heart_disease.csv
├── requirements.txt
└── README.md
```

## Running the Application Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
py -m streamlit run app.py
```

The application will open in a web browser.

## Dataset

The application uses the `heart_disease.csv` dataset supplied with this project.

## Purpose

This application is intended for educational, research, and demonstration purposes. Machine learning predictions should not be treated as a substitute for professional medical diagnosis or clinical decision-making.
