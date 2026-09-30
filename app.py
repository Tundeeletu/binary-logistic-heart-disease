# ============================================================
# HEART DISEASE PREDICTION SYSTEM
# STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ HEART DISEASE PREDICTION SYSTEM")
st.write("Machine Learning Patient Assessment")
st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

file_path = "heart_disease.csv"

df = pd.read_csv(file_path)


# ============================================================
# DATA PREPROCESSING
# ============================================================

categ = [
    'sex',
    'dataset',
    'cp',
    'fbs',
    'restecg',
    'exang',
    'slope',
    'thal'
]

le = LabelEncoder()

df[categ] = df[categ].apply(le.fit_transform)


df['target'] = df['target'].map({
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1
})


cat_col = [
    'sex',
    'dataset',
    'cp',
    'fbs',
    'restecg',
    'exang',
    'slope',
    'ca',
    'thal',
    'target'
]


cont_col = [
    'age',
    'trestbps',
    'chol',
    'thalch',
    'oldpeak'
]


# ============================================================
# STANDARDIZATION
# ============================================================

ss = StandardScaler()

df[cont_col] = ss.fit_transform(df[cont_col])


# ============================================================
# COMBINE DATA
# ============================================================

new_df = pd.concat(
    [df[cat_col], df[cont_col]],
    axis=1
)

new_df = new_df.astype('float64')

new_df['ca'] = new_df['ca'].fillna(
    new_df['ca'].mean()
)


# ============================================================
# X AND Y
# ============================================================

X = new_df.drop(
    columns=['target']
)

y = new_df['target']


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# MODELS
# ============================================================

models = {

    'Logistic Regression':
        LogisticRegression(max_iter=1000),

    'SVM':
        SVC(),

    'KNN':
        KNeighborsClassifier(),

    'Decision Tree':
        DecisionTreeClassifier(
            random_state=42
        ),

    'Random Forest':
        RandomForestClassifier(
            random_state=42
        ),

    'Gradient Boosting':
        GradientBoostingClassifier(
            random_state=42
        ),

    'Naive Bayes':
        GaussianNB(),

    'Extra Trees':
        ExtraTreesClassifier(
            random_state=42
        ),

    'MLP':
        MLPClassifier(
            max_iter=1000,
            random_state=42
        )

}


# ============================================================
# MODEL COMPARISON
# ============================================================

result = {}

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    Accuracy = accuracy_score(
        y_test,
        y_pred
    )

    result[name] = Accuracy


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = max(
    result,
    key=result.get
)

best_model = models[
    best_model_name
]

best_model.fit(
    X_train,
    y_train
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"Selected Model: {best_model_name}"
    )

with col2:

    st.info(
        f"Best Accuracy: "
        f"{result[best_model_name] * 100:.1f}%"
    )

st.divider()


# ============================================================
# PATIENT PREDICTION
# ============================================================

st.subheader("Patient Prediction")

choice = st.radio(
    "Select prediction option:",
    [
        "Existing Patient",
        "New Patient"
    ]
)


# ============================================================
# EXISTING PATIENT
# ============================================================

if choice == "Existing Patient":

    st.subheader(
        "Existing Patient"
    )

    heart_disease_patient = st.number_input(
        "Enter patient index number",
        min_value=0,
        max_value=len(X) - 1,
        value=0,
        step=1
    )


    if st.button(
        "PREDICT EXISTING PATIENT"
    ):

        patient_data = X.iloc[
            [heart_disease_patient]
        ]

        prediction = best_model.predict(
            patient_data
        )


        if prediction[0] == 0.0:

            status = "NO HEART DISEASE"

            message = (
                "The model predicts that the "
                "patient does not have heart disease."
            )

            st.success(
                f"✓ {status}"
            )

            st.write(message)

        else:

            status = "HEART DISEASE"

            message = (
                "The model predicts that the "
                "patient has heart disease."
            )

            st.error(
                f"⚠️ {status}"
            )

            st.write(message)


        st.info(
            f"Patient Index: "
            f"{heart_disease_patient}"
        )


# ============================================================
# NEW PATIENT
# ============================================================

elif choice == "New Patient":

    st.subheader(
        "New Patient Information"
    )

    st.write(
        "Enter the patient's information below."
    )


    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    st.markdown(
        "### Categorical Features"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        sex = st.number_input(
            "Sex",
            value=0.0
        )

        dataset = st.number_input(
            "Dataset",
            value=0.0
        )

        cp = st.number_input(
            "CP",
            value=0.0
        )


    with col2:

        fbs = st.number_input(
            "FBS",
            value=0.0
        )

        restecg = st.number_input(
            "Rest ECG",
            value=0.0
        )

        exang = st.number_input(
            "Exang",
            value=0.0
        )


    with col3:

        slope = st.number_input(
            "Slope",
            value=0.0
        )

        ca = st.number_input(
            "CA",
            value=0.0
        )

        thal = st.number_input(
            "Thal",
            value=0.0
        )


    # --------------------------------------------------------
    # CONTINUOUS FEATURES
    # --------------------------------------------------------

    st.markdown(
        "### Continuous Features"
    )

    col1, col2 = st.columns(2)


    with col1:

        age = st.number_input(
            "Age",
            value=50.0
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            value=120.0
        )

        chol = st.number_input(
            "Cholesterol",
            value=200.0
        )


    with col2:

        thalch = st.number_input(
            "Maximum Heart Rate (thalch)",
            value=150.0
        )

        oldpeak = st.number_input(
            "Oldpeak",
            value=1.0
        )


    st.divider()


    # ========================================================
    # NEW PATIENT PREDICTION
    # ========================================================

    if st.button(
        "PREDICT NEW PATIENT"
    ):


        continuous_data = pd.DataFrame(

            [[
                age,
                trestbps,
                chol,
                thalch,
                oldpeak
            ]],

            columns=cont_col
        )


        scaled_continuous = ss.transform(
            continuous_data
        )


        scaled_continuous = pd.DataFrame(

            scaled_continuous,

            columns=cont_col
        )


        categorical_data = pd.DataFrame(

            [[
                sex,
                dataset,
                cp,
                fbs,
                restecg,
                exang,
                slope,
                ca,
                thal
            ]],

            columns=[
                'sex',
                'dataset',
                'cp',
                'fbs',
                'restecg',
                'exang',
                'slope',
                'ca',
                'thal'
            ]
        )


        patient_entry_df = pd.concat(

            [
                categorical_data,
                scaled_continuous
            ],

            axis=1
        )


        patient_entry_df = (
            patient_entry_df[X.columns]
        )


        prediction = best_model.predict(
            patient_entry_df
        )


        if prediction[0] == 0.0:

            status = "NO HEART DISEASE"

            message = (
                "The model predicts that the "
                "patient does not have heart disease."
            )

            st.success(
                f"✓ {status}"
            )

            st.write(message)

        else:

            status = "HEART DISEASE"

            message = (
                "The model predicts that the "
                "patient has heart disease."
            )

            st.error(
                f"⚠️ {status}"
            )

            st.write(message)


        st.info(
            "New patient information was entered "
            "into the prediction system."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Prediction generated using the trained "
    "machine-learning model."
)
