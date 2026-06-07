import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("diabetes_knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page Configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Sidebar
st.sidebar.title("About")

st.sidebar.markdown("""
### Diabetes Prediction System

This project predicts the likelihood of diabetes using Machine Learning algorithms.

### Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- XGBoost

### Best Model
**KNN**

### Test Accuracy
**76.62%**
""")

# Replace with your actual GitHub repo link
st.sidebar.markdown(
    "[View Source Code](https://github.com/pullapravalika/Diabetes-Prediction-ML)"
)

# Main Title
st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details below to predict diabetes risk.")

# Create Columns
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0, max_value=20)
    glucose = st.number_input("Glucose", min_value=40, max_value=300)
    bp = st.number_input("Blood Pressure", min_value=40, max_value=200)
    skinthickness = st.number_input("Skin Thickness", min_value=0, max_value=100)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0)
    age = st.number_input("Age", min_value=1, max_value=120)

# Prediction Button
if st.button("Predict"):

    if (
        preg == 0 and
        glucose == 40 and
        bp == 40 and
        skinthickness == 0 and
        insulin == 0 and
        bmi == 10.0 and
        dpf == 0.0 and
        age == 1
    ):
        st.warning("⚠️ Please enter patient details before prediction.")

    else:
        data = np.array([[
            preg,
            glucose,
            bp,
            skinthickness,
            insulin,
            bmi,
            dpf,
            age
        ]])

        data = scaler.transform(data)

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("🔴 The person is likely to have diabetes.")
        else:
            st.success("🟢 The person is unlikely to have diabetes.")

# Model Comparison Section
st.subheader("📊 Model Performance Comparison")

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "SVM",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "KNN"
    ],
    "Train Accuracy": [
        78.0,
        78.8,
        79.8,
        92.0,
        81.8,
        79.6
    ],
    "Test Accuracy": [
        70.0,
        70.8,
        73.4,
        75.0,
        72.7,
        76.6
    ]
})

st.dataframe(results)

# Optional chart
st.bar_chart(
    results.set_index("Model")[["Train Accuracy", "Test Accuracy"]]
)

st.success("🏆 Best Model: KNN (Test Accuracy: 76.62%)")