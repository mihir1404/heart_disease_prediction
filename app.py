import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
model = load_model("heart_ann_model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.write("Enter your health information to see if you are at risk of heart disease.")

age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–Typical angina, 1–Atypical, 2–Non-anginal, 3–Asymptomatic)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG Result (0–Normal, 1–ST-T abnormality, 2–Left ventricular hypertrophy)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–Upsloping, 1–Flat, 2–Downsloping)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0–Normal, 1–Fixed defect, 2–Reversible defect)", [0, 1, 2])

# Convert inputs to model format
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Standardize input using the saved scaler
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    result = (prediction > 0.5).astype("int32")[0][0]

    if result == 1:
        st.error("⚠️ The model predicts a **risk of heart disease**.")
    else:
        st.success("✅ The model predicts **no risk** of heart disease.")