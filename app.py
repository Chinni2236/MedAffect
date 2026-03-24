import streamlit as st
import numpy as np
import joblib

# Load models
liver_model = joblib.load("liver_model.pkl")
kidney_model = joblib.load("kidney_model.pkl")
heart_model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("MedAffect - Drug Risk Prediction")

# Inputs
age = st.slider("Age", 18, 80)
sex = st.selectbox("Sex", ["Female", "Male"])
bmi = st.slider("BMI", 15.0, 40.0)
diabetes = st.selectbox("Diabetes", [0,1])
alt = st.slider("ALT Level", 10.0, 100.0)
egfr = st.slider("eGFR", 30.0, 120.0)
drug_hepatotoxic = st.selectbox("Hepatotoxic Drug", [0,1])
drug_renal = st.selectbox("Renal Drug", [0,1])

sex = 1 if sex == "Male" else 0

input_data = np.array([age, sex, bmi, diabetes, alt, egfr, drug_hepatotoxic, drug_renal])

if st.button("Predict"):
    input_scaled = scaler.transform([input_data])

    liver = liver_model.predict_proba(input_scaled)[0][1]
    kidney = kidney_model.predict_proba(input_scaled)[0][1]
    heart = heart_model.predict_proba(input_scaled)[0][1]

    confidence = np.mean([liver, kidney, heart])

    st.subheader("Results")

    st.write(f"Liver Risk: {liver:.2f}")
    st.write(f"Kidney Risk: {kidney:.2f}")
    st.write(f"Heart Risk: {heart:.2f}")
    st.write(f"Confidence Score: {confidence:.2f}")