import streamlit as st
import numpy as np
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="MedAffect",
    layout="wide",
    page_icon="🧠"
)

# ------------------ DARK THEME ------------------
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #00d4ff;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #9aa4b2;
        margin-bottom: 30px;
    }
    .card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="title">MedAffect</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Driven Drug Risk Prediction System</div>', unsafe_allow_html=True)

# ------------------ LOAD MODELS ------------------
liver_model = joblib.load("liver_model.pkl")
kidney_model = joblib.load("kidney_model.pkl")
heart_model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------ INPUT SECTION ------------------
st.markdown("### 🧍 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 30)
    sex = st.selectbox("Sex", ["Female", "Male"])
    bmi = st.slider("BMI", 15.0, 40.0, 22.0)

with col2:
    diabetes = st.selectbox("Diabetes", [0, 1])
    alt = st.slider("ALT Level", 10.0, 100.0, 30.0)
    egfr = st.slider("eGFR", 30.0, 120.0, 80.0)

st.markdown("### 💊 Drug Properties")

drug_hepatotoxic = st.selectbox("Hepatotoxic Drug", [0, 1])
drug_renal = st.selectbox("Renal Impact Drug", [0, 1])

# Convert
sex = 1 if sex == "Male" else 0

input_data = np.array([age, sex, bmi, diabetes, alt, egfr, drug_hepatotoxic, drug_renal])

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict Risk"):

    input_scaled = scaler.transform([input_data])

    liver = liver_model.predict_proba(input_scaled)[0][1]
    kidney = kidney_model.predict_proba(input_scaled)[0][1]
    heart = heart_model.predict_proba(input_scaled)[0][1]

    confidence = np.mean([liver, kidney, heart])

    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    def risk_color(val):
        if val > 0.6:
            return "🔴 High"
        elif val > 0.3:
            return "🟡 Moderate"
        else:
            return "🟢 Low"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
        <h3>🧠 Liver</h3>
        <h2>{liver:.2f}</h2>
        <p>{risk_color(liver)}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
        <h3>🧬 Kidney</h3>
        <h2>{kidney:.2f}</h2>
        <p>{risk_color(kidney)}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
        <h3>❤️ Heart</h3>
        <h2>{heart:.2f}</h2>
        <p>{risk_color(heart)}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"""
    <div class="card">
    <h3>📌 Overall Confidence</h3>
    <h2>{confidence:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)
