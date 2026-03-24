import streamlit as st
import numpy as np
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="MedAffect",
    layout="wide",
    page_icon="🧠"
)

# ------------------ STYLING ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #00e5ff;
    text-shadow: 0px 0px 20px #00e5ff;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* Glass Cards */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,229,255,0.1);
    text-align: center;
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 30px rgba(0,229,255,0.4);
}

/* Progress bars */
.progress-bar {
    height: 12px;
    border-radius: 10px;
    background: #1e293b;
    overflow: hidden;
    margin-top: 10px;
}

.progress-fill {
    height: 12px;
    border-radius: 10px;
}

/* Section headers */
.section {
    font-size: 22px;
    margin-top: 20px;
    margin-bottom: 10px;
    color: #38bdf8;
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
st.markdown('<div class="section">🧍 Patient Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 30)
    sex = st.selectbox("Sex", ["Female", "Male"])
    bmi = st.slider("BMI", 15.0, 40.0, 22.0)

with col2:
    diabetes = st.selectbox("Diabetes", [0, 1])
    alt = st.slider("ALT Level", 10.0, 100.0, 30.0)
    egfr = st.slider("eGFR", 30.0, 120.0, 80.0)

st.markdown('<div class="section">💊 Drug Properties</div>', unsafe_allow_html=True)

drug_hepatotoxic = st.selectbox("Hepatotoxic Drug", [0, 1])
drug_renal = st.selectbox("Renal Impact Drug", [0, 1])

# Convert inputs
sex = 1 if sex == "Male" else 0

input_data = np.array([age, sex, bmi, diabetes, alt, egfr, drug_hepatotoxic, drug_renal])

# ------------------ PREDICTION ------------------
if st.button("🚀 Predict Risk"):

    input_scaled = scaler.transform([input_data])

    liver = liver_model.predict_proba(input_scaled)[0][1]
    kidney = kidney_model.predict_proba(input_scaled)[0][1]
    heart = heart_model.predict_proba(input_scaled)[0][1]

    confidence = np.mean([liver, kidney, heart])

    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    # Color logic
    def get_color(val):
        if val > 0.6:
            return "#ff4d4d"
        elif val > 0.3:
            return "#facc15"
        else:
            return "#22c55e"

    # Render card
    def render_card(title, value, emoji):
        color = get_color(value)
        percent = int(value * 100)

        st.markdown(f"""
        <div class="card">
            <h3>{emoji} {title}</h3>
            <h1 style="color:{color};">{percent}%</h1>
            <div class="progress-bar">
                <div class="progress-fill" style="width:{percent}%; background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        render_card("Liver Risk", liver, "🧠")

    with col2:
        render_card("Kidney Risk", kidney, "🧬")

    with col3:
        render_card("Heart Risk", heart, "❤️")

    st.markdown("---")

    # Confidence Card
    conf_percent = int(confidence * 100)
    st.markdown(f"""
    <div class="card">
        <h3>📌 Overall Confidence</h3>
        <h1 style="color:#00e5ff;">{conf_percent}%</h1>
        <div class="progress-bar">
            <div class="progress-fill" style="width:{conf_percent}%; background:#00e5ff;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
