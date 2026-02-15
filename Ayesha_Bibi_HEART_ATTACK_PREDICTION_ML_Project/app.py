# app.py

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_artifacts():
    obj = joblib.load("heart_attack.pkl")
    if isinstance(obj, dict):
        model = obj["model"]
        feature_names = obj.get("features", None)
    else:
        model = obj
        feature_names = None
    return model, feature_names

model, feature_names = load_artifacts()

# ---------------------------
# Title
# ---------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown("""
**Dataset:** Cleveland Heart Disease Dataset  
**Type:** Random Forest Classifier  
**Deployment:** Streamlit Web Application
""")
st.divider()

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("🧑‍⚕️ Patient Clinical Data")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)

trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

restecg = st.sidebar.selectbox(
    "Resting ECG",
    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)

thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0)

slope = st.sidebar.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping", "Flat", "Downsloping"]
)

ca = st.sidebar.slider("Number of Major Vessels (0–3)", 0, 3, 0)

thal = st.sidebar.selectbox(
    "Thalassemia",
    ["Normal", "Fixed Defect", "Reversible Defect"]
)

# ---------------------------
# Encoding Inputs
# ---------------------------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_map = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

# ---------------------------
# Prepare Input Vector (LOCKED ORDER)
# ---------------------------
X = np.array([[
    age,
    sex,
    cp_map[cp],
    trestbps,
    chol,
    fbs,
    restecg_map[restecg],
    thalach,
    exang,
    oldpeak,
    slope_map[slope],
    ca,
    thal_map[thal]
]])

# ---------------------------
# Debug: Show input values
# ---------------------------
st.write("🧪 Debug: Input feature values", X)

# ---------------------------
# Prediction
# ---------------------------
st.subheader("🔍 Prediction Result")

if st.button("🚀 Predict Heart Attack"):
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    if pred == 1:
        st.error("⚠️ HIGH RISK: Heart Attack Detected")
    else:
        st.success("✅ LOW RISK: No Heart Attack Detected")

    st.metric("Heart Attack Probability", f"{prob * 100:.2f}%")
    st.progress(float(prob))

    # Probability Bar Chart
    fig, ax = plt.subplots()
    ax.bar(["No Attack", "Attack"], [1 - prob, prob], color=["green", "red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

# ---------------------------
# Model Explanation
# ---------------------------
st.divider()
st.subheader("📘 Model Details")

st.markdown("""
- Trained on **Cleveland Heart Disease Dataset**
- Random Forest classifier (300 trees, max depth 10)
- Model saved with **joblib** as `heart_attack.pkl`
- Feature order locked to match input UI
- Fully interactive Streamlit deployment
""")

# ---------------------------
# Disclaimer
# ---------------------------
st.warning("""
⚠️ Educational use only.  
Not a real medical diagnostic tool.
""")

st.markdown("---")
st.markdown("👩‍💻 **Developed by Ayesha Bibi** | Machine Learning Project")
