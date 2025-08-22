import io
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# ---------------------------
# App Config & Style
# ---------------------------
st.set_page_config(page_title="Arthritis Predictor", page_icon="🦴", layout="wide")

# Soft theming via CSS
st.markdown("""
<style>
.stApp {background: linear-gradient(180deg,#f8fbff 0%, #ffffff 40%);}
.block-card {background:#fff;border:1px solid #eef2f7;border-radius:16px;padding:18px;
  box-shadow:0 2px 12px rgba(16,24,40,0.06);}
h1, h2, h3 {font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu;}
.badge {display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;
  background:#eff6ff;color:#1e40af;border:1px solid #dbeafe;margin-left:6px;}
.result-box {padding:22px;border-radius:16px;text-align:center;font-size:20px;font-weight:600;}
.good {background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;}
.bad {background:#fef2f2;color:#991b1b;border:1px solid #fecaca;}
.stButton>button {border-radius:10px;padding:10px 16px;font-weight:600;}
.small {font-size:12px;color:#6b7280;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load model & scaler
# ---------------------------
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")   # trained only on Age column
        return model, scaler
    except:
        return None, None

model, age_scaler = load_model_and_scaler()

FEATURES = [
    "General_Health","Checkup","Exercise","Heart_Disease","Skin_Cancer","Other_Cancer",
    "Depression","Diabetes","Gender","Age_Category","Smoking_History","Age",
    "Polyuria","Polydipsia","sudden weight loss","weakness","Polyphagia","Genital thrush",
    "visual blurring","Itching","Irritability","delayed healing","muscle stiffness",
    "Alopecia","Obesity","Mental_Health_Status","Stress_Level","Online_Support_Usage",
    "address","guardian","schoolsup","activities","nursery","higher","internet","romantic"
]

# ---------------------------
# Sidebar Info
# ---------------------------
with st.sidebar:
    st.markdown("## 🦴 Arthritis Predictor")
    st.markdown("Early-risk screening (Educational use only).")
    st.info("Upload trained **model.pkl** and **scaler.pkl** in same folder.")
    st.caption("• Age is standardized using the scaler\n• All inputs must match training columns")

# ---------------------------
# Main Title
# ---------------------------
st.markdown("<h1>Arthritis Prediction <span class='badge'>Interactive Form</span></h1>", unsafe_allow_html=True)
st.caption("Fill in the patient details below. Get prediction & PDF report instantly.")

if model is None or age_scaler is None:
    st.error("Model or Scaler not found. Place **model.pkl** and **scaler.pkl** in the same folder.")
    st.stop()

# ---------------------------
# Input Form
# ---------------------------
with st.form("prediction_form"):

    st.markdown("<div class='block-card'>", unsafe_allow_html=True)
    st.subheader("👤 General & Lifestyle Details")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        General_Health = st.slider("General Health Rating", 1, 5, 3, help="1=Poor, 5=Excellent")
        Checkup = st.selectbox("Regular Checkup?", [0, 1], help="1=Yes, 0=No")
        Exercise = st.selectbox("Exercise Habit?", [0, 1], help="1=Yes, 0=No")
    with c2:
        Gender = st.radio("Gender", ["Male", "Female"], help="Select patient gender")
        Gender = 1 if Gender == "Male" else 0
        Age = st.slider("Age (Years)", 10, 100, 40)
        Age_Category = st.slider("Age Category (1-13)", 1, 13, 5, help="Categorized age group")
    with c3:
        Smoking_History = st.selectbox("Smoking History?", [0, 1])
        Online_Support_Usage = st.selectbox("Uses Online Support?", [0, 1])
        address = st.selectbox("Address Type", [0, 1], help="0=Urban, 1=Rural")
    with c4:
        activities = st.selectbox("Physical Activities?", [0, 1])
        internet = st.selectbox("Internet Access?", [0, 1])
        romantic = st.selectbox("Romantic Relationship?", [0, 1])
    st.markdown("</div>", unsafe_allow_html=True)

    # Medical History
    st.markdown("<div class='block-card' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.subheader("🏥 Medical History")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        Heart_Disease = st.selectbox("Heart Disease?", [0, 1])
        Skin_Cancer = st.selectbox("Skin Cancer?", [0, 1])
    with c6:
        Other_Cancer = st.selectbox("Other Cancer?", [0, 1])
        Depression = st.selectbox("Depression?", [0, 1])
    with c7:
        Diabetes = st.selectbox("Diabetes?", [0, 1])
        Obesity = st.selectbox("Obesity?", [0, 1])
    with c8:
        Mental_Health_Status = st.slider("Mental Health Score", 1, 5, 3)
        Stress_Level = st.slider("Stress Level", 1, 5, 3)
    st.markdown("</div>", unsafe_allow_html=True)

    # Symptoms
    st.markdown("<div class='block-card' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.subheader("🧬 Symptoms")
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        Polyuria = st.selectbox("Polyuria?", [0, 1])
        Polydipsia = st.selectbox("Polydipsia?", [0, 1])
    with c10:
        sudden_weight_loss = st.selectbox("Sudden Weight Loss?", [0, 1])
        weakness = st.selectbox("Weakness?", [0, 1])
    with c11:
        Polyphagia = st.selectbox("Polyphagia?", [0, 1])
        Genital_thrush = st.selectbox("Genital Thrush?", [0, 1])
    with c12:
        visual_blurring = st.selectbox("Visual Blurring?", [0, 1])
        Itching = st.selectbox("Itching?", [0, 1])

    c13, c14, c15 = st.columns(3)
    with c13:
        Irritability = st.selectbox("Irritability?", [0, 1])
    with c14:
        delayed_healing = st.selectbox("Delayed Healing?", [0, 1])
    with c15:
        muscle_stiffness = st.selectbox("Muscle Stiffness?", [0, 1])
        Alopecia = st.selectbox("Alopecia?", [0, 1])
    st.markdown("</div>", unsafe_allow_html=True)

    # Support & Education
    st.markdown("<div class='block-card' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.subheader("👪 Support & Education")
    d1, d2 = st.columns(2)
    with d1:
        guardian = st.selectbox("Guardian Support?", [0, 1])
        schoolsup = st.selectbox("School Support?", [0, 1])
    with d2:
        nursery = st.selectbox("Attended Nursery?", [0, 1])
        higher = st.selectbox("Higher Education?", [0, 1])
    st.markdown("</div>", unsafe_allow_html=True)

    # Submit Button
    submitted = st.form_submit_button("🚀 Predict Arthritis")

# ---------------------------
# Prediction Logic
# ---------------------------
features_dict = {
    "General_Health": General_Health, "Checkup": Checkup, "Exercise": Exercise,
    "Heart_Disease": Heart_Disease, "Skin_Cancer": Skin_Cancer, "Other_Cancer": Other_Cancer,
    "Depression": Depression, "Diabetes": Diabetes, "Gender": Gender,
    "Age_Category": Age_Category, "Smoking_History": Smoking_History, "Age": Age,
    "Polyuria": Polyuria, "Polydipsia": Polydipsia, "sudden weight loss": sudden_weight_loss,
    "weakness": weakness, "Polyphagia": Polyphagia, "Genital thrush": Genital_thrush,
    "visual blurring": visual_blurring, "Itching": Itching, "Irritability": Irritability,
    "delayed healing": delayed_healing, "muscle stiffness": muscle_stiffness,
    "Alopecia": Alopecia, "Obesity": Obesity, "Mental_Health_Status": Mental_Health_Status,
    "Stress_Level": Stress_Level, "Online_Support_Usage": Online_Support_Usage,
    "address": address, "guardian": guardian, "schoolsup": schoolsup, "activities": activities,
    "nursery": nursery, "higher": higher, "internet": internet, "romantic": romantic
}

def predict(features):
    raw_age = features["Age"]
    scaled_age = age_scaler.transform([[raw_age]])[0][0]
    features["Age"] = scaled_age
    X = pd.DataFrame([[features[k] for k in FEATURES]], columns=FEATURES)
    yhat = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
    return int(yhat), prob, raw_age, scaled_age

# ---------------------------
# Run Prediction
# ---------------------------
if submitted:
    with st.spinner("Running prediction..."):
        time.sleep(1)
        pred_label, proba, raw_age, scaled_age = predict(features_dict)

    st.markdown("### 🩺 Prediction Result")
    if pred_label == 1:
        st.markdown("<div class='result-box bad'>⚠️ Arthritis Positive<br><span class='small'>Consult a medical professional.</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box good'>✅ Arthritis Negative<br><span class='small'>Keep up healthy habits!</span></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", "Positive" if pred_label else "Negative")
    col2.metric("Confidence", f"{proba*100:.1f}%" if proba else "N/A")
    col3.metric("Scaled Age", f"{scaled_age:.3f}")

    # PDF Generation
    def build_pdf():
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        W, H = A4
        y = H - 2*cm

        def line(txt, bold=False):
            nonlocal y
            c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
            c.drawString(2*cm, y, txt)
            y -= 14

        c.setFont("Helvetica-Bold", 16)
        c.drawString(2*cm, y, "Arthritis Prediction Report")
        y -= 24
        c.setFont("Helvetica", 9)
        c.drawString(2*cm, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 18
        c.line(2*cm, y, W-2*cm, y)
        y -= 16

        line("Prediction Summary", bold=True)
        line(f"- Result: {'Positive' if pred_label else 'Negative'}")
        line(f"- Confidence: {proba*100:.1f}%" if proba else "- Confidence: N/A")
        line(f"- Raw Age: {raw_age}, Scaled Age: {scaled_age:.3f}")
        y -= 10
        line("Inputs Used", bold=True)

        for k in FEATURES:
            line(f"- {k}: {features_dict[k]}")
            if y < 3*cm:
                c.showPage(); y = H - 2*cm

        c.showPage(); c.save(); buf.seek(0)
        return buf.getvalue()

    pdf_data = build_pdf()
    st.download_button("📄 Download PDF Report", data=pdf_data,
                       file_name=f"arthritis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                       mime="application/pdf")
