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
/* page background */
.stApp {background: linear-gradient(180deg,#f8fbff 0%, #ffffff 40%);}
/* section cards */
.block-card {background:#fff;border:1px solid #eef2f7;border-radius:16px;padding:18px;
  box-shadow:0 2px 12px rgba(16,24,40,0.06);}
/* titles */
h1, h2, h3 {font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu;}
/* badges */
.badge {display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;
  background:#eff6ff;color:#1e40af;border:1px solid #dbeafe;margin-left:6px;}
/* big result box */
.result-box {padding:22px;border-radius:16px;text-align:center;font-size:20px;font-weight:600;}
.good {background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;}
.bad {background:#fef2f2;color:#991b1b;border:1px solid #fecaca;}
/* buttons wide */
.stButton>button {border-radius:10px;padding:10px 16px;font-weight:600;}
/* small label */
.small {font-size:12px;color:#6b7280;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load model & age-scaler
# ---------------------------
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("model (2).pkl")
        age_scaler = joblib.load("scaler (1).pkl")  # <-- Age-only scaler
        return model, age_scaler
    except Exception as e:
        return None, None

model, age_scaler = load_model_and_scaler()

# Feature order (must match training exactly, excluding target 'Arthritis')
FEATURES = [
    "General_Health","Checkup","Exercise","Heart_Disease","Skin_Cancer","Other_Cancer",
    "Depression","Diabetes","Gender","Age_Category","Smoking_History","Age",
    "Polyuria","Polydipsia","sudden weight loss","weakness","Polyphagia","Genital thrush",
    "visual blurring","Itching","Irritability","delayed healing","muscle stiffness",
    "Alopecia","Obesity","Mental_Health_Status","Stress_Level","Online_Support_Usage",
    "address","guardian","schoolsup","activities","nursery","higher","internet","romantic"
]

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("## 🦴 Arthritis Predictor")
    st.markdown("Early-risk screening (for education only).")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("- Inputs match your dataset\n- Only **Age** is standardized\n- Model predicts **Arthritis** 0/1")
    st.markdown("---")
    st.markdown("**Tips**")
    st.caption("• Keep model.pkl & scaler.pkl in the same folder.\n• The scaler must be trained only on Age.\n• Columns must match training order.")

st.markdown("<h1>Arthritis Prediction <span class='badge'>Age scaled only</span></h1>", unsafe_allow_html=True)
st.caption("Fill in patient details. When you predict, a PDF summary becomes available to download.")

if model is None or age_scaler is None:
    st.error("Model or Scaler not found. Put **model.pkl** and **scaler.pkl** beside this file.")
    st.stop()

# ---------------------------
# UI Sections
# ---------------------------
st.markdown("<div class='block-card'>", unsafe_allow_html=True)
st.subheader("👤 General & Lifestyle")

c1, c2, c3, c4 = st.columns(4)
with c1:
    General_Health = st.slider("General Health (1-5)", 1, 5, 3)
    Checkup = st.selectbox("Regular Checkup", [0, 1])
    Exercise = st.selectbox("Exercise", [0, 1])
with c2:
    Gender = st.radio("Gender", ["Male", "Female"])
    Gender = 1 if Gender == "Male" else 0
    Age = st.slider("Age", 10, 100, 40)
    Age_Category = st.slider("Age Category", 1, 13, 5)
with c3:
    Smoking_History = st.selectbox("Smoking History", [0, 1])
    Online_Support_Usage = st.selectbox("Online Support Usage", [0, 1])
    address = st.selectbox("Address (0=Urban, 1=Rural)", [0, 1])
with c4:
    activities = st.selectbox("Activities", [0, 1])
    internet = st.selectbox("Internet Access", [0, 1])
    romantic = st.selectbox("Romantic Relationship", [0, 1])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='block-card' style='margin-top:14px;'>", unsafe_allow_html=True)
st.subheader("🏥 Medical History")
c5, c6, c7, c8 = st.columns(4)
with c5:
    Heart_Disease = st.selectbox("Heart Disease", [0, 1])
    Skin_Cancer = st.selectbox("Skin Cancer", [0, 1])
with c6:
    Other_Cancer = st.selectbox("Other Cancer", [0, 1])
    Depression = st.selectbox("Depression", [0, 1])
with c7:
    Diabetes = st.selectbox("Diabetes", [0, 1])
    Obesity = st.selectbox("Obesity", [0, 1])
with c8:
    Mental_Health_Status = st.slider("Mental Health (1-5)", 1, 5, 3)
    Stress_Level = st.slider("Stress Level (1-5)", 1, 5, 3)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='block-card' style='margin-top:14px;'>", unsafe_allow_html=True)
st.subheader("🧬 Symptoms")
c9, c10, c11, c12 = st.columns(4)
with c9:
    Polyuria = st.selectbox("Polyuria", [0, 1])
    Polydipsia = st.selectbox("Polydipsia", [0, 1])
with c10:
    sudden_weight_loss = st.selectbox("Sudden Weight Loss", [0, 1])
    weakness = st.selectbox("Weakness", [0, 1])
with c11:
    Polyphagia = st.selectbox("Polyphagia", [0, 1])
    Genital_thrush = st.selectbox("Genital Thrush", [0, 1])
with c12:
    visual_blurring = st.selectbox("Visual Blurring", [0, 1])
    Itching = st.selectbox("Itching", [0, 1])

c13, c14, c15 = st.columns(3)
with c13:
    Irritability = st.selectbox("Irritability", [0, 1])
with c14:
    delayed_healing = st.selectbox("Delayed Healing", [0, 1])
with c15:
    muscle_stiffness = st.selectbox("Muscle Stiffness", [0, 1])
    Alopecia = st.selectbox("Alopecia", [0, 1])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='block-card' style='margin-top:14px;'>", unsafe_allow_html=True)
st.subheader("👪 Support & Education")
d1, d2, d3, d4 = st.columns(4)
with d1:
    guardian = st.selectbox("Guardian Support", [0, 1])
    schoolsup = st.selectbox("School Support", [0, 1])
with d2:
    nursery = st.selectbox("Nursery Attended", [0, 1])
    higher = st.selectbox("Higher Education", [0, 1])
with d3:
    # placeholders kept for parity with dataset
    pass
with d4:
    pass
st.markdown("</div>", unsafe_allow_html=True)

# Build feature dict in EXACT order
features_dict = {
    "General_Health": General_Health,
    "Checkup": Checkup,
    "Exercise": Exercise,
    "Heart_Disease": Heart_Disease,
    "Skin_Cancer": Skin_Cancer,
    "Other_Cancer": Other_Cancer,
    "Depression": Depression,
    "Diabetes": Diabetes,
    "Gender": Gender,
    "Age_Category": Age_Category,
    "Smoking_History": Smoking_History,
    "Age": Age,  # placeholder; will be replaced by scaled value
    "Polyuria": Polyuria,
    "Polydipsia": Polydipsia,
    "sudden weight loss": sudden_weight_loss,
    "weakness": weakness,
    "Polyphagia": Polyphagia,
    "Genital thrush": Genital_thrush,
    "visual blurring": visual_blurring,
    "Itching": Itching,
    "Irritability": Irritability,
    "delayed healing": delayed_healing,
    "muscle stiffness": muscle_stiffness,
    "Alopecia": Alopecia,
    "Obesity": Obesity,
    "Mental_Health_Status": Mental_Health_Status,
    "Stress_Level": Stress_Level,
    "Online_Support_Usage": Online_Support_Usage,
    "address": address,
    "guardian": guardian,
    "schoolsup": schoolsup,
    "activities": activities,
    "nursery": nursery,
    "higher": higher,
    "internet": internet,
    "romantic": romantic
}

# ---------------------------
# Predict & PDF
# ---------------------------
col_run, col_pdf = st.columns([2,1])

with col_run:
    run = st.button("🚀 Predict Arthritis", use_container_width=True)

pred_label = None
proba = None
pdf_bytes = None

def predict(features: dict):
    # scale age only
    scaled_age = age_scaler.transform([[features["Age"]]])[0][0]
    features = {**features, "Age": scaled_age}
    # to DF in correct order
    X = pd.DataFrame([[features[k] for k in FEATURES]], columns=FEATURES)
    # predict
    yhat = model.predict(X)[0]
    # try probabilities if available
    p = None
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(X)[0][1])
    return int(yhat), p, scaled_age

if run:
    with st.spinner("Running model…"):
        # tiny progress bar for feel
        pb = st.progress(0)
        for i in range(1, 6):
            time.sleep(0.08)
            pb.progress(i*20)
        pred_label, proba, scaled_age_val = predict(features_dict)

    st.markdown("### 🩺 Prediction Result")
    if pred_label == 1:
        st.markdown("<div class='result-box bad'>⚠️ Arthritis Positive<br><span class='small'>Educational output. Consult a clinician.</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box good'>✅ Arthritis Negative<br><span class='small'>Maintain healthy habits.</span></div>", unsafe_allow_html=True)

    # nice metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Output", "Positive" if pred_label==1 else "Negative")
    m2.metric("Confidence", f"{proba*100:.1f}%" if proba is not None else "—")
    m3.metric("Scaled Age", f"{scaled_age_val:.3f}")

    # ---- Build PDF ----
    def build_pdf_bytes():
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        W, H = A4
        y = H - 2*cm

        def line(txt, dy=14, bold=False):
            nonlocal y
            if bold:
                c.setFont("Helvetica-Bold", 11)
            else:
                c.setFont("Helvetica", 10)
            c.drawString(2*cm, y, txt)
            y -= dy

        # Header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2*cm, y, "Arthritis Prediction Report")
        y -= 24
        c.setFont("Helvetica", 9)
        c.drawString(2*cm, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 18
        c.line(2*cm, y, W-2*cm, y)
        y -= 16

        # Result
        line("Result:", bold=True)
        result_text = "Arthritis Positive" if pred_label==1 else "Arthritis Negative"
        conf_text = f"Confidence: {proba*100:.1f}%" if proba is not None else "Confidence: N/A"
        line(f"- {result_text}")
        line(f"- {conf_text}")
        y -= 8

        # Inputs
        line("Inputs", bold=True); y -= 2
        for k in FEATURES:
            val = features_dict[k]
            # show raw age (not scaled) and also scaled
            if k == "Age":
                line(f"- Age: {val} (scaled in model)")
            else:
                line(f"- {k}: {val}")
            if y < 3*cm:
                c.showPage()
                y = H - 2*cm

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    pdf_bytes = build_pdf_bytes()

with col_pdf:
    if pdf_bytes:
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"arthritis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.caption("Generate a prediction to enable the PDF report.")
