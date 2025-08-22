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
# Friendly label mappings
# ---------------------------
YES_NO = {"Yes": 1, "No": 0}
YES_NO_INV = {v: k for k, v in YES_NO.items()}

GENDER_MAP = {"Male": 1, "Female": 0}
GENDER_INV = {v: k for k, v in GENDER_MAP.items()}

ADDRESS_MAP = {"Urban": 0, "Rural": 1}
ADDRESS_INV = {v: k for k, v in ADDRESS_MAP.items()}

# Helper to render a Yes/No selectbox and return numeric value plus label
def yn_input(label, help_text=None, key=None, default="No"):
    lbl = st.selectbox(label, list(YES_NO.keys()), index=0 if default=="No" else 1, help=help_text, key=key)
    return YES_NO[lbl], lbl

# ---------------------------
# Load model & age-scaler
# ---------------------------
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("model (2).pkl")       # keep your filenames
        age_scaler = joblib.load("scaler (1).pkl") # age-only scaler
        return model, age_scaler
    except Exception:
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
# UI Sections (with human-friendly labels)
# ---------------------------
# We'll store both numeric values (for model) and human labels (for PDF)
ui_labels = {}     # human-readable values for the PDF
num_values = {}    # numeric values for the model

st.markdown("<div class='block-card'>", unsafe_allow_html=True)
st.subheader("👤 General & Lifestyle")

c1, c2, c3, c4 = st.columns(4)
with c1:
    General_Health = st.slider("General Health (1=Poor, 5=Excellent)", 1, 5, 3, help="Self-rated overall health")
    Checkup_num, Checkup_lbl = yn_input("Regular Checkup?", "Routine medical checkups in last year")
    Exercise_num, Exercise_lbl = yn_input("Exercise Habit?", "Regular physical activity at least 3x/week")

with c2:
    Gender_lbl = st.radio("Gender", list(GENDER_MAP.keys()), index=0)
    Gender_num = GENDER_MAP[Gender_lbl]
    Age = st.slider("Age (years)", 10, 100, 40)
    Age_Category = st.slider("Age Category (1-13)", 1, 13, 5, help="Categorized age group used in training")

with c3:
    Smoking_num, Smoking_lbl = yn_input("Smoking History?", "Ever smoked regularly?")
    OnlineSupport_num, OnlineSupport_lbl = yn_input("Online Support Usage?", "Uses online health/community support?")
    address_lbl = st.selectbox("Address Type", list(ADDRESS_MAP.keys()), help="Residence type")
    address_num = ADDRESS_MAP[address_lbl]

with c4:
    activities_num, activities_lbl = yn_input("Physical Activities?", "Participates in physical activities")
    internet_num, internet_lbl = yn_input("Internet Access at Home?", "Has reliable internet at home")
    romantic_num, romantic_lbl = yn_input("In a Romantic Relationship?", "Currently in a romantic relationship")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='block-card' style='margin-top:14px;'>", unsafe_allow_html=True)
st.subheader("🏥 Medical History")
c5, c6, c7, c8 = st.columns(4)
with c5:
    Heart_num, Heart_lbl = yn_input("Heart Disease?", "Diagnosed with heart disease")
    SkinCancer_num, SkinCancer_lbl = yn_input("Skin Cancer?", "History of skin cancer")
with c6:
    OtherCancer_num, OtherCancer_lbl = yn_input("Other Cancer?", "Any non-skin cancer diagnoses")
    Depression_num, Depression_lbl = yn_input("Depression?", "Diagnosed depression")
with c7:
    Diabetes_num, Diabetes_lbl = yn_input("Diabetes?", "Diagnosed diabetes")
    Obesity_num, Obesity_lbl = yn_input("Obesity?", "Clinically classified as obese")
with c8:
    Mental_Health_Status = st.slider("Mental Health Score (1-5)", 1, 5, 3, help="1=Very Poor, 5=Excellent")
    Stress_Level = st.slider("Stress Level (1-5)", 1, 5, 3, help="1=Low, 5=High")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='block-card' style='margin-top:14px;'>", unsafe_allow_html=True)
st.subheader("🧬 Symptoms")
c9, c10, c11, c12 = st.columns(4)
with c9:
    Polyuria_num, Polyuria_lbl = yn_input("Polyuria?", "Frequent urination")
    Polydipsia_num, Polydipsia_lbl = yn_input("Polydipsia?", "Excessive thirst")
with c10:
    sudden_weight_loss_num, sudden_weight_loss_lbl = yn_input("Sudden Weight Loss?", "Unintentional recent weight loss")
    weakness_num, weakness_lbl = yn_input("Weakness?", "Generalized weakness")
with c11:
    Polyphagia_num, Polyphagia_lbl = yn_input("Polyphagia?", "Excessive hunger")
    Genital_thrush_num, Genital_thrush_lbl = yn_input("Genital Thrush?", "Recurrent yeast infection")
with c12:
    visual_blurring_num, visual_blurring_lbl = yn_input("Visual Blurring?", "Blurred vision")
    Itching_num, Itching_lbl = yn_input("Itching?", "Persistent itching")

c13, c14, c15 = st.columns(3)
with c13:
    Irritability_num, Irritability_lbl = yn_input("Irritability?", "Easily annoyed or frustrated")
with c14:
    delayed_healing_num, delayed_healing_lbl = yn_input("Delayed Healing?", "Wounds take longer to heal")
with c15:
    muscle_stiffness_num, muscle_stiffness_lbl = yn_input("Muscle Stiffness?", "Persistent stiffness")
    Alopecia_num, Alopecia_lbl = yn_input("Alopecia?", "Hair loss")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='block-card' style='margin-top:14px;'>", unsafe_allow_html=True)
st.subheader("👪 Support & Education")
d1, d2, d3, d4 = st.columns(4)
with d1:
    guardian_num, guardian_lbl = yn_input("Guardian Support?", "Support from parent/guardian")
    schoolsup_num, schoolsup_lbl = yn_input("School Support?", "Received additional school support")
with d2:
    nursery_num, nursery_lbl = yn_input("Nursery Attended?", "Attended pre-school/nursery")
    higher_num, higher_lbl = yn_input("Higher Education?", "Pursued higher education")
with d3:
    pass
with d4:
    pass
st.markdown("</div>", unsafe_allow_html=True)

# Collect human-readable labels for PDF
ui_labels.update({
    "General_Health": f"{General_Health}",
    "Checkup": Checkup_lbl,
    "Exercise": Exercise_lbl,
    "Heart_Disease": Heart_lbl,
    "Skin_Cancer": SkinCancer_lbl,
    "Other_Cancer": OtherCancer_lbl,
    "Depression": Depression_lbl,
    "Diabetes": Diabetes_lbl,
    "Gender": Gender_lbl,
    "Age_Category": f"{Age_Category}",
    "Smoking_History": Smoking_lbl,
    "Age": f"{Age}",
    "Polyuria": Polyuria_lbl,
    "Polydipsia": Polydipsia_lbl,
    "sudden weight loss": sudden_weight_loss_lbl,
    "weakness": weakness_lbl,
    "Polyphagia": Polyphagia_lbl,
    "Genital thrush": Genital_thrush_lbl,
    "visual blurring": visual_blurring_lbl,
    "Itching": Itching_lbl,
    "Irritability": Irritability_lbl,
    "delayed healing": delayed_healing_lbl,
    "muscle stiffness": muscle_stiffness_lbl,
    "Alopecia": Alopecia_lbl,
    "Obesity": Obesity_lbl,
    "Mental_Health_Status": f"{Mental_Health_Status}",
    "Stress_Level": f"{Stress_Level}",
    "Online_Support_Usage": OnlineSupport_lbl,
    "address": address_lbl,
    "guardian": guardian_lbl,
    "schoolsup": schoolsup_lbl,
    "activities": activities_lbl,
    "nursery": nursery_lbl,
    "higher": higher_lbl,
    "internet": internet_lbl,
    "romantic": romantic_lbl
})

# Build numeric features for the model
num_values.update({
    "General_Health": General_Health,
    "Checkup": YES_NO[Checkup_lbl],
    "Exercise": YES_NO[Exercise_lbl],
    "Heart_Disease": YES_NO[Heart_lbl],
    "Skin_Cancer": YES_NO[SkinCancer_lbl],
    "Other_Cancer": YES_NO[OtherCancer_lbl],
    "Depression": YES_NO[Depression_lbl],
    "Diabetes": YES_NO[Diabetes_lbl],
    "Gender": GENDER_MAP[Gender_lbl],
    "Age_Category": Age_Category,
    "Smoking_History": YES_NO[Smoking_lbl],
    "Age": Age,  # raw; will be replaced by scaled
    "Polyuria": YES_NO[Polyuria_lbl],
    "Polydipsia": YES_NO[Polydipsia_lbl],
    "sudden weight loss": YES_NO[sudden_weight_loss_lbl],
    "weakness": YES_NO[weakness_lbl],
    "Polyphagia": YES_NO[Polyphagia_lbl],
    "Genital thrush": YES_NO[Genital_thrush_lbl],
    "visual blurring": YES_NO[visual_blurring_lbl],
    "Itching": YES_NO[Itching_lbl],
    "Irritability": YES_NO[Irritability_lbl],
    "delayed healing": YES_NO[delayed_healing_lbl],
    "muscle stiffness": YES_NO[muscle_stiffness_lbl],
    "Alopecia": YES_NO[Alopecia_lbl],
    "Obesity": YES_NO[Obesity_lbl],
    "Mental_Health_Status": Mental_Health_Status,
    "Stress_Level": Stress_Level,
    "Online_Support_Usage": YES_NO[OnlineSupport_lbl],
    "address": ADDRESS_MAP[address_lbl],
    "guardian": YES_NO[guardian_lbl],
    "schoolsup": YES_NO[schoolsup_lbl],
    "activities": YES_NO[activities_lbl],
    "nursery": YES_NO[nursery_lbl],
    "higher": YES_NO[higher_lbl],
    "internet": YES_NO[internet_lbl],
    "romantic": YES_NO[romantic_lbl]
})

# ---------------------------
# Predict & PDF
# ---------------------------
col_run, col_pdf = st.columns([2,1])
with col_run:
    run = st.button("🚀 Predict Arthritis", use_container_width=True)

pred_label = None
proba = None
pdf_bytes = None

def predict(numeric_features: dict):
    # scale age only
    raw_age = numeric_features["Age"]
    scaled_age = age_scaler.transform([[raw_age]])[0][0]
    feats_scaled_age = {**numeric_features, "Age": scaled_age}

    # to DF in correct order
    X = pd.DataFrame([[feats_scaled_age[k] for k in FEATURES]], columns=FEATURES)
    # predict
    yhat = model.predict(X)[0]
    # probability if available
    p = None
    if hasattr(model, "predict_proba"):
        try:
            p = float(model.predict_proba(X)[0][1])
        except Exception:
            p = None
    return int(yhat), p, raw_age, scaled_age

if run:
    with st.spinner("Running model…"):
        pb = st.progress(0)
        for i in range(1, 6):
            time.sleep(0.08)
            pb.progress(i*20)
        pred_label, proba, raw_age_val, scaled_age_val = predict(num_values)

    st.markdown("### 🩺 Prediction Result")
    if pred_label == 1:
        st.markdown("<div class='result-box bad'>⚠️ Arthritis Positive<br><span class='small'>Educational output. Consult a clinician.</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box good'>✅ Arthritis Negative<br><span class='small'>Maintain healthy habits.</span></div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Model Output", "Positive" if pred_label==1 else "Negative")
    m2.metric("Confidence", f"{proba*100:.1f}%" if proba is not None else "N/A")
    m3.metric("Scaled Age", f"{scaled_age_val:.3f}")

    # ---- Build PDF ----
    def build_pdf_bytes():
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        W, H = A4
        y = H - 2*cm

        def line(txt, dy=14, bold=False):
            nonlocal y
            c.setFont("Helvetica-Bold" if bold else "Helvetica", 10 if not bold else 11)
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
        line(f"- Age: {raw_age_val} (scaled: {scaled_age_val:.3f})")
        y -= 8

        # Inputs (human-friendly)
        line("Inputs (Human-Readable)", bold=True); y -= 2
        for k in FEATURES:
            display_val = ui_labels.get(k, "—")
            # we already printed Age w/ scaled value above; show age label as well for completeness
            if k == "Age":
                line(f"- Age: {display_val} (scaled in model)")
            else:
                line(f"- {k}: {display_val}")
            if y < 3*cm:
                c.showPage()
                y = H - 2*cm
                line("Inputs (cont.)", bold=True)

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
