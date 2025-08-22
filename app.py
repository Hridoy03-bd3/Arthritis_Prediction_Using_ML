import io
import time
import joblib
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

# Modern CSS styling
st.markdown("""
<style>
/* Page background */
.stApp {background: linear-gradient(180deg,#f0f4ff 0%, #ffffff 50%);}

/* Section cards */
.block-card {
    background:#fff;
    border:1px solid #d1d9e6;
    border-radius:20px;
    padding:20px;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
    margin-bottom:16px;
}

/* Section titles */
h1, h2, h3 {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight:600;}

/* Badges */
.badge {
    display:inline-block;
    padding:4px 12px;
    border-radius:999px;
    font-size:12px;
    background:#e0e7ff;
    color:#1e40af;
    border:1px solid #c7d2fe;
    margin-left:6px;
}

/* Big result box */
.result-box {
    padding:24px;
    border-radius:16px;
    text-align:center;
    font-size:20px;
    font-weight:700;
}
.good {background:#e6fffa;color:#065f46;border:1px solid #34d399;}
.bad {background:#ffe4e6;color:#b91c1c;border:1px solid #fca5a5;}

/* Buttons */
.stButton>button {
    border-radius:12px;
    padding:12px 18px;
    font-weight:600;
    background: linear-gradient(90deg,#6366f1,#818cf8);
    color:white;
    border:none;
}

/* Small label */
.small {font-size:12px;color:#6b7280;}

/* Responsive columns */
@media (max-width: 800px) {
    .stColumn {width:100% !important; margin-bottom:12px;}
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Label mappings
# ---------------------------
YES_NO = {"Yes": 1, "No": 0}
GENDER_MAP = {"Male": 1, "Female": 0}
ADDRESS_MAP = {"Urban": 0, "Rural": 1}

def yn_input(label, help_text=None, key=None, default="No"):
    lbl = st.selectbox(label, list(YES_NO.keys()), index=0 if default=="No" else 1, help=help_text, key=key)
    return YES_NO[lbl], lbl

# ---------------------------
# Load model & scaler
# ---------------------------
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("model (2).pkl")
        age_scaler = joblib.load("scaler (1).pkl")
        return model, age_scaler
    except Exception:
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
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("## 🦴 Arthritis Predictor")
    st.markdown("Early-risk screening (educational purpose).")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("- Fill patient details\n- Only **Age** is scaled\n- Model predicts **Arthritis** 0/1")
    st.markdown("---")
    st.markdown("**Tips**")
    st.caption("• Keep model.pkl & scaler.pkl in the same folder.\n• Columns must match training order.\n• Age scaler only applies to Age feature.")

st.markdown("<h1>Arthritis Prediction <span class='badge'>Age scaled only</span></h1>", unsafe_allow_html=True)
st.caption("Fill in patient details. Prediction generates a downloadable PDF report.")

if model is None or age_scaler is None:
    st.error("Model or Scaler not found. Place **model.pkl** and **scaler.pkl** in the same folder.")
    st.stop()

# ---------------------------
# UI Sections
# ---------------------------
ui_labels = {}
num_values = {}

# Helper function to build each section card
def section_card(title, icon="🧾"):
    st.markdown(f"<div class='block-card'><h3>{icon} {title}</h3></div>", unsafe_allow_html=True)

# General & Lifestyle
st.markdown("<div class='block-card'>", unsafe_allow_html=True)
st.subheader("👤 General & Lifestyle")
c1, c2, c3, c4 = st.columns(4)
with c1:
    General_Health = st.slider("General Health (1=Poor, 5=Excellent)", 1, 5, 3)
    Checkup_num, Checkup_lbl = yn_input("Regular Checkup?")
    Exercise_num, Exercise_lbl = yn_input("Exercise Habit?")
with c2:
    Gender_lbl = st.radio("Gender", list(GENDER_MAP.keys()), index=0)
    Gender_num = GENDER_MAP[Gender_lbl]
    Age = st.slider("Age (years)", 10, 100, 40)
    Age_Category = st.slider("Age Category (1-13)", 1, 13)
with c3:
    Smoking_num, Smoking_lbl = yn_input("Smoking History?")
    OnlineSupport_num, OnlineSupport_lbl = yn_input("Online Support Usage?")
    address_lbl = st.selectbox("Address Type", list(ADDRESS_MAP.keys()))
    address_num = ADDRESS_MAP[address_lbl]
with c4:
    activities_num, activities_lbl = yn_input("Physical Activities?")
    internet_num, internet_lbl = yn_input("Internet Access at Home?")
    romantic_num, romantic_lbl = yn_input("In a Romantic Relationship?")
st.markdown("</div>", unsafe_allow_html=True)

# Medical History
st.markdown("<div class='block-card'>", unsafe_allow_html=True)
st.subheader("🏥 Medical History")
c5, c6, c7, c8 = st.columns(4)
with c5:
    Heart_num, Heart_lbl = yn_input("Heart Disease?")
    SkinCancer_num, SkinCancer_lbl = yn_input("Skin Cancer?")
with c6:
    OtherCancer_num, OtherCancer_lbl = yn_input("Other Cancer?")
    Depression_num, Depression_lbl = yn_input("Depression?")
with c7:
    Diabetes_num, Diabetes_lbl = yn_input("Diabetes?")
    Obesity_num, Obesity_lbl = yn_input("Obesity?")
with c8:
    Mental_Health_Status = st.slider("Mental Health Score (1-5)", 1, 5, 3)
    Stress_Level = st.slider("Stress Level (1-5)", 1, 5, 3)
st.markdown("</div>", unsafe_allow_html=True)

# Symptoms
st.markdown("<div class='block-card'>", unsafe_allow_html=True)
st.subheader("🧬 Symptoms")
c9, c10, c11, c12 = st.columns(4)
with c9:
    Polyuria_num, Polyuria_lbl = yn_input("Polyuria?")
    Polydipsia_num, Polydipsia_lbl = yn_input("Polydipsia?")
with c10:
    sudden_weight_loss_num, sudden_weight_loss_lbl = yn_input("Sudden Weight Loss?")
    weakness_num, weakness_lbl = yn_input("Weakness?")
with c11:
    Polyphagia_num, Polyphagia_lbl = yn_input("Polyphagia?")
    Genital_thrush_num, Genital_thrush_lbl = yn_input("Genital Thrush?")
with c12:
    visual_blurring_num, visual_blurring_lbl = yn_input("Visual Blurring?")
    Itching_num, Itching_lbl = yn_input("Itching?")
c13, c14, c15 = st.columns(3)
with c13:
    Irritability_num, Irritability_lbl = yn_input("Irritability?")
with c14:
    delayed_healing_num, delayed_healing_lbl = yn_input("Delayed Healing?")
with c15:
    muscle_stiffness_num, muscle_stiffness_lbl = yn_input("Muscle Stiffness?")
    Alopecia_num, Alopecia_lbl = yn_input("Alopecia?")
st.markdown("</div>", unsafe_allow_html=True)

# Support & Education
st.markdown("<div class='block-card'>", unsafe_allow_html=True)
st.subheader("👪 Support & Education")
d1, d2, d3, d4 = st.columns(4)
with d1:
    guardian_num, guardian_lbl = yn_input("Guardian Support?")
    schoolsup_num, schoolsup_lbl = yn_input("School Support?")
with d2:
    nursery_num, nursery_lbl = yn_input("Nursery Attended?")
    higher_num, higher_lbl = yn_input("Higher Education?")
st.markdown("</div>", unsafe_allow_html=True)

# Collect labels and numeric features
ui_labels.update({
    "General_Health": General_Health, "Checkup": Checkup_lbl, "Exercise": Exercise_lbl,
    "Heart_Disease": Heart_lbl, "Skin_Cancer": SkinCancer_lbl, "Other_Cancer": OtherCancer_lbl,
    "Depression": Depression_lbl, "Diabetes": Diabetes_lbl, "Gender": Gender_lbl,
    "Age_Category": Age_Category, "Smoking_History": Smoking_lbl, "Age": Age,
    "Polyuria": Polyuria_lbl, "Polydipsia": Polydipsia_lbl, "sudden weight loss": sudden_weight_loss_lbl,
    "weakness": weakness_lbl, "Polyphagia": Polyphagia_lbl, "Genital thrush": Genital_thrush_lbl,
    "visual blurring": visual_blurring_lbl, "Itching": Itching_lbl, "Irritability": Irritability_lbl,
    "delayed healing": delayed_healing_lbl, "muscle stiffness": muscle_stiffness_lbl, "Alopecia": Alopecia_lbl,
    "Obesity": Obesity_lbl, "Mental_Health_Status": Mental_Health_Status, "Stress_Level": Stress_Level,
    "Online_Support_Usage": OnlineSupport_lbl, "address": address_lbl, "guardian": guardian_lbl,
    "schoolsup": schoolsup_lbl, "activities": activities_lbl, "nursery": nursery_lbl,
    "higher": higher_lbl, "internet": internet_lbl, "romantic": romantic_lbl
})

num_values.update({
    "General_Health": General_Health, "Checkup": YES_NO[Checkup_lbl], "Exercise": YES_NO[Exercise_lbl],
    "Heart_Disease": YES_NO[Heart_lbl], "Skin_Cancer": YES_NO[SkinCancer_lbl], "Other_Cancer": YES_NO[OtherCancer_lbl],
    "Depression": YES_NO[Depression_lbl], "Diabetes": YES_NO[Diabetes_lbl], "Gender": GENDER_MAP[Gender_lbl],
    "Age_Category": Age_Category, "Smoking_History": YES_NO[Smoking_lbl], "Age": Age,
    "Polyuria": YES_NO[Polyuria_lbl], "Polydipsia": YES_NO[Polydipsia_lbl], "sudden weight loss": YES_NO[sudden_weight_loss_lbl],
    "weakness": YES_NO[weakness_lbl], "Polyphagia": YES_NO[Polyphagia_lbl], "Genital thrush": YES_NO[Genital_thrush_lbl],
    "visual blurring": YES_NO[visual_blurring_lbl], "Itching": YES_NO[Itching_lbl], "Irritability": YES_NO[Irritability_lbl],
    "delayed healing": YES_NO[delayed_healing_lbl], "muscle stiffness": YES_NO[muscle_stiffness_lbl], "Alopecia": YES_NO[Alopecia_lbl],
    "Obesity": YES_NO[Obesity_lbl], "Mental_Health_Status": Mental_Health_Status, "Stress_Level": Stress_Level,
    "Online_Support_Usage": YES_NO[OnlineSupport_lbl], "address": ADDRESS_MAP[address_lbl],
    "guardian": YES_NO[guardian_lbl], "schoolsup": YES_NO[schoolsup_lbl], "activities": YES_NO[activities_lbl],
    "nursery": YES_NO[nursery_lbl], "higher": YES_NO[higher_lbl], "internet": YES_NO[internet_lbl],
    "romantic": YES_NO[romantic_lbl]
})

# ---------------------------
# Prediction
# ---------------------------
col_run, col_pdf = st.columns([2,1])
with col_run:
    run = st.button("🚀 Predict Arthritis", use_container_width=True)

pred_label = None
proba = None
pdf_bytes = None

def predict(numeric_features: dict):
    scaled_age = age_scaler.transform([[numeric_features["Age"]]])[0][0]
    X = pd.DataFrame([[scaled_age if k=="Age" else numeric_features[k] for k in FEATURES]], columns=FEATURES)
    yhat = model.predict(X)[0]
    p = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None
    return int(yhat), p, numeric_features["Age"], scaled_age

if run:
    with st.spinner("Predicting…"):
        time.sleep(0.5)
        pred_label, proba, raw_age_val, scaled_age_val = predict(num_values)

    st.markdown("### 🩺 Prediction Result")
    if pred_label == 1:
        st.markdown(f"<div class='result-box bad'>⚠️ Arthritis Positive<br><span class='small'>Educational purpose. Consult a clinician.</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-box good'>✅ Arthritis Negative<br><span class='small'>Maintain healthy habits.</span></div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Model Output", "Positive" if pred_label==1 else "Negative")
    m2.metric("Confidence", f"{proba*100:.1f}%" if proba is not None else "N/A")
    m3.metric("Scaled Age", f"{scaled_age_val:.3f}")

    # PDF generation
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

        c.setFont("Helvetica-Bold", 16)
        c.drawString(2*cm, y, "Arthritis Prediction Report")
        y -= 24
        c.setFont("Helvetica", 9)
        c.drawString(2*cm, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 18
        c.line(2*cm, y, W-2*cm, y)
        y -= 16

        line("Result:", bold=True)
        result_text = "Arthritis Positive" if pred_label==1 else "Arthritis Negative"
        line(f"- {result_text}")
        line(f"- Confidence: {proba*100:.1f}%" if proba else "- Confidence: N/A")
        line(f"- Age: {raw_age_val} (scaled: {scaled_age_val:.3f})")
        y -= 8

        line("Inputs (Human-Readable)", bold=True)
        for k in FEATURES:
            val = ui_labels.get(k, "—")
            if k == "Age":
                line(f"- Age: {val} (scaled in model)")
            else:
                line(f"- {k}: {val}")
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
