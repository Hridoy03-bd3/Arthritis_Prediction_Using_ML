import streamlit as st
import numpy as np
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import base64

# Load the saved model
model = joblib.load("model.joblib")

# Custom CSS for modern UI
st.markdown("""
<style>
    .main { background-color: #f9fafb; }
    .block-card {
        background: white; 
        border-radius: 12px; 
        padding: 20px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
        margin-bottom: 20px;
    }
    .title { 
        font-size: 28px; 
        text-align: center; 
        color: #1f2937;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #2563eb;
        margin-bottom: 10px;
    }
    .prediction-card {
        background: #e0f2fe;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-top: 20px;
    }
    .result {
        font-size: 22px;
        font-weight: bold;
        color: #1e3a8a;
    }
    .probability {
        font-size: 18px;
        color: #334155;
    }
    .btn-download {
        background: #2563eb;
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        text-decoration: none;
    }
    .btn-download:hover {
        background: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🧬 Diabetes Risk Prediction System</div>", unsafe_allow_html=True)

# Mapping dictionaries
yes_no = {"Yes": 1, "No": 0}
gender_map = {"Male": 1, "Female": 0}
address_map = {"Urban": 0, "Rural": 1}

# Helper function for Yes/No inputs
def yn_input(label):
    choice = st.radio(label, list(yes_no.keys()), horizontal=True)
    return yes_no[choice]

# Helper for PDF report
def create_pdf(filename, sections):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    for section_name, items in sections.items():
        flowables.append(Paragraph(f"<b>{section_name}</b>", styles["Heading2"]))
        flowables.append(Spacer(1, 10))
        for label, value in items.items():
            flowables.append(Paragraph(f"{label}: {value}", styles["Normal"]))
        flowables.append(Spacer(1, 15))

    doc.build(flowables)

# --------------------- UI Sections ----------------------

# General Information
with st.container():
    st.markdown("<div class='block-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🏥 General Information</div>", unsafe_allow_html=True)
    Age = st.slider("Age", 1, 120, 25)
    Gender = st.radio("Gender", list(gender_map.keys()), horizontal=True)
    address = st.radio("Living Area", list(address_map.keys()), horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Medical History
with st.container():
    st.markdown("<div class='block-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧾 Medical History</div>", unsafe_allow_html=True)
    family_history = yn_input("Family history of diabetes?")
    htn = yn_input("Hypertension?")
    heart_disease = yn_input("Heart disease?")
    st.markdown("</div>", unsafe_allow_html=True)

# Symptoms Section
with st.container():
    st.markdown("<div class='block-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🩺 Symptoms</div>", unsafe_allow_html=True)
    Polyuria = yn_input("Polyuria?")
    Itching = yn_input("Itching?")
    Alopecia = yn_input("Alopecia?")
    st.markdown("</div>", unsafe_allow_html=True)

# Lifestyle Section
with st.container():
    st.markdown("<div class='block-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🏃 Lifestyle Factors</div>", unsafe_allow_html=True)
    exercise = yn_input("Regular Exercise?")
    balanced_diet = yn_input("Balanced Diet?")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------- Prediction -----------------------

if st.button("🔍 Predict"):
    # Convert categorical values
    Gender_val = gender_map[Gender]
    address_val = address_map[address]

    # Prepare features for model
    features = np.array([[Age, Gender_val, address_val, family_history, htn, heart_disease,
                          Polyuria, Itching, Alopecia, exercise, balanced_diet]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    # Result Card
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(f"<div class='result'>High Risk of Diabetes</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result'>Low Risk of Diabetes</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='probability'>Probability: {probability:.2f}%</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # PDF Report
    sections = {
        "General Information": {
            "Age": Age, "Gender": Gender, "Address": address
        },
        "Medical History": {
            "Family History": "Yes" if family_history else "No",
            "Hypertension": "Yes" if htn else "No",
            "Heart Disease": "Yes" if heart_disease else "No"
        },
        "Symptoms": {
            "Polyuria": "Yes" if Polyuria else "No",
            "Itching": "Yes" if Itching else "No",
            "Alopecia": "Yes" if Alopecia else "No"
        },
        "Lifestyle": {
            "Exercise": "Yes" if exercise else "No",
            "Balanced Diet": "Yes" if balanced_diet else "No"
        },
        "Prediction": {
            "Risk Level": "High Risk" if prediction == 1 else "Low Risk",
            "Probability": f"{probability:.2f}%"
        }
    }

    pdf_filename = "Diabetes_Report.pdf"
    create_pdf(pdf_filename, sections)

    with open(pdf_filename, "rb") as f:
        pdf_data = f.read()
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}" class="btn-download">📥 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)
