import streamlit as st
import pandas as pd
import numpy as np
import joblib

# PAGE CONFIGURATION 
st.set_page_config(page_title="Diabetes Analytics", layout="wide")

# LOAD MODELS 
# We use @st.cache_resource so it only loads the model once, making the app fast
@st.cache_resource
def load_models():
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_models()

# APP LAYOUT 
st.title("🏥 Diabetes Predictive Analytics & Readmission Dashboard")
st.write("Developed by [Your Name] | Powered by Machine Learning")

# Create two tabs for the two parts of your project
tab1, tab2 = st.tabs(["🩺 Patient Risk Predictor", "📊 Clinical Research Insights"])

# TAB 1: THE PREDICTOR (Part 1) 
with tab1:
    st.header("Patient Vitals Input")
    st.write("Adjust the sliders to predict the patient's diabetes risk based on the trained Logistic Regression model.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Assuming standard Pima Indian Diabetes features. 
        # Adjust these names if your dataset differs slightly!
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.slider("Glucose Level", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
        
    with col2:
        insulin = st.slider("Insulin", 0, 846, 79)
        bmi = st.slider("BMI", 0.0, 67.1, 32.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.5)
        age = st.slider("Age", 21, 81, 33)

    # Prediction Logic
    if st.button("Calculate Risk Score", type="primary"):
        # Combine inputs into a dataframe
        user_data = pd.DataFrame({
            'Pregnancies': [pregnancies], 'Glucose': [glucose], 
            'BloodPressure': [blood_pressure], 'SkinThickness': [skin_thickness], 
            'Insulin': [insulin], 'BMI': [bmi], 
            'DiabetesPedigreeFunction': [dpf], 'Age': [age]
        })
        
        # Scale the inputs using the scaler we saved from Colab
        user_data_scaled = scaler.transform(user_data)
        
        # Make Prediction
        prediction = model.predict(user_data_scaled)
        probability = model.predict_proba(user_data_scaled)[0][1] # Probability of Class 1
        
        st.divider()
        if prediction[0] == 1:
            st.error(f"⚠️ **High Risk of Diabetes Detected.** (Probability: {probability*100:.1f}%)")
        else:
            st.success(f"✅ **Low Risk.** (Probability: {probability*100:.1f}%)")

# TAB 2: CLINICAL INSIGHTS (Part 2)
with tab2:
    st.header("Readmission & Intervention Analysis")
    st.write("This section explores the relationship between HbA1c monitoring, medication changes, and 30-day hospital readmission rates.")
    
    # Display the images you saved from Colab
    st.subheader("1. Proactive Intervention (HbA1c vs. Medication Change)")
    st.image("heatmap.png", caption="Heatmap showing medication change rates based on HbA1c testing.")
    st.info("Insight: Abnormal results (>8) trigger a medication change 65% of the time, proving that data visibility directly drives proactive clinical action.")
    
    st.subheader("2. Readmission Reduction")
    st.image("barchart.png", caption="Readmission rates stratified by Primary Diagnosis.")
    st.info("Insight: Proactive HbA1c measurement is associated with a decrease in critical <30-day hospital readmissions, particularly for primary diabetic cohorts.")
    st.subheader("3. Additional Readmission Insights")
    st.image("barchart2.png", caption="Furthur breakdown of hospital readmission rates.")
