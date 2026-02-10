import streamlit as st
import pandas as pd
import joblib
import pickle

# Load model and feature list
model = joblib.load('student_depression_rf_model.pkl')
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.title("Student Depression Risk Assessment")
st.info("Risk Analysis Dashboard (Random Forest Tuned Model)")

# --- INPUT SECTION ---
col1, col2 = st.columns(2)

with col1:
    academic_pressure = st.slider("Academic Pressure (1-5)", 1.0, 5.0, 3.0)
    age = st.number_input("Age", 18, 35, 20)
    study_satisfaction = st.slider("Study Satisfaction (1-5)", 1.0, 5.0, 3.0)
    cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5)
    
with col2:
    work_study_hours = st.slider("Work/Study Hours per Day", 0.0, 12.0, 6.0)
    financial_stress = st.selectbox("Financial Stress Level", [1.0, 2.0, 3.0, 4.0, 5.0])
    gender = st.selectbox("Gender", ["Male", "Female"])
    family_history = st.radio("Family History of Mental Illness", ["Yes", "No"])
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])

if st.button("Analyze Risk Profile"):
    # Create the base dictionary
    data = {
        'Academic Pressure': academic_pressure,
        'Age': age,
        'Study Satisfaction': study_satisfaction,
        'CGPA': cgpa,
        'Work/Study Hours': work_study_hours,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Family History of Mental Illness_Yes': 1 if family_history == "Yes" else 0,
        f'Financial Stress_{float(financial_stress)}': 1,
        f"Sleep Duration_'{sleep_duration}'": 1,
        f"Dietary Habits_{dietary_habits}": 1
    }
    
    input_df = pd.DataFrame([data])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[feature_names] 
    
    # Get probability of Depression (Class 1)
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Results")

    # --- SPECTRUM LOGIC ---
    # Low Risk: < 35%
    # Moderate/Borderline: 35% - 60%
    # High Risk: > 60%

    if probability < 0.35:
        st.success(f"### Low Risk Profile")
        st.write(f"Confidence score: **{1-probability:.2%}** (Safe)")
        st.progress(probability)
        
    elif 0.35 <= probability <= 0.60:
        st.warning(f"### Moderate / Borderline Risk")
        st.write(f"Confidence score: **{probability:.2%}**")
        st.write(" **Warning:** You lowkey might have depression. While not a definitive 'High Risk',  it shows suggest vulnerability.")
        st.progress(probability)
        
    else:
        st.error(f"### High Risk Profile")
        st.write(f"Confidence score: **{probability:.2%}**")
        st.write(" **Urgent:** Several key indicators (like Academic Pressure or Financial Stress) strongly align with at-risk profiles.")
        st.progress(probability)

    # Adding a visual spectrum bar explanation
    st.write("---")
    st.caption("Risk Spectrum: 0-35% Low | 35-60% Moderate | 60-100% High")