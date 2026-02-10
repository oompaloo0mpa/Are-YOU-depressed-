import streamlit as st
import pandas as pd
import joblib
import pickle
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="BIS", page_icon="😔", layout="wide") #this emoji was mine to i found this off telegram

# --- BACKGROUND & CSS LOGIC ---
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_design(bg_image_path):
    # If you haven't downloaded an image yet, you can replace this with a URL
    # or keep the try/except to fall back to a standard color
    try:
        bin_str = get_base64(bg_image_path)
        bg_img_style = f'background-image: url("data:image/png;base64,{bin_str}");'
    except FileNotFoundError:
        bg_img_style = 'background-color: #f0f2f6;'

    st.markdown(f"""
    <style>
    /* 1. Sets Background and Kills Streamlit's Default White Blocks */
    .stApp {{
        {bg_img_style}
        background-size: cover;
        background-attachment: fixed;
    }}

    /* 2. Target the main content blocks to make them transparent */
    [data-testid="stVerticalBlock"] > div {{
        background-color: transparent !important;
    }}
    
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0) !important;
    }}

    /* 3. The Glass Card Container */
    .glass-card {{
        background: rgba(255, 255, 255, 0.72); /* Transparency */
        backdrop-filter: blur(15px); /* The "Frosted" look */
        -webkit-backdrop-filter: blur(15px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 35px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }}

    /* 4. Styled Button */
    .stButton>button {{
        width: 100%;
        border-radius: 15px;
        height: 3.5em;
        background: linear-gradient(45deg, #2E7D32, #4CAF50);
        color: white;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    </style>
    """, unsafe_allow_html=True)

# CALL THE DESIGN (Change 'assets/background.jpg' to your actual file name)
set_design('assets/background.jpg')

# --- DATA LOADING ---
@st.cache_resource
def load_assets():
    model = joblib.load('student_depression_rf_model.pkl')
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

model, feature_names = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=100)
    st.title("Support Center")
    st.markdown("---")
    st.info("**Did you know?**\n\nApproximately **37%** of college students report symptoms of depression.")
    st.info("**Support is key:**\n\n**66%** of students cite mental health as the main reason for leaving college.")
    st.markdown("---")
    st.subheader("Need to talk?")
    st.warning("Helpline: 117 | Crisis Text: WhatsApping: 6669 1771")

# --- MAIN UI ---
st.title("Blue Is Sadness - Student Depression Risk Checker")
st.markdown("##### TP's own wikihow to know you’re depressed")

col1, col2 = st.columns(2)

with col1:
    st.subheader("✏️Academic Life✏️") #YES I MADE THE EMOJIS MYSELF I JUST THOUGHT THE STUPID WEBSTIE LOOKED SO BORING
    st.markdown("##### A student's life is tough, we get it.")
    academic_pressure = st.select_slider("Current Academic Pressure", options=[round(x * 0.1, 1) for x in range(10, 51)], value=3.0, help="1: Relaxed | 5: Overwhelmed")
    study_satisfaction = st.select_slider("Study Satisfaction", options=[round(x * 0.1, 1) for x in range(10, 51)], value=3.0)
    cgpa_input = st.number_input("Current CGPA", 0.0, 4.0, 3.0, step=0.01, help="Singapore GPA scale: 0.0 - 4.0")
    age = st.number_input("Age", 18, 35, 20)

with col2:
    st.subheader("🫂Lifestyle & Stress🫂") # yes just to clarify this was also my doing i literally had to go on emojipedia to find the emoji
    st.markdown("##### How's life outside of academics?")
    financial_stress = st.select_slider("Financial Stress", options=[round(x * 0.1, 1) for x in range(10, 51)], value=3.0)
    work_study_hours = st.slider("Daily Work/Study Hours", 0, 16, 6)
    sleep_duration = st.selectbox("Average Sleep", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
    dietary_habits = st.radio("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"], horizontal=True)

with st.expander("Additional Background Info"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    family_history = st.checkbox("Family History of Mental Health Issues")

# --- CALCULATION ---
if st.button("Generate My Risk Profile"):
    # Convert Singapore GPA (0-4) to model's scale (0-10)
    cgpa = cgpa_input * 2.5
    
    # Map data to match model's training features
    data = {
        'Academic Pressure': float(academic_pressure),
        'Age': age,
        'Study Satisfaction': float(study_satisfaction),
        'CGPA': cgpa,
        'Work/Study Hours': float(work_study_hours),
        'Gender_Male': 1 if gender == "Male" else 0,
        'Family History of Mental Illness_Yes': 1 if family_history else 0,
        f'Financial Stress_{float(financial_stress)}': 1,
        f"Sleep Duration_'{sleep_duration}'": 1,
        f"Dietary Habits_{dietary_habits}": 1
    }

    input_df = pd.DataFrame([data])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("### 📊 Assessment Summary") #basically every emoji here is mine
    
    m1, m2 = st.columns(2)

    if probability < 0.45:
        st.balloons()
        m1.metric("Risk Level", "LOW", delta="- Safe")
        m2.metric("Confidence", f"{(1-probability)*100:.1f}")
        st.success("### ✅ Everything looks good!") #yes this emoji too
        st.write("Your profile suggests a healthy balance. Keep maintaining your current habits!")

    elif 0.45 <= probability <= 0.70:
        m1.metric("Risk Level", "MODERATE", delta="⚠️ Warning", delta_color="off")
        m2.metric("Confidence", f"{probability*100:.1f}")
        st.warning("### ⚠️ Moderate / Borderline Risk") #yes this emoji too
        st.write("**Note:** You are showing some signs of vulnerability. Consider focusing on rest and stress reduction.")

    else:
        m1.metric("Risk Level", "HIGH", delta="🚨 Critical", delta_color="inverse")
        m2.metric("Confidence", f"{probability*100:.1f}")
        st.error("### 🚨 High Risk Profile Identified") #this one i asked copilot to find for me
        st.write("**Urgent:** Key indicators align with at-risk profiles. Reaching out to a counselor is recommended.")

    st.markdown("#### 😬Risk Gauge") #mine too
    st.progress(probability)
    st.caption(f"Risk Probability: {probability*100:.1f}%")