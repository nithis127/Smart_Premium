import streamlit as st
import pandas as pd
import joblib
from scipy import stats

# =========================================================
# Custom Function (Required if used inside preprocessing pipeline)
# =========================================================
def boxcox_transform(X, l=0.5):
    X = X.copy()
    X[:, 1] = stats.boxcox(X[:, 1], lmbda=l)
    return X

# If your preprocessing pipeline requires the function:
globals()["boxcox_transform"] = boxcox_transform

# =========================================================
# Load Preprocessing Pipeline & Model
# =========================================================
preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")
model = joblib.load("smart_premium_model.pkl")

# =========================================================
# Page Config
# =========================================================
st.set_page_config(page_title="💎 SmartPremium Ultra", page_icon="💎", layout="wide")

# =========================================================
# Initialize session state
# =========================================================
st.session_state.setdefault("page", "form")

# =========================================================
# Modern Dark Theme CSS (Stable selectors)
# =========================================================
st.markdown("""
<style>
body {
    background: #0a0f1b;
    font-family: "Poppins", sans-serif;
    color: #e0e0e0;
}

/* Header */
.header {
    text-align: center;
    padding: 0.6rem 0;
    color: #fff;
    border-radius: 12px;
    backdrop-filter: blur(10px);
    background: linear-gradient(120deg, rgba(0,188,212,0.5), rgba(33,150,243,0.5));
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.header h1 {
    font-size: 1.6rem;
    font-weight: 700;
}
.header p {
    font-size: 0.85rem;
    color: #c0d6e4;
}

/* Tiles */
.tile {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    margin-bottom: 15px;
    transition: 0.3s ease;
}
.tile:hover {
    transform: scale(1.05);
    background: rgba(255,255,255,0.1);
}

.tile-icon {
    font-size: 1.5rem;
    margin-bottom: 0.3rem;
    color: #00e5ff;
}
.tile-value {
    font-size: 1.2rem;
    font-weight: 600;
    color: #03a9f4;
}
.tile-label {
    font-size: 0.85rem;
    color: #b0bec5;
}

/* Result Tile */
.result-tile {
    background: linear-gradient(145deg, #00bcd4, #2196f3);
    color: white;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 8px 30px rgba(33,150,243,0.4);
}

.result-tile h2 {
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00bcd4, #2196f3);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 1em;
    font-weight: 600;
    border: none;
    width: 100%;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #2196f3, #00bcd4);
}

/* Stable input styling */
input, select, textarea {
    background-color: rgba(255,255,255,0.08) !important;
    color: #e0e0e0 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Header
# =========================================================
st.markdown("""
<div class="header">
    <h1>💎 Smart Premium</h1>
    <p>Predicting Insurance Costs with Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Form Page
# =========================================================
if st.session_state.page == "form":

    st.markdown("## 🧾 Enter Customer Details")

    # Customer Info
    st.markdown("### 👤 Customer Info")
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("🎂 Age", 18, 65, 28)
        Gender = st.selectbox("🚻 Gender", ["Male", "Female"])
        Marital_Status = st.selectbox("💍 Marital Status", ["Married", "Divorced", "Single"])
        Education_Level = st.selectbox("🎓 Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    with col2:
        Occupation = st.selectbox("💼 Occupation", ["Employed", "Self-Employed", "Unemployed"])
        Number_of_Dependents = st.number_input("👶 Dependents", 0, 4, 1)
        Annual_Income = st.number_input("💵 Annual Income ($)", 1.0, 150000.0, 30000.0)
    with col3:
        Smoking_Status = st.selectbox("🚬 Smoking Status", ["No", "Yes"])
        Exercise_Frequency = st.selectbox("🏋️ Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
        Health_Score = st.number_input("💪 Health Score", 2.0, 60.0, 45.0)

    # Policy Info
    st.markdown("### 🧾 Policy Info")
    col1, col2, col3 = st.columns(3)
    with col1:
        Location = st.selectbox("📍 Location", ["Urban", "Suburban", "Rural"])
        Property_Type = st.selectbox("🏠 Property Type", ["House", "Apartment", "Condo"])
    with col2:
        Policy_Type = st.selectbox("🧾 Policy Type", ["Basic", "Comprehensive", "Premium"])
        Previous_Claims = st.slider("🧮 Previous Claims", 0, 10, 1)
    with col3:
        Vehicle_Age = st.slider("🚗 Vehicle Age (years)", 0, 20, 5)
        Credit_Score = st.slider("💳 Credit Score", 300, 850, 700)
        Insurance_Duration = st.slider("⏳ Insurance Duration (years)", 0, 10, 5)

    # Predict button
    if st.button("🚀 Predict Premium"):
        try:
            input_data = pd.DataFrame({
                "Age": [Age], "Gender": [Gender], "Annual Income": [Annual_Income],
                "Marital Status": [Marital_Status], "Number of Dependents": [Number_of_Dependents],
                "Education Level": [Education_Level], "Occupation": [Occupation],
                "Health Score": [Health_Score], "Location": [Location], "Policy Type": [Policy_Type],
                "Previous Claims": [Previous_Claims], "Vehicle Age": [Vehicle_Age],
                "Credit Score": [Credit_Score], "Insurance Duration": [Insurance_Duration],
                "Smoking Status": [Smoking_Status],
                "Exercise Frequency": [Exercise_Frequency], "Property Type": [Property_Type]
            })

            processed = preprocessing_pipeline.transform(input_data)
            prediction = model.predict(processed)[0]
            premium = f"${prediction:,.2f}"

            st.session_state.prediction = premium
            st.session_state.page = "result"

            # Clean structured tiles with icon + label
            st.session_state.tiles = {
                "🎂": ("Age", Age),
                "🚻": ("Gender", Gender),
                "💍": ("Marital Status", Marital_Status),
                "🎓": ("Education", Education_Level),
                "💼": ("Occupation", Occupation),
                "👶": ("Dependents", Number_of_Dependents),
                "💵": ("Income", f"${Annual_Income:,.0f}"),
                "💪": ("Health", Health_Score),
                "🚬": ("Smoking", Smoking_Status),
                "🏋️": ("Exercise", Exercise_Frequency),
                "📍": ("Location", Location),
                "🏠": ("Property", Property_Type),
                "🧾": ("Policy", Policy_Type),
                "🧮": ("Claims", Previous_Claims),
                "🚗": ("Vehicle Age", Vehicle_Age),
                "💳": ("Credit Score", Credit_Score),
                "⏳": ("Duration", Insurance_Duration)
            }

            st.rerun()

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# =========================================================
# Result Page
# =========================================================
elif st.session_state.page == "result":

    st.markdown("## 🖼️ Input Summary")

    tile_cols = st.columns(6)

    for i, (icon, (label, value)) in enumerate(st.session_state.tiles.items()):
        with tile_cols[i % 6]:
            st.markdown(f"""
            <div class="tile">
                <div class="tile-icon">{icon}</div>
                <div class="tile-value">{value}</div>
                <div class="tile-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Predicted Premium
    st.markdown("## 💸 Predicted Premium")

    st.markdown(f"""
    <div class='result-tile'>
        <h2>💸 {st.session_state.prediction}</h2>
        <p><b>Estimated Premium Amount</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Back button
    if st.button("🔙 Back to Form"):
        st.session_state.page = "form"
        st.rerun()