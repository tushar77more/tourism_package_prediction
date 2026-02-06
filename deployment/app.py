import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Predictor", layout="wide")
st.title(" Tourism Package Purchase Predictor")

# --- 1. Load Model ---
# Note: Using your repo_id from the error message. 
# Make sure the filename matches what you saved in train.py
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="tushar77more/tourism_package_predictor", 
            filename="tourism_xgb_model.pkl"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 2. Input UI ---
st.header(" Customer Information")

# Using columns to make the UI more compact
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", 18, 70, 30)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration of Pitch", 0, 100, 15)

with col2:
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
    NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 3)

with col3:
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfChildrenVisiting = st.number_input("Number of Children", 0, 5, 0)

st.divider()
col4, col5, col6 = st.columns(3)

with col4:
    OwnCar = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col5:
    Passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col6:
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])

# --- 3. Prediction Logic ---
if st.button(" Predict Purchase Intent"):
    if model is not None:
        # Create dictionary with EXACT column names from the error message
        data = {
            'Age': Age,
            'TypeofContact': TypeofContact,
            'CityTier': CityTier,
            'DurationOfPitch': DurationOfPitch,
            'Occupation': Occupation,
            'Gender': Gender,
            'NumberOfPersonVisiting': NumberOfPersonVisiting,
            'NumberOfFollowups': NumberOfFollowups,
            'ProductPitched': ProductPitched,
            'PreferredPropertyStar': PreferredPropertyStar,
            'MaritalStatus': MaritalStatus,
            'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
            'OwnCar': OwnCar,
            'Passport': Passport,
            'PitchSatisfactionScore': PitchSatisfactionScore,
            'Designation': Designation
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Image of a Scikit-Learn Pipeline showing how data flows through transformers
        

        try:
            prediction = model.predict(input_df)[0]
            
            st.divider()
            if prediction == 1:
                st.balloons()
                st.success("### Prediction: **The customer is likely to purchase the package!** ")
            else:
                st.warning("### Prediction: **The customer is unlikely to purchase the package.** ")
        
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("This usually happens if the categorical values (like Occupation) in the app don't match the ones seen during training.")
    else:
        st.error("Model not loaded. Please check the logs.")
