import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Predictor", layout="wide")
st.title(" Tourism Package Purchase Predictor")

# --- 1. Load Model ---
@st.cache_resource
def load_model():
    try:
        # Check if model exists locally first, else download
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

# We create inputs for EVERY column mentioned in the error
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", 18, 70, 30)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration of Pitch", 0, 100, 15)
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])

with col2:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
    NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 3)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])

with col3:
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfChildrenVisiting = st.number_input("Number of Children", 0, 5, 0)
    OwnCar = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    Passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])

# --- 3. Prediction Logic ---
if st.button("Predict Purchase Intent"):
    if model is not None:
        # Construct the DataFrame with EXACTLY the keys from your error message
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

        input_df = pd.DataFrame([data])

        try:
            # The model pipeline handles the scaling and encoding internally
            prediction = model.predict(input_df)[0]
            
            st.divider()
            if prediction == 1:
                st.balloons()
                st.success("### Prediction: **Customer LIKELY to buy the package!** ")
            else:
                st.warning("### Prediction: **Customer UNLIKELY to buy the package.** ")
        
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Check if categorical values match the training set labels.")
    else:
        st.error("Model not loaded properly.")
