import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Tourism Predictor", layout="wide")
st.title(" Tourism Package Purchase Predictor")

# --- 1. Load Model Locally ---
@st.cache_resource
def load_model():
    # The model is uploaded to the root of the Space via hosting.py
    model_filename = "tourism_xgb_model.pkl"
    
    if os.path.exists(model_filename):
        try:
            return joblib.load(model_filename)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f" Model file '{model_filename}' not found in the Space!")
        st.info(f"Files currently in Space: {os.listdir('.')}")
        return None

model = load_model()

# --- 2. Input UI ---
st.header(" Enter Customer Details")
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", 18, 70, 30)
    # ADD THESE TWO:
    MonthlyIncome = st.number_input("Monthly Income", 5000, 100000, 25000)
    NumberOfTrips = st.number_input("Number of Trips", 1, 20, 3)
    
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])

with col2:
    DurationOfPitch = st.number_input("Duration of Pitch", 0, 100, 15)
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
    NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 3)

with col3:
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfChildrenVisiting = st.number_input("Number of Children", 0, 5, 0)
    OwnCar = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# These can go below the columns
col4, col5 = st.columns(2)
with col4:
    Passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col5:
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])

# --- 3. Prediction Logic ---
if st.button(" Predict"):
    if model:
        data = {
            'Age': Age,
            'MonthlyIncome': MonthlyIncome,     # Added
            'NumberOfTrips': NumberOfTrips,     # Added
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
            #prediction = model.predict(input_df)[0]
            # DEBUG: This will show in your Streamlit app
            st.write(f"Model Type: {type(model)}") 
            # It SHOULD say <class 'sklearn.pipeline.Pipeline'>
            # If it says <class 'xgboost.sklearn.XGBClassifier'>, the preprocessor is missing!
            
            prediction = model.predict(input_df)[0]
            st.divider()
            if prediction == 1:
                st.balloons()
                st.success("### Prediction: **Customer LIKELY to buy the package!** ")
            else:
                st.warning("### Prediction: **Customer UNLIKELY to buy the package.** ")
        
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model not loaded properly.")
