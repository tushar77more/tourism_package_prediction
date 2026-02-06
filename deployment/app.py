import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.title("Tourism Package Purchase Predictor")

# Download model from HF
model_path = hf_hub_download(
    repo_id="tushar77more/tourism_model",
    filename="tourism_xgb_model.pkl"
)

model = joblib.load(model_path)

st.header("Enter Customer Details")

Age = st.number_input("Age", 18, 70, 30)
MonthlyIncome = st.number_input("Monthly Income", 1000, 100000, 20000)
NumberOfTrips = st.number_input("Number of Trips", 0, 10, 2)
CityTier = st.selectbox("City Tier", [1, 2, 3])

# Add rest of fields similarly...

if st.button("Predict"):
    input_df = pd.DataFrame([[Age, MonthlyIncome, NumberOfTrips, CityTier]],
                            columns=["Age", "MonthlyIncome", "NumberOfTrips", "CityTier"])

    prediction = model.predict(input_df)[0]
    st.success("Customer will buy package" if prediction == 1 else "Customer will NOT buy")
