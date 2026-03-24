import streamlit as st
import pandas as pd
import joblib

model_svr = joblib.load('svr_model.pkl')
model_features = joblib.load('model_features.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Insurance Charges Prediction App💵")
st.write("Please enter the following details to predict insurance charges:")

input_age = st.number_input("Age", min_value=0, max_value=120, value=30)
input_bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
input_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
input_is_female = st.selectbox("Sex", ["Male", "Female"])
input_is_smoker = st.selectbox("Smoker", ["Yes", "No"])
input_region = st.selectbox("Region", ["northwest", "southeast", "southwest", "northeast"])

if st.button("Predict Charges"):
    
    if input_bmi < 18.5:
        bmi_cat = "Underweight"
    elif 18.5 <= input_bmi < 25.0:
        bmi_cat = "Normal"
    elif 25.0 <= input_bmi < 30.0:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"

    raw_input = {
        'age': input_age,
        'bmi': input_bmi,
        'children': input_children,
        'is_female': 1 if input_is_female == "Female" else 0,
        'is_smoker': 1 if input_is_smoker == "Yes" else 0,
        f'region_{input_region}': 1,
        f'bmi_category_{bmi_cat}': 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_features]

    cols_to_scale = ['age', 'bmi']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    prediction = model_svr.predict(input_df)[0]
    
    st.subheader(f"Predicted Insurance Charges: ${prediction:,.2f}")