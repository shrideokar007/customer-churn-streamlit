import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📊 Customer Churn Prediction App")

uploaded_file = st.file_uploader("Upload a customer CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Uploaded Data")
    st.dataframe(df.head())

    st.subheader("📈 Churn Predictions")
    df_encoded = df.copy()
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService']:
        df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    predictions = model.predict(df_encoded)
    df['Churn Prediction'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
    st.dataframe(df)
