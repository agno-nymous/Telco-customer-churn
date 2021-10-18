import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import shap

st.title("Tesco")
st.write(""" ## Customer Churn Analysis """)


@st.cache
def load_model():
    with open('./model/best_model.pkl', 'rb') as f:
        model = joblib.load(f)
    return model

#create a button to upload the test file
uploaded_file = st.file_uploader("Upload your test csv file", type=["csv"])
if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    test_df = test_df.drop(columns=['CustomerID'])

