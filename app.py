import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import utils.resp_enc as resp_enc
# import shap

# model = xgb.Booster(model_file='./model/best_model.pkl')
st.title("Tesco")
st.write(""" ## Customer Churn Analysis """)

def load_model():
    with open('./model/best_model.pkl', 'rb') as f:
        model = joblib.load(f)
    return model

def get_features(model):
    features = model.get_booster().feature_names
    return features


#create a button to upload the test file
uploaded_file = st.file_uploader("Upload your test csv file", type=["csv"])
if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    response_enc = resp_enc.response_encoding()
    response_enc = joblib.load('./model/resp_enc.pkl')
    test_df = response_enc.transform(test_df)
    features = get_features(load_model())
    x_test = test_df[features]
    # x_test = test_df.loc[:, features]
    st.write(x_test)
    y_pred_proba = load_model().predict_proba(x_test)[:,1]
    st.write(y_pred_proba)
    y_pred = load_model().predict(x_test)
    st.write(y_pred.sum()/len(y_pred))
    



