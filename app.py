import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import utils.resp_enc as resp_enc
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(layout='wide')
st.title("Tesco")
st.subheader("""Customer Churn Analysis""")

def load_model():
    with open('./model/best_model.pkl', 'rb') as f:
        model = joblib.load(f)
    return model

def get_features(model):
    features = model.get_booster().feature_names
    return features
    
def get_shap(x):
    explainer = joblib.load('./model/shap_explainer.pkl')
    shap_values = explainer(x)
    return explainer, shap_values

#https://gist.github.com/andfanilo/6bad569e3405c89b6db1df8acf18df0e
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

#create a button to upload the test file
uploaded_file = st.file_uploader("Upload your test csv file", type=["csv"])
if uploaded_file:
    test_df = pd.read_csv(uploaded_file)#.sort_values(by=['customerID'])
    response_enc = resp_enc.response_encoding()
    response_enc = joblib.load('./model/resp_enc.pkl')
    test_df = response_enc.transform(test_df)
    features = get_features(load_model())
    x_test = test_df.loc[:, features]
    y_pred_proba = load_model().predict_proba(x_test)[:,1]
    y_pred = load_model().predict(x_test)



    #plotting shap force plot
    explainer, shap_values = get_shap(x_test)
    # st.write()

    id = st.selectbox('Select a customer id to analyse',test_df.customerID)
    st.write(
        f""" ### Shapley force plot
        Customer ID: {id}

        Red: Increases probability of churn
        Blue: Decreases probability of churn
        width: strength of impact
        """
    )
    st.write(
        """
        `* Top Numbers are probability (Bold one is the probability of churn)`
        
        `* Bottom Numbers are feature values`
        """)
    index = test_df.query('customerID == @id').index[0]
    # plot = show_force_plot(index, shap_values, explainer)
    st_shap(
        shap.plots.force(
        explainer.expected_value,
        shap_values.values[index,:],
        x_test.iloc[index,:],
        link = "logit" #to show probability instead os shapley value
            ),
            height = 500)




