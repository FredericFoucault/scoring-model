import streamlit as st
import pandas as pd
import numpy as np
import requests


#load the data
df = pd.read_csv("app_clean_final.csv")
dictionnaire = {}
for colonne in df.columns:
    dictionnaire[colonne] = df[colonne].tolist()
 


@app.route('/data',methods =["GET"])
def index():
    return jsonify(dictionnaire)





def load_model():
    model = joblib.load('model_scoring_LGBM.joblib')
    return model

# chargelement du model
model = load_model()





def predict(option):

    # selection des données du client
    dt = pd.DataFrame(dictionnaire)
    client_data = dt[dt.index == option,]
    pred = model.predict_proba(client_data)

    rec = pred[0]
    return rec







st.title('prediction de scoring bancaire')


option = st.selectbox("choisissez le numero de client", options = list(range(0,5000)))

st.write('You selected',option)


if option:
    rec =predict(option)

    prob_granted = rec * 100

    if prob_granted > 50:
        st.write(f"le credit est accepté à {prob_granted:.2f} %")

