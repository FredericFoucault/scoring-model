import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

#data = requests.get('http://127.0.0.1:8080/data').text
data = requests.get('https://appprediction-b8add0149604.herokuapp.com/data').text

data = pd.DataFrame(json.loads(data))
id_client = data.index.values




st.title('prediction de scoring bancaire')


option = st.selectbox("choisissez le numero de client", (tuple(id_client)))

st.write('Vous avez selectionné le client avec l\'ID numéro',option)



if option:
    req = requests.get(f'http://127.0.0.1:8080/predict?ClientID={int(option)}')
    resultat = req.json()
    rec = resultat.get('prediction')
    inv_rec = 100 -rec
    if rec <= 50:
        st.write(f"Il est certain a {inv_rec:.2f} % que le client remboursera son crédit" )
    else:
        st.write(f"Il est certain a {inv_rec:.2f} % que le client ne remboursera pas son crédit")
