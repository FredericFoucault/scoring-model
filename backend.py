from flask import Flask, render_template, jsonify,request
import pandas as pd
import json
import joblib

app = Flask(__name__)

#load the data
df = pd.read_csv("app_clean_final.csv")
dictionnaire = {}
for colonne in df.columns:
    dictionnaire[colonne] = df[colonne].tolist()
 


@app.route('/data',methods =["GET"])
def index():
    return jsonify(dictionnaire)

dt = pd.DataFrame(dictionnaire)
dt=dt.drop(['SK_ID_CURR','TARGET'],axis=1)



@app.route('/', methods=['GET'])
def hello():
    return jsonify({"hello":"Fred"})


def load_model():
    model = joblib.load('model_scoring_LGBM.joblib')
    return model

# chargelement du model
model = load_model()


# prediction
@app.route('/predict', methods=['GET','POST'])
def predict():
    # selection des données du client
    ClientID = request.args.get('ClientID')
    pred = model.predict_proba(dt[dt.index == int(float(ClientID))])[:,1].tolist()

    rec = pred[0] * 100
    return jsonify({"prediction": rec })


if __name__ == "__main__":
    #app.run(host="0.0.0.0",port=8080)
    app.run()

