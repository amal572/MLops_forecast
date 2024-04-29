import pickle

from flask import Flask, request, jsonify

import mlflow
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np

import xgboost as xgb


RUN_ID = 'f4e87216c9d5474f93b07dd3b43898a3'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


logged_model = f'runs:/{RUN_ID}/models_mlflow'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
    #data = pd.read_csv('test.csv')
    X_test = ride[['COMPANY_KEY','COMPANY_ITEM_KEY','BILLING_COUNTRY_KEY','APPLE_YEAR','APPLE_WEEK','STOCK_QUANTITY','INVOICED_QUANTITY','IN_TRANSIT_QUANTITY'
    ,'SALES_MIN_WINDOW','SALES_MAX_WINDOW','SALES_MEAN_WINDOW','SALES_last_WINDOW','lag1','lag2'
    ,'lag3','lag4','lag5','lag6','lag7','lag8','lag9','lag10','lag11','lag12']]
    y_test = ride["SELL_OUT_QUANTITY"]
    return X_test

def predict(data):
    y_pred = model.predict(np.float32(data))
    return y_pred

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():

    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'SELL_OUT_QUANTITY': pred
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

