from flask import Blueprint, request, jsonify
#from model.prediction import make_prediction
from model.run_pipeline import make_prediction
import pandas as pd
prediction_app=Blueprint('prediction_route',__name__)

@prediction_app.route('/prediction',methods=['POST'])

def predict():
    json_data=request.get_json()
    df = pd.DataFrame(json_data.get("values"),columns=json_data.get("columns"))
    prediction=make_prediction(df)
    return jsonify({'success':True,"prediction":prediction})
@prediction_app.route('/health', methods = ['GET'])
def health():
    return "hello"





