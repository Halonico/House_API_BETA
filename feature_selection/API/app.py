from flask import Flask
from ModelView import prediction_app
def create_app():
    server = Flask('house_prediction')
    server.register_blueprint(prediction_app)
    return server