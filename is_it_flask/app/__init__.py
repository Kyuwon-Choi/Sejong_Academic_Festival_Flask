from flask import Flask
from flask_cors import CORS
import logging

def create_app():
    app = Flask(__name__)
    CORS(app)  # CORS 설정 추가
    
    logging.basicConfig(level=logging.DEBUG)
    
    with app.app_context():
        from . import routes  # import routes
        
    return app
