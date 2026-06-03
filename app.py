from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import pandas as pd

app = Flask(__name__)
CORS(app) 

# Load model
model_data = joblib.load('phishing_detector_model.pkl')
if isinstance(model_data, dict):
    model = model_data['model']
    feature_names = model_data['features']
else:
    model = model_data
    feature_names = list(model.feature_names_in_)

# Load examples database
with open('examples.json', 'r') as f:
    examples_list = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    incoming_url = data.get('url', '').strip()
    
    # Match URL to get features (or default to first item for demo purposes)
    matched_example = next((ex for ex in examples_list if incoming_url.startswith(ex['url']) or ex['url'].startswith(incoming_url)), None)
    if not matched_example:
        matched_example = examples_list[0] 
        
    features_dict = matched_example['features']
    input_df = pd.DataFrame([features_dict])[feature_names]
    
    # Predict
    pred_code = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    confidence = float(probabilities[pred_code])
    
    return jsonify({
        'url': incoming_url,
        'status': 'phishing' if pred_code == 1 else 'legitimate',
        'confidence': confidence
    })
