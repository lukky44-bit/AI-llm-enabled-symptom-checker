import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# Load environment variables from .env
load_dotenv()

# ======== Load ML components ========
with open(r'models/modelsvectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open(r'models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

model = load_model(r'models/symptom_diagnosis_model_keras.h5')

# ======== Load Datasets ========
desc_df = pd.read_csv(r'data/symptom_Description.csv')
precaution_df = pd.read_csv(r'data/symptom_precaution.csv')
severity_df = pd.read_csv(r'data/Symptom-severity.csv')

desc_df['Disease'] = desc_df['Disease'].str.lower().str.strip()
precaution_df['Disease'] = precaution_df['Disease'].str.lower().str.strip()
severity_df['Symptom'] = severity_df['Symptom'].str.lower().str.strip()

desc_dict = dict(zip(desc_df['Disease'], desc_df['Description']))
precaution_dict = {
    row['Disease']: [row[col] for col in precaution_df.columns if col != 'Disease' and pd.notna(row[col])]
    for idx, row in precaution_df.iterrows()
}
severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))

GEMINI_API_KEY = os.getenv('GEMINI_API')
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

def gemini_llm_query(prompt):
    if not GEMINI_API_KEY:
        return "Gemini API key not set."
    url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    if not symptoms:
        return jsonify({'error': 'No symptoms provided.'}), 400

    symptoms_list = [s.strip().lower() for s in symptoms.split(',') if s.strip()]
    vector = vectorizer.transform([" ".join(symptoms_list)]).toarray()
    prediction = model.predict(vector)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]
    disease_key = predicted_disease.lower().strip()

    desc = desc_dict.get(disease_key, "Description not available.")
    precautions = precaution_dict.get(disease_key, ["No precautions available."])
    total_severity = sum(severity_dict.get(symptom, 0) for symptom in symptoms_list)
    emergency = total_severity >= 13

    return jsonify({
        'predicted_disease': predicted_disease,
        'description': desc,
        'precautions': precautions,
        'emergency': emergency
    })

@app.route('/llm', methods=['POST'])
def llm():
    data = request.get_json()
    user_input = data.get('input', '')
    if not user_input:
        return jsonify({'error': 'No input provided.'}), 400
    try:
        answer = gemini_llm_query(user_input)
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
