AI-Powered Symptom Checker 🩺
A smart health monitoring system that uses deep learning and LLMs to predict diseases based on user symptoms. It provides detailed descriptions, precautionary advice, and emergency severity alerts through an intuitive UI.

🚀 Features
Symptom-based disease prediction using a Keras deep learning model

Integrated datasets for disease descriptions, precautions, and symptom severity

Emergency trigger if symptoms exceed severity threshold

Gemini LLM integration for natural language Q&A on health

Modern UI built with HTML/CSS + Flask backend

🛠 Tech Stack
Frontend: HTML, CSS, JavaScript

Backend: Flask (Python)

ML Model: TensorFlow / Keras

LLM API: Gemini (Google Generative Language API)

Data Handling: Pandas, Scikit-learn

Deployment: Localhost (can be hosted on any platform)

📂 Project Structure
ai-health-symptom-checker/
├── models/
│ ├── vectorizer.pkl
│ ├── label_encoder.pkl
│ ├── symptom_diagnosis_model_keras.h5
├── data/
│ ├── symptom_Description.csv
│ ├── symptom_precaution.csv
│ ├── Symptom-severity.csv
├── ui.html # Frontend UI
├── app.py # Flask backend
├── .env # (for GEMINI_API key)
├── model_training.ipynb
└── README.md
📦 Installation
Clone this repo

Install dependencies
pip install -r requirements.txt
Add your Gemini API Key in .env:
GEMINI_API=your_api_key_here
Run the Flask app:
python app.py

🔍 How It Works
Enter symptoms like fever, cough, headache

App predicts the disease, provides description & precautions

If severity is high, an emergency alert (Call 108) is shown

Use the LLM chat box to ask questions like "What is dengue?"

🤖 LLM Integration
Ask health-related questions using Gemini-powered LLM:

“What does fatigue indicate?”

“How to prevent viral infections?”

📄 License
MIT License – use freely for personal and academic purposes.
