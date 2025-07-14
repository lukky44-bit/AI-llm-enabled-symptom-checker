import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ======== Load ML components ========
with open(r'C:\Users\Lakshan\Desktop\ai-health project\models\modelsvectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open(r'C:\Users\Lakshan\Desktop\ai-health project\models\label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

model = load_model(r'C:\Users\Lakshan\Desktop\ai-health project\models\symptom_diagnosis_model_keras.h5')

# ======== Load Datasets ========
desc_df = pd.read_csv(r'data/symptom_Description.csv')
precaution_df = pd.read_csv(r'data/symptom_precaution.csv')
severity_df = pd.read_csv(r'C:\Users\Lakshan\Desktop\ai-health project\data\Symptom-severity.csv')

# Lowercase for matching
desc_df['Disease'] = desc_df['Disease'].str.lower().str.strip()
precaution_df['Disease'] = precaution_df['Disease'].str.lower().str.strip()
severity_df['Symptom'] = severity_df['Symptom'].str.lower().str.strip()

# Convert to dictionaries
desc_dict = dict(zip(desc_df['Disease'], desc_df['Description']))
precaution_dict = {
    row['Disease']: [row[col] for col in precaution_df.columns if col != 'Disease' and pd.notna(row[col])]
    for idx, row in precaution_df.iterrows()
}
severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))

# ======== Prediction Function ========
def predict_disease():
    input_text = entry.get().strip()
    if not input_text:
        messagebox.showwarning("Input Error", "Please enter symptoms.")
        return

    symptoms_list = [s.strip().lower() for s in input_text.split(',') if s.strip()]
    vector = vectorizer.transform([" ".join(symptoms_list)]).toarray()
    prediction = model.predict(vector)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]
    disease_key = predicted_disease.lower().strip()

    # Get description & precautions
    desc = desc_dict.get(disease_key, "Description not available.")
    precautions = precaution_dict.get(disease_key, ["No precautions available."])

    # Calculate total severity
    total_severity = sum(severity_dict.get(symptom, 0) for symptom in symptoms_list)
    emergency = total_severity >= 13  # Adjust threshold as needed

    result = f"🔍 Predicted Disease: {predicted_disease}\n\n📖 Description:\n{desc}\n\n💡 Precautions:\n- " + "\n- ".join(precautions)

    if emergency:
        result += "\n\n🚨 Emergency Severity Detected!"
        emergency_button.pack(pady=10)
    else:
        emergency_button.pack_forget()

    output_label.config(text=result)

# ======== Emergency Call Placeholder ========
def call_emergency():
    messagebox.showinfo("Emergency", "📞 Calling Emergency Services (108)...")

# ======== GUI Setup ========
root = tk.Tk()
root.title("AI Symptom Checker")
root.geometry("650x600")

tk.Label(root, text="Enter Symptoms (comma-separated):", font=("Arial", 14)).pack(pady=10)
entry = tk.Entry(root, width=70, font=("Arial", 12))
entry.pack(pady=10)

tk.Button(root, text="Predict Disease", command=predict_disease, font=("Arial", 12), bg="green", fg="white").pack(pady=10)

output_label = tk.Label(root, text="", wraplength=600, justify="left", font=("Arial", 12))
output_label.pack(pady=20)

emergency_button = tk.Button(root, text="🚨 Call Emergency (108)", font=("Arial", 12), bg="red", fg="white", command=call_emergency)

root.mainloop()
