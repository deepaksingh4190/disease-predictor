from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('../models/model.pkl')

@app.route('/')
def home():
    return "Disease Prediction API is running!"
@app.route('/predict', methods=['POST'])
def predict():
    # Symptom to number mapping (same as during training)
    symptom_mapping = {
        "itching": 0,
        "skin_rash": 1,
        "nodal_skin_eruptions": 2,
        "continuous_sneezing": 3,
        "shivering": 4,
        "chills": 5,
        "watering_from_eyes": 6,
        "stomach_pain": 7,
        "acidity": 8,
        "ulcers_on_tongue": 9,
        "vomiting": 10,
        "cough": 11,
        "chest_pain": 12,
        "headache": 13,
        "dizziness": 14,
        "loss_of_balance": 15,
        "lack_of_concentration": 16,
        "fatigue": 17,
        "weight_loss": 18,
        "restlessness": 19,
        "lethargy": 20,
        "irregular_sugar_level": 21,
        "": 22  # empty/default
    }

    try:
        data = request.get_json()

        # Read symptoms from request (default to empty string)
        symptoms = [
            data.get('Symptom_1', ''),
            data.get('Symptom_2', ''),
            data.get('Symptom_3', ''),
            data.get('Symptom_4', ''),
            data.get('Symptom_5', ''),
            data.get('Symptom_6', ''),
            data.get('Symptom_7', ''),
            data.get('Symptom_8', ''),
            data.get('Symptom_9', ''),
            data.get('Symptom_10', ''),
            data.get('Symptom_11', ''),
            data.get('Symptom_12', ''),
            data.get('Symptom_13', ''),
            data.get('Symptom_14', ''),
            data.get('Symptom_15', ''),
            data.get('Symptom_16', ''),
            data.get('Symptom_17', ''),
        ]

        # Encode symptoms using the mapping
        symptoms_encoded = [symptom_mapping.get(s, 22) for s in symptoms]
        symptoms_encoded = np.array(symptoms_encoded).reshape(1, -1)

        # Predict the disease
        prediction = model.predict(symptoms_encoded)

        # Map prediction to actual disease name
        disease_mapping = {
            0: 'Allergy',
            1: 'Fungal infection',
            2: 'GERD'
        }

        return jsonify({'predicted_disease': disease_mapping[int(prediction[0])]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
