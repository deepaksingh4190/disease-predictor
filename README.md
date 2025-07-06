#  Disease Prediction System using Machine Learning

This project is a **Disease Prediction System** built with **Python, Machine Learning (Random Forest)** and a **Flask API**.  
It predicts possible diseases based on 17 input symptoms given by the user.

---

##  Features

- Accepts up to 17 symptoms as input
- Predicts disease using trained Random Forest model
- REST API built with Flask
- CSV-based dataset
- Model saved using `joblib`
- Jupyter Notebook for data analysis, training, and evaluation

---

##  Folder Structure

disease_predictor/
├── api/
│ └── app.py # Flask API
├── data/
│ └── symptoms_dataset.csv # Input dataset
├── models/
│ └── model.pkl # Trained ML model
├── notebooks/
│ └── disease_model.ipynb # Data exploration & model training
└── README.md # Project overview

## Run the Flask API:

cd api
python3 app.py

API will be available at http://127.0.0.1:5000

## JSON Payload:
{
  "Symptom_1": "itching",
  "Symptom_2": "skin_rash",
  "Symptom_3": "nodal_skin_eruptions",
  "Symptom_4": "",
  "Symptom_5": "",
  "Symptom_6": "",
  "Symptom_7": "",
  "Symptom_8": "",
  "Symptom_9": "",
  "Symptom_10": "",
  "Symptom_11": "",
  "Symptom_12": "",
  "Symptom_13": "",
  "Symptom_14": "",
  "Symptom_15": "",
  "Symptom_16": "",
  "Symptom_17": ""
}

## Response
{
  "predicted_disease": "Fungal infection"
}

## Accuracy
Achieved ~100% accuracy on test data

## Tech Stack
Python

Pandas

Scikit-learn

Flask

Joblib

Jupyter Notebook

## Future Scope
Integrate real-world medical datasets

Add symptom weightage

Deploy on cloud (Render/Heroku)

Build frontend UI for user interaction

Made by Deepak Singh
https://github.com/deepaksingh4190
https://www.linkedin.com/in/deepak-singh-2a448b319/

