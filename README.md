# Diabetes Prediction API

A simple and lightweight REST API that predicts the likelihood of diabetes using a machine learning model.

## 🚀 Features

- Predict diabetes risk based on user input
- Built using Flask
- Integrated with a trained ML model (e.g., Logistic Regression)
- API endpoint for predictions

## 🧠 Model Inputs

The API expects the following input features (as JSON):

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

## 📦 Requirements

- Python 3.8+
- Flask
- scikit-learn
- pandas
- numpy

Install dependencies:

```bash
pip install -r requirements.txt
