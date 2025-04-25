from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
import pickle
import numpy as np
from faker import Faker

app = Flask(__name__)

# Setup JWT
app.config['JWT_SECRET_KEY'] = 'your_secret_key'
jwt = JWTManager(app)

# Load the trained model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize faker
fake = Faker()

# Login route to get token
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    if username == 'admin' and password == 'password':
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token), 200

    return jsonify(message='Invalid credentials'), 401

# Prediction route
@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    data = request.json

    features = np.array([[  
    data['pregnancies'],
    data['glucose_level'],
    data['blood_pressure'],
    data['skin_thickness'],
    data['insulin'],
    data['bmi'],
    data['diabetes_pedigree'],
    data['age']
]])

    
    prediction = model.predict(features)
    result = "Diabetes" if prediction[0] == 1 else "Non Diabetes"
    return jsonify(result=result)

# Route to generate fake data
@app.route('/generate-fake-data', methods=['GET'])
def generate_fake_data():
    fake_data = {
        'age': fake.random_int(min=20, max=80),
        'bmi': fake.random_int(min=15, max=45),
        'blood_pressure': fake.random_int(min=50, max=180),
        'glucose_level': fake.random_int(min=70, max=200),
        'insulin': fake.random_int(min=10, max=300),
        'skin_thickness': fake.random_int(min=10, max=60)
    }
    return jsonify(fake_data)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
