from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import predict_house_price
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests, if needed

# Load the scaler and trained model from .pkl files if they are available
# Example: Uncomment and modify the lines below if you have saved scaler and model files
# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# with open("theta_optimal.pkl", "rb") as f:
#     theta_optimal = pickle.load(f)

# Dummy prediction function (replace with your actual model's prediction function)
def predict_house_price(input_features):
    # Example: Use a dummy prediction value for testing
    # Replace this with actual model prediction logic
    predicted_price = 200000  # Dummy prediction for testing
    return predicted_price

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON data from the request
        data = request.get_json()
        bedrooms = data.get('bedrooms')
        bathrooms = data.get('bathrooms')
        floors = data.get('floors')
        year = data.get('year')

        # Ensure all fields are provided
        if bedrooms is None or bathrooms is None or floors is None or year is None:
            return jsonify({"error": "All input fields are required (bedrooms, bathrooms, floors, year)."}), 400

        # Convert inputs to integers or floats, if needed
        try:
            input_features = [
                int(bedrooms),
                int(bathrooms),
                int(floors),
                int(year)
            ]
        except ValueError:
            return jsonify({"error": "All input values must be valid numbers."}), 400

        # Predict the house price using the model
        predicted_price = predict_house_price(input_features)

        # Return the prediction in JSON format
        return jsonify({"prediction": predicted_price})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == '__main__':
    app.run(debug=True)
