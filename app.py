from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load the saved model
model = pickle.load(open('churnmodel.pkl', 'rb'))

# Define the app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    # Get the data from the request
    data = request.get_json()

    # Log received data for debugging
    print(f"Received data: {data}")

    # Convert numerical values from strings to float
    try:
        data['tenure'] = float(data['tenure'])
        data['MonthlyCharges'] = float(data['MonthlyCharges'])
        data['TotalCharges'] = float(data['TotalCharges'])
    except KeyError as e:
        print(f"Missing key: {e}")
        return jsonify({'error': f'Missing key: {e}'}), 400
    except ValueError as e:
        print(f"Value error: {e}")
        return jsonify({'error': f'Value error: {e}'}), 400

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Prepare the data for prediction
    try:
        le = LabelEncoder()
        categorical_features = ['Contract', 'TechSupport', 'OnlineSecurity']
        for col in categorical_features:
            df[col] = le.fit_transform(df[col])
    except Exception as e:
        # Handle errors during data preparation (e.g., missing keys)
        print(f"Error preparing data: {e}")
        return jsonify({'error': 'Error preparing data'}), 400

    # Ensure all relevant features are present
    relevant_features = ['tenure', 'Contract', 'TechSupport', 'OnlineSecurity', 'MonthlyCharges', 'TotalCharges']
    try:
        df = df[relevant_features]
    except KeyError as e:
        # Handle missing features in the received data
        print(f"Missing feature(s) in request: {e}")
        return jsonify({'error': 'Missing required data'}), 400

    # Make prediction using the model
    prediction = model.predict(df)[0]

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
