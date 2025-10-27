import os
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the model and scaler from the models folder
try:
    model = tf.keras.models.load_model(r'models/churn_model.h5')
    scaler = joblib.load(r'models/scaler (1).pkl')
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    raise e

# Define the expected feature order (based on your training)
FEATURE_ORDER = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
    'EstimatedSalary'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        print(f"Received data: {data}")
        
        # Create DataFrame with exact same feature order as training
        input_data = pd.DataFrame([{
            'CreditScore': float(data['CreditScore']),
            'Geography': data['Geography'],
            'Gender': data['Gender'],
            'Age': float(data['Age']),
            'Tenure': float(data['Tenure']),
            'Balance': float(data['Balance']),
            'NumOfProducts': float(data['NumOfProducts']),
            'HasCrCard': float(data['HasCrCard']),
            'IsActiveMember': float(data['IsActiveMember']),
            'EstimatedSalary': float(data['EstimatedSalary'])
        }])
        
        # Ensure correct column order
        input_data = input_data[FEATURE_ORDER]
        print(f"Input data: {input_data}")
        
        # Preprocess the input data (EXACTLY as done during training)
        # Encode categorical variables
        input_data['Geography'] = input_data['Geography'].map({'France': 0, 'Germany': 1, 'Spain': 2})
        input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1})
        
        # Handle any missing values that might occur from mapping
        if input_data.isnull().any().any():
            missing_cols = input_data.columns[input_data.isnull().any()].tolist()
            return jsonify({
                'error': f'Missing or invalid values in columns: {missing_cols}',
                'status': 'error'
            })
        
        # Scale the features
        input_scaled = scaler.transform(input_data)
        print(f"Scaled data shape: {input_scaled.shape}")
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probability = prediction[0][0]
        
        # Convert to binary prediction
        churn_prediction = 'Yes' if probability > 0.5 else 'No'
        confidence = probability if churn_prediction == 'Yes' else 1 - probability
        
        result = {
            'prediction': churn_prediction,
            'probability': float(probability),
            'confidence': float(confidence),
            'confidence_percentage': f"{(confidence * 100):.1f}%",
            'status': 'success'
        }
        
        print(f"Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(error_msg)
        return jsonify({
            'error': error_msg,
            'status': 'error'
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify model is loaded and working"""
    try:
        # Test with sample data
        sample_data = pd.DataFrame([{
            'CreditScore': 650,
            'Geography': 'France',
            'Gender': 'Male',
            'Age': 45,
            'Tenure': 3,
            'Balance': 125000.50,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 150000.00
        }])[FEATURE_ORDER]
        
        # Preprocess
        sample_data['Geography'] = sample_data['Geography'].map({'France': 0, 'Germany': 1, 'Spain': 2})
        sample_data['Gender'] = sample_data['Gender'].map({'Male': 0, 'Female': 1})
        
        input_scaled = scaler.transform(sample_data)
        prediction = model.predict(input_scaled)
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'message': 'Churn prediction model is ready',
            'test_prediction': float(prediction[0][0])
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Endpoint for batch predictions"""
    try:
        data = request.get_json()
        if not data or 'customers' not in data:
            return jsonify({'error': 'No customer data provided'}), 400
        
        results = []
        for i, customer in enumerate(data['customers']):
            try:
                input_data = pd.DataFrame([customer])[FEATURE_ORDER]
                
                # Preprocess
                input_data['Geography'] = input_data['Geography'].map({'France': 0, 'Germany': 1, 'Spain': 2})
                input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1})
                
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                probability = prediction[0][0]
                
                churn_prediction = 'Yes' if probability > 0.5 else 'No'
                confidence = probability if churn_prediction == 'Yes' else 1 - probability
                
                results.append({
                    'customer_id': i,
                    'churn_prediction': churn_prediction,
                    'probability': float(probability),
                    'confidence': float(confidence),
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'customer_id': i,
                    'error': str(e),
                    'status': 'error'
                })
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)