def verify_deployment_setup():
    print("🔍 Verifying deployment setup...")
    
    # Check for required packages
    required_packages = ['joblib', 'tensorflow', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    # Now import the packages
    import os
    import joblib
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    
    # Check if model files exist
    required_files = ['churn_model.h5', 'scaler.pkl']
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            missing_files.append(file)
            print(f"❌ {file} not found")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    # Try to load model and scaler
    try:
        model = tf.keras.models.load_model('churn_model.h5')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    try:
        scaler = joblib.load('scaler.pkl')
        print("✅ Scaler loaded successfully")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return False
    
    # Test prediction
    try:
        # Create sample data that matches your model's expected input
        sample_data = pd.DataFrame([{
            'CreditScore': 650,
            'Geography': 0,  # France
            'Gender': 0,     # Male
            'Age': 45,
            'Tenure': 3,
            'Balance': 125000.50,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 150000.00
        }])
        
        # Make sure the columns are in the right order
        expected_columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                           'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                           'EstimatedSalary']
        sample_data = sample_data[expected_columns]
        
        # Scale the data
        input_scaled = scaler.transform(sample_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        print(f"✅ Test prediction successful!")
        print(f"   Prediction: {prediction[0][0]:.4f}")
        print(f"   Interpretation: {'Churn' if prediction[0][0] > 0.5 else 'No Churn'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test prediction failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_deployment_setup()
    if success:
        print("\n🎉 Deployment setup verified successfully!")
    else:
        print("\n💥 Deployment setup has issues that need to be fixed.")