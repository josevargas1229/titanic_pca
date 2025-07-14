from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust origin for deployment

# Load the model and components
try:
    model = joblib.load('titanic_model.pkl')
    pca = joblib.load('pca_transformer.pkl')
    age_model = joblib.load('age_model.pkl')
    imputer_embarked = joblib.load('imputer_embarked.pkl')
    imputer_cabin = joblib.load('imputer_cabin.pkl')
    encoder_sex = joblib.load('encoder_sex.pkl')
    encoder_embarked = joblib.load('encoder_embarked.pkl')
    encoder_deck = joblib.load('encoder_deck.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    class_names = joblib.load('class_names.pkl')
    print("Model and components loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Could not load model or components. Ensure export_models.py has been run.")
    print(f"Missing file: {e}")
    model = None

@app.route('/')
def home():
    """Main page with service information"""
    return {
        "message": "Titanic Survival Prediction Service",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Make a single prediction",
            "/health": "GET - Check service status",
            "/model-info": "GET - Model information"
        }
    }

@app.route('/health')
def health():
    """Endpoint to check service status"""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    return jsonify({"status": "ok", "message": "Service running correctly"})

@app.route('/model-info')
def model_info():
    """Information about the model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_type": "KNeighborsClassifier",
        "n_neighbors": model.n_neighbors,
        "pca_components": pca.n_components_,
        "feature_names": feature_names,
        "classes": class_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction for a single passenger"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        if 'features' not in data:
            return jsonify({
                "error": "Field 'features' is required",
                "example": {
                    "features": [3, 30.0, 8.05, "Unknown", "male"]
                },
                "feature_names": ["Pclass", "Age", "Fare", "Cabin", "Sex"]
            }), 400
        
        features = data['features']
        if len(features) != 5:
            return jsonify({
                "error": f"Exactly 5 features required, got {len(features)}",
                "feature_names": ["Pclass", "Age", "Fare", "Cabin", "Sex"]
            }), 400
        
        # Create DataFrame with input feature names
        input_feature_names = ["Pclass", "Age", "Fare", "Cabin", "Sex"]
        sample_df = pd.DataFrame([features], columns=input_feature_names)
        
        # Handle missing Age
        if pd.isna(sample_df['Age']).any():
            age_features = ['Fare', 'Pclass', 'Cabin']
            temp_df = sample_df[age_features].copy()
            temp_df[['Cabin']] = imputer_cabin.transform(temp_df[['Cabin']])
            temp_df['Cabin'] = temp_df['Cabin'].str[0]
            temp_df[['Cabin']] = encoder_deck.transform(temp_df[['Cabin']])
            sample_df.loc[pd.isna(sample_df['Age']), 'Age'] = age_model.predict(temp_df)
        
        # Preprocess the input
        sample_df[['Cabin']] = imputer_cabin.transform(sample_df[['Cabin']])
        sample_df['Cabin'] = sample_df['Cabin'].str[0]
        sample_df[['Cabin']] = encoder_deck.transform(sample_df[['Cabin']])
        
        # Encode Sex
        encoded_sex = encoder_sex.transform(sample_df[['Sex']])
        encoded_sex_cols = encoder_sex.get_feature_names_out(['Sex'])
        sample_df.drop(columns=['Sex'], inplace=True)
        sample_df[encoded_sex_cols] = encoded_sex
        
        # Select final features
        final_features = ['Sex_female', 'Age', 'Fare', 'Pclass', 'Cabin']
        sample_df = sample_df[final_features]
        
        # Scale features
        sample_df = scaler.transform(sample_df)
        
        # Apply PCA
        sample_pca = pca.transform(sample_df)
        
        # Make prediction
        prediction = model.predict(sample_pca)[0]
        probabilities = model.predict_proba(sample_pca)[0]
        
        # Create response
        response = {
            "input_features": {
                name: value for name, value in zip(input_feature_names, features)
            },
            "prediction": {
                "class": class_names[prediction],
                "class_index": int(prediction)
            },
            "probabilities": {
                class_name: float(prob) for class_name, prob in zip(class_names, probabilities)
            },
            "confidence": float(max(probabilities))
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Make predictions for multiple passengers"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        if not data or 'samples' not in data:
            return jsonify({
                "error": "Field 'samples' is required",
                "example": {
                    "samples": [
                        [3, 30.0, 8.05, "Unknown", "male"],
                        [1, 25.0, 71.2833, "C85", "female"]
                    ]
                },
                "feature_names": ["Pclass", "Age", "Fare", "Cabin", "Sex"]
            }), 400
        
        samples = np.array(data['samples'])
        if samples.shape[1] != 5:
            return jsonify({
                "error": f"Each sample must have exactly 5 features",
                "feature_names": ["Pclass", "Age", "Fare", "Cabin", "Sex"]
            }), 400
        
        # Create DataFrame
        input_feature_names = ["Pclass", "Age", "Fare", "Cabin", "Sex"]
        samples_df = pd.DataFrame(samples, columns=input_feature_names)
        
        # Handle missing Age
        mask = pd.isna(samples_df['Age'])
        if mask.any():
            age_features = ['Fare', 'Pclass', 'Cabin']
            temp_df = samples_df[age_features].copy()
            temp_df[['Cabin']] = imputer_cabin.transform(temp_df[['Cabin']])
            temp_df['Cabin'] = temp_df['Cabin'].str[0]
            temp_df[['Cabin']] = encoder_deck.transform(temp_df[['Cabin']])
            samples_df.loc[mask, 'Age'] = age_model.predict(temp_df[mask])
        
        # Preprocess
        samples_df[['Cabin']] = imputer_cabin.transform(samples_df[['Cabin']])
        samples_df['Cabin'] = samples_df['Cabin'].str[0]
        samples_df[['Cabin']] = encoder_deck.transform(samples_df[['Cabin']])
        
        # Encode Sex
        encoded_sex = encoder_sex.transform(samples_df[['Sex']])
        encoded_sex_cols = encoder_sex.get_feature_names_out(['Sex'])
        samples_df.drop(columns=['Sex'], inplace=True)
        samples_df[encoded_sex_cols] = encoded_sex
        
        # Select final features
        final_features = ['Sex_female', 'Age', 'Fare', 'Pclass', 'Cabin']
        samples_df = samples_df[final_features]
        
        # Scale features
        samples_df = scaler.transform(samples_df)
        
        # Apply PCA
        samples_pca = pca.transform(samples_df)
        
        # Make predictions
        predictions = model.predict(samples_pca)
        probabilities = model.predict_proba(samples_pca)
        
        # Create response
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append({
                "sample_index": i,
                "input_features": samples[i].tolist(),
                "prediction": {
                    "class": class_names[pred],
                    "class_index": int(pred)
                },
                "probabilities": {
                    class_name: float(prob) for class_name, prob in zip(class_names, probs)
                },
                "confidence": float(max(probs))
            })
        
        return jsonify({
            "results": results,
            "total_samples": len(results)
        })
        
    except Exception as e:
        return jsonify({"error": f"Batch prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    required_files = [
        'titanic_model.pkl', 'pca_transformer.pkl', 'age_model.pkl',
        'imputer_embarked.pkl', 'imputer_cabin.pkl', 'encoder_sex.pkl',
        'encoder_embarked.pkl', 'encoder_deck.pkl', 'scaler.pkl',
        'feature_names.pkl', 'class_names.pkl'
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"- {file}")
        print("\nRun export_models.py first to generate these files.")
    else:
        print("All model files are present.")
        print("Starting Flask server...")
        print("Available endpoints:")
        print("- GET /: General information")
        print("- GET /health: Service status")
        print("- GET /model-info: Model information")
        print("- POST /predict: Single prediction")
        print("- POST /predict-batch: Batch prediction")
    
    app.run(debug=True, host='0.0.0.0', port=5000)