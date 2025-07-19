from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import json

app = Flask(__name__)

# Load the trained model, scaler, and metadata
try:
    model = joblib.load('demographic_cluster_model.pkl')
    scaler = joblib.load('demographic_scaler.pkl')
    
    # Load the feature names that were used during training
    with open('model_features.txt', 'r') as f:
        trained_features = [line.strip() for line in f.readlines()]
    
    # Load model metadata
    with open('model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    needs_scaling = model_metadata.get('needs_scaling', False)
    
    print("✅ Model and scaler loaded successfully!")
    print(f"📊 Model type: {model_metadata.get('model_type', 'Unknown')}")
    print(f"📊 Model expects {len(trained_features)} features:")
    for i, feature in enumerate(trained_features):
        print(f"  {i+1}. {feature}")
    print(f"🔧 Needs scaling: {needs_scaling}")
        
except FileNotFoundError as e:
    print(f"❌ Model files not found: {e}")
    model = None
    scaler = None
    trained_features = []
    needs_scaling = False
    model_metadata = {}

# Define cluster labels based on your analysis
cluster_labels = {
    0: "Young Singles/Couples", 
    1: "Middle-aged Small Families",
    2: "Senior Singles/Couples", 
    3: "Middle-aged Large Families"
}

def predict_customer_segment(age, education, income, total_dependents, is_parent, 
                           kidhome=0, teenhome=0, marital_status='Single'):
    """Predict demographic cluster for a new customer"""
    if model is None or scaler is None:
        return None, None
    
    # Create marital status dummy variables
    marital_dummies = {
        'Marital_Divorced': 1 if marital_status == 'Divorced' else 0,
        'Marital_Married': 1 if marital_status == 'Married' else 0,
        'Marital_Single': 1 if marital_status == 'Single' else 0,
        'Marital_Together': 1 if marital_status == 'Together' else 0,
        'Marital_Widow': 1 if marital_status == 'Widow' else 0
    }
    
    # Create feature dictionary
    features_dict = {
        'Age': age,
        'Education': education,
        'Income': income,
        'Total_Dependents': total_dependents,
        'Is_Parent': is_parent,
        'Kidhome': kidhome,
        'Teenhome': teenhome,
        **marital_dummies
    }
    
    # Create feature array in the same order as training
    features = [features_dict.get(feature, 0) for feature in trained_features]
    features_array = np.array(features).reshape(1, -1)
    
    try:
        # Apply scaling if needed
        if needs_scaling:
            features_array = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        
        return int(prediction), probabilities.tolist()
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

@app.route('/')
def home():
    """Home page with form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        age = float(data['age'])
        education = float(data['education'])
        income = float(data['income'])
        total_dependents = int(data['total_dependents'])
        is_parent = int(data['is_parent'])
        kidhome = int(data.get('kidhome', 0))
        teenhome = int(data.get('teenhome', 0))
        marital_status = data.get('marital_status', 'Single')
        
        cluster, probabilities = predict_customer_segment(
            age, education, income, total_dependents, is_parent, 
            kidhome, teenhome, marital_status
        )
        
        if cluster is None:
            return jsonify({'error': 'Model not loaded or prediction failed'}), 500
        
        response = {
            'cluster': cluster,
            'cluster_label': cluster_labels.get(cluster, 'Unknown'),
            'confidence': round(probabilities[cluster], 3),
            'all_probabilities': {f'Cluster {i}': round(prob, 3) for i, prob in enumerate(probabilities)},
            'input_data': {
                'age': age, 'education': education, 'income': income,
                'total_dependents': total_dependents, 'is_parent': is_parent,
                'kidhome': kidhome, 'teenhome': teenhome, 'marital_status': marital_status
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Handle form submission from HTML"""
    try:
        age = float(request.form['age'])
        education = float(request.form['education'])
        income = float(request.form['income'])
        total_dependents = int(request.form['total_dependents'])
        is_parent = int(request.form['is_parent'])
        kidhome = int(request.form.get('kidhome', 0))
        teenhome = int(request.form.get('teenhome', 0))
        marital_status = request.form.get('marital_status', 'Single')
        
        cluster, probabilities = predict_customer_segment(
            age, education, income, total_dependents, is_parent, 
            kidhome, teenhome, marital_status
        )
        
        if cluster is None:
            return render_template('index.html', error='Model not loaded or prediction failed')
        
        result = {
            'cluster': cluster,
            'cluster_label': cluster_labels.get(cluster, 'Unknown'),
            'confidence': round(probabilities[cluster], 3),
            'all_probabilities': {f'Cluster {i}': round(prob, 3) for i, prob in enumerate(probabilities)}
        }
        
        return render_template('index.html', result=result, input_data=request.form)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/api/info')
def model_info():
    """API endpoint for model information"""
    info = {
        'model_type': model_metadata.get('model_type', 'Unknown'),
        'features': trained_features,
        'clusters': cluster_labels,
        'model_loaded': model is not None,
        'needs_scaling': needs_scaling,
        'n_features': len(trained_features)
    }
    return jsonify(info)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Created templates directory")
    
    print("🚀 Starting Flask app...")
    print("📊 Demographic Customer Segmentation API")
    print("🌐 Visit: http://localhost:5000")
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except KeyboardInterrupt:
        print("\n⛔ Flask app stopped by user")
    except Exception as e:
        print(f"❌ Error running Flask app: {e}")