from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import json

app = Flask(__name__)

# Initialize global variables
model = None
scaler = None
trained_features = []
needs_scaling = False
model_metadata = {}

# Additional global variables for purchase clusters
purchase_model = None
purchase_model_loaded = False

def load_model_files():
    global model, scaler, trained_features, needs_scaling, model_metadata
    
    print("Checking for model files...")
    
    # Check if files exist
    required_files = ['demographic_cluster_model.pkl', 'demographic_scaler.pkl']
    missing_files = []

    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing model files: {missing_files}")
        print("Please run train_model.py to create the required files first!")
        return False
    
    try:
        # Load model and scaler
        model = joblib.load('demographic_cluster_model.pkl')
        scaler = joblib.load('demographic_scaler.pkl')
        print("Model and scaler loaded successfully")
        
        # Load features file if it exists
        if os.path.exists('model_features.txt'):
            with open('model_features.txt', 'r') as f:
                trained_features = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(trained_features)} features from file")
        else:
            # Use default features if file doesn't exist
            trained_features = [
                'Year_Birth', 'Teenhome', 'Kidhome', 'Income', 'Education',
                'Marital_Divorced', 'Marital_Married', 'Marital_Single', 
                'Marital_Together', 'Marital_Widow'
            ]
            print(f"Using default features: {len(trained_features)} features")
        
        # Load metadata if it exists
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                model_metadata = json.load(f)
                needs_scaling = model_metadata.get('needs_scaling', False)
            print(f"Model type: {model_metadata.get('model_type', 'Unknown')}")
        else:
            needs_scaling = False
            print("No metadata file found, assuming no scaling needed")
        
        print(f"Model expects {len(trained_features)} features:")
        for i, feature in enumerate(trained_features):
            print(f"  {i+1}. {feature}")
        print(f"Needs scaling: {needs_scaling}")
        
        return True
        
    except Exception as e:
        print(f"Error loading model files: {e}")
        model = None
        scaler = None
        trained_features = []
        return False

# Load purchase cluster model
def load_purchase_model():
    global purchase_model, purchase_model_loaded

    print("Loading purchase cluster model...")
    model_path = 'purchase_clusters/model/random_forest_v3_model.joblib'

    try:
        if os.path.exists(model_path):
            purchase_model = joblib.load(model_path)
            purchase_model_loaded = True
            print("Purchase cluster model loaded successfully")
            return True
        else:
            print(f"Purchase model file not found at {model_path}")
            purchase_model_loaded = False
            return False
    except Exception as e:
        print(f"Error loading purchase model: {e}")
        purchase_model_loaded = False
        return False

# Load model on startup
model_loaded = load_model_files()
purchase_model_loaded = load_purchase_model()

# Define cluster labels
cluster_labels = {
    0: "Young Singles/Couples", 
    1: "Middle-aged Small Families",
    2: "Senior Singles/Couples", 
    3: "Middle-aged Large Families"
}

# Purchase cluster descriptions
purchase_cluster_descriptions = {
    0: "Budget-conscious customers with infrequent purchases primarily in discounts and promotions.",
    1: "Regular shoppers with balanced category purchases and moderate spending patterns.",
    2: "Premium buyers with high purchase frequency, especially in wines and gourmet products.",
}

def predict_customer_segment(year_birth, teenhome, kidhome, income, education, marital_status='Single'):
    """Predict demographic cluster for a new customer"""
    if model is None:
        print("Model is not loaded")
        return None, None
    
    try:
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
            'Year_Birth': year_birth,
            'Teenhome': teenhome,
            'Kidhome': kidhome,
            'Income': income,
            'Education': education,
            **marital_dummies
        }
        
        print(f"Input features: {features_dict}")
        
        # Create feature array in the same order as training
        features = [features_dict.get(feature, 0) for feature in trained_features]
        features_array = np.array(features).reshape(1, -1)
        
        print(f"Feature array shape: {features_array.shape}")
        print(f"Feature values: {features}")
        
        # Apply scaling if needed
        if needs_scaling and scaler is not None:
            print("Applying scaling...")
            features_array = scaler.transform(features_array)
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        
        print(f"Prediction successful: Cluster {prediction}")
        print(f"Probabilities: {probabilities}")
        
        return int(prediction), probabilities.tolist()
        
    except Exception as e:
        print(f"Prediction error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None

# Purchase prediction function
def predict_purchase_cluster(year_birth, kidhome, teenhome, marital_status, education, income):
    """Predict purchase cluster for a customer using the purchase_clusters model"""
    if not purchase_model_loaded:
        print("Purchase model is not loaded")
        return None, None

    try:
        # Create input data dictionary (raw features)
        data_dict = {
            'Year_Birth': [int(year_birth)],
            'Kidhome': [int(kidhome)],
            'Teenhome': [int(teenhome)],
            'Marital_Status': [marital_status],
            'Education': [int(education)],  # Already ordinal encoded from form
            'Income': [float(income)]
        }

        print(f"Purchase prediction input: {data_dict}")

        # Convert to DataFrame
        df = pd.DataFrame(data_dict)

        # Make prediction using the full pipeline
        prediction = purchase_model.predict(df)[0]

        # Get prediction probabilities for confidence
        X_processed = purchase_model.named_steps['preprocessing'].transform(df)
        probabilities = purchase_model.named_steps['classifier'].predict_proba(X_processed)[0]
        confidence = probabilities.max() * 100

        print(f"Purchase cluster prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}%")

        return int(prediction), confidence

    except Exception as e:
        print(f"Purchase prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

@app.route('/')
def landing():
    """Landing page with segmentation options"""
    return render_template('index.html')

@app.route('/demographics')
def demographics():
    """Demographics segmentation page"""
    return render_template('demographics.html', model_loaded=model_loaded)

@app.route('/spending')
def spending():
    """Spending behavior segmentation page"""
    return render_template('spending.html')

@app.route('/behavior')
def behavior():
    """Purchase behavior segmentation page"""
    return render_template('behavior.html')

@app.route('/lifecycle')
def lifecycle():
    """Customer lifecycle segmentation page"""
    return render_template('lifecycle.html')

@app.route('/purchase')
def purchase_page():
    """Purchase cluster prediction page"""
    return render_template('purchase.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        print(f"Received prediction request: {data}")
        
        year_birth = int(data['year_birth'])
        teenhome = int(data['teenhome'])
        kidhome = int(data['kidhome'])
        income = float(data['income'])
        education = float(data['education'])
        marital_status = data.get('marital_status', 'Single')
        
        cluster, probabilities = predict_customer_segment(
            year_birth, teenhome, kidhome, income, education, marital_status
        )
        
        if cluster is None:
            return jsonify({'error': 'Model not loaded or prediction failed'}), 500
        
        response = {
            'cluster': cluster,
            'cluster_label': cluster_labels.get(cluster, 'Unknown'),
            'confidence': round(probabilities[cluster], 3),
            'all_probabilities': {f'Cluster {i}': round(prob, 3) for i, prob in enumerate(probabilities)},
            'input_data': {
                'year_birth': year_birth, 'teenhome': teenhome, 'kidhome': kidhome,
                'income': income, 'education': education, 'marital_status': marital_status
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Handle form submission from HTML"""
    try:
        print("Received form submission")
        
        year_birth = int(request.form['year_birth'])
        teenhome = int(request.form['teenhome'])
        kidhome = int(request.form['kidhome'])
        income = float(request.form['income'])
        education = float(request.form['education'])
        marital_status = request.form.get('marital_status', 'Single')
        
        print(f"Form data: year_birth={year_birth}, teenhome={teenhome}, kidhome={kidhome}")
        print(f"Form data: income={income}, education={education}, marital={marital_status}")
        
        cluster, probabilities = predict_customer_segment(
            year_birth, teenhome, kidhome, income, education, marital_status
        )
        
        if cluster is None:
            error_msg = 'Model not loaded or prediction failed. Please check if model files exist.'
            return render_template('demographics.html', error=error_msg, model_loaded=model_loaded)
        
        result = {
            'cluster': cluster,
            'cluster_label': cluster_labels.get(cluster, 'Unknown'),
            'confidence': round(probabilities[cluster], 3),
            'all_probabilities': {f'Cluster {i}': round(prob, 3) for i, prob in enumerate(probabilities)}
        }
        
        print(f"Prediction result: {result}")
        
        return render_template('demographics.html', result=result, input_data=request.form, model_loaded=model_loaded)
    
    except Exception as e:
        print(f"Form prediction error: {e}")
        import traceback
        traceback.print_exc()
        return render_template('demographics.html', error=str(e), model_loaded=model_loaded)

@app.route('/predict_purchase', methods=['POST'])
def predict_purchase():
    """Predict purchase cluster based on form data"""
    if request.method == 'POST':
        try:
            # Get form data
            year_birth = request.form.get('Year_Birth')
            kidhome = request.form.get('Kidhome')
            teenhome = request.form.get('Teenhome')
            marital_status = request.form.get('Marital_Status')
            education = request.form.get('Education')
            income = request.form.get('Income')

            # Validate inputs
            if not all([year_birth, kidhome, teenhome, marital_status, education, income]):
                return jsonify({
                    'success': False,
                    'message': 'All fields are required'
                })

            # Make prediction
            cluster, confidence = predict_purchase_cluster(
                year_birth, kidhome, teenhome, marital_status, education, income
            )

            if cluster is not None:
                # Get cluster description
                description = purchase_cluster_descriptions.get(cluster, "Unknown purchase pattern")

                return jsonify({
                    'success': True,
                    'cluster': int(cluster),
                    'description': description,
                    'confidence': round(float(confidence), 2)
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Prediction failed'
                })

        except Exception as e:
            print(f"Error in purchase prediction route: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            })

@app.route('/api/info')
def model_info():
    """API endpoint for model information"""
    info = {
        'model_type': model_metadata.get('model_type', 'Unknown'),
        'features': trained_features,
        'clusters': cluster_labels,
        'model_loaded': model is not None,
        'needs_scaling': needs_scaling,
        'n_features': len(trained_features),
        'files_exist': {
            'model': os.path.exists('demographic_cluster_model.pkl'),
            'scaler': os.path.exists('demographic_scaler.pkl'),
            'features': os.path.exists('model_features.txt'),
            'metadata': os.path.exists('model_metadata.json')
        }
    }
    return jsonify(info)

@app.route('/api/reload_model')
def reload_model():
    """Reload model files"""
    success = load_model_files()
    return jsonify({
        'success': success,
        'model_loaded': model is not None,
        'message': 'Model reloaded successfully' if success else 'Failed to reload model'
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created templates directory")
    
    print("Starting Flask app...")
    print("Demographic Customer Segmentation API")
    print("Visit: http://localhost:5000")
    print(f"Model loaded: {model_loaded}")
    
    if not model_loaded:
        print("\nWARNING: Model not loaded!")
        print("To fix this:")
        print("   1. Run your train_model.py script")
        print("   2. Make sure it creates the required files")
        print("   3. Restart this Flask app")
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except KeyboardInterrupt:
        print("\nFlask app stopped by user")
    except Exception as e:
        print(f"Error running Flask app: {e}")