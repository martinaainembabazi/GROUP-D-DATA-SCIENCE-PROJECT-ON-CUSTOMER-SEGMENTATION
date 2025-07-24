from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import json
import sys
import pickle

app = Flask(__name__)

# Initialize global variables for demographic segmentation
model = None
scaler = None
trained_features = []
needs_scaling = False
model_metadata = {}

# Initialize global variables for lifecycle segmentation
lifecycle_model = None
lifecycle_scaler = None
lifecycle_label_encoder = None
lifecycle_features = []
lifecycle_metadata = {}
lifecycle_df = None
lifecycle_model_loaded = False

# Initialize global variables for purchase behavior segmentation
purchase_model = None
purchase_scaler = None
purchase_features = ['Income', 'Age', 'Education', 'Marital_Together', 'Marital_Single', 'Marital_Divorced', 'Marital_Widow', 'Marital_Married', 'Total_Dependents']
purchase_model_loaded = False

# Initialize global variables for spending behavior segmentation
spending_model = None
spending_scaler = None
spending_features = ['Income', 'Age', 'Education', 'Marital_Together', 'Marital_Single', 'Marital_Divorced', 'Marital_Widow', 'Marital_Married', 'Total_Dependents']
spending_model_loaded = False

def load_model_files():
    global model, scaler, trained_features, needs_scaling, model_metadata
    
    print("Checking for demographic model files...")
    
    # Check if files exist
    required_files = ['demographic_cluster_model.pkl', 'demographic_scaler.pkl']
    missing_files = []

    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing demographic model files: {missing_files}")
        print("Please run train_model.py to create the required files first!")
        return False
    
    try:
        # Load model and scaler
        model = joblib.load('demographic_cluster_model.pkl')
        scaler = joblib.load('demographic_scaler.pkl')
        print("Demographic model and scaler loaded successfully")
        
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
            print(f"Demographic model type: {model_metadata.get('model_type', 'Unknown')}")
        else:
            needs_scaling = False
            print("No demographic metadata file found, assuming no scaling needed")
        
        print(f"Demographic model expects {len(trained_features)} features:")
        for i, feature in enumerate(trained_features):
            print(f"  {i+1}. {feature}")
        print(f"Needs scaling: {needs_scaling}")
        
        return True
        
    except Exception as e:
        print(f"Error loading demographic model files: {e}")
        model = None
        scaler = None
        trained_features = []
        return False

# Paths for lifecycle model files
LIFECYCLE_MODEL_DIR = os.path.join('CustomerLifeCycle', 'model', 'model')
LIFECYCLE_DATA_PATH = os.path.join('CustomerLifeCycle', 'lifecycle.csv')

def load_lifecycle_model_files():
    global lifecycle_model, lifecycle_scaler, lifecycle_label_encoder, lifecycle_features, lifecycle_df, lifecycle_model_loaded
    print("Checking for lifecycle model files...")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model directory: {LIFECYCLE_MODEL_DIR}")
    print(f"Data file path: {LIFECYCLE_DATA_PATH}")
    
    try:
        # Load lifecycle model and related files
        lifecycle_model_path = os.path.join(LIFECYCLE_MODEL_DIR, 'model.pkl')
        lifecycle_scaler_path = os.path.join(LIFECYCLE_MODEL_DIR, 'lifecycle_scaler.pkl')
        lifecycle_label_encoder_path = os.path.join(LIFECYCLE_MODEL_DIR, 'label_encoder.pkl')
        lifecycle_features_path = os.path.join(LIFECYCLE_MODEL_DIR, 'lifecycle_features.pkl')
        lifecycle_metadata_path = os.path.join(LIFECYCLE_MODEL_DIR, 'lifecycle_model_metadata.pkl')

        # Check if files exist
        missing_files = []
        for p in [lifecycle_model_path, lifecycle_scaler_path, lifecycle_label_encoder_path, lifecycle_features_path, lifecycle_metadata_path]:
            if not os.path.exists(p):
                missing_files.append(p)
            else:
                print(f"Found file: {p}")
        if missing_files:
            print(f"Missing lifecycle model files: {missing_files}")
            print("Please run train_lifecycle_model.py to create the required files first!")
            return False

        # Load lifecycle data
        if os.path.exists(LIFECYCLE_DATA_PATH):
            lifecycle_df = pd.read_csv(LIFECYCLE_DATA_PATH)
            print(f"Loaded lifecycle.csv: {lifecycle_df.shape} rows, columns: {list(lifecycle_df.columns)}")
            print(f"Sample IDs: {lifecycle_df['ID'].head().tolist()}")
        else:
            print(f"Lifecycle data file not found at {LIFECYCLE_DATA_PATH}")
            return False

        # Load each file with specific error handling
        try:
            lifecycle_model = joblib.load(lifecycle_model_path)
            print("Loaded model.pkl successfully")
        except Exception as e:
            print(f"Error loading model.pkl: {str(e)}")
            return False

        try:
            lifecycle_scaler = joblib.load(lifecycle_scaler_path)
            print("Loaded lifecycle_scaler.pkl successfully")
        except Exception as e:
            print(f"Error loading lifecycle_scaler.pkl: {str(e)}")
            return False

        try:
            lifecycle_label_encoder = joblib.load(lifecycle_label_encoder_path)
            print("Loaded label_encoder.pkl successfully")
        except Exception as e:
            print(f"Error loading label_encoder.pkl: {str(e)}")
            return False

        try:
            lifecycle_features = joblib.load(lifecycle_features_path)
            print(f"Loaded lifecycle_features.pkl successfully: {lifecycle_features}")
        except Exception as e:
            print(f"Error loading lifecycle_features.pkl: {str(e)}")
            return False

        try:
            lifecycle_metadata = joblib.load(lifecycle_metadata_path)
            print(f"Loaded lifecycle_model_metadata.pkl successfully: {lifecycle_metadata}")
        except Exception as e:
            print(f"Error loading lifecycle_model_metadata.pkl: {str(e)}")
            return False

        print(f"Lifecycle model expects {len(lifecycle_features)} features: {lifecycle_features}")
        print("Lifecycle model and data loaded successfully!")
        lifecycle_model_loaded = True
        return True

    except Exception as e:
        print(f"Unexpected error loading lifecycle model files: {str(e)}")
        lifecycle_model_loaded = False
        return False

def load_purchase_model_files():
    global purchase_model, purchase_scaler, purchase_model_loaded
    
    print("Checking for purchase model files...")
    
    # Check if files exist
    purchase_model_path = os.path.join('purchase_clusters', 'model', 'random_forest_v3_model.joblib')
    purchase_scaler_path = os.path.join('purchase_clusters', 'model', 'scaler_v3.joblib')
    
    required_files = [purchase_model_path, purchase_scaler_path]
    missing_files = []

    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing purchase model files: {missing_files}")
        print("Please run the purchase model training script to create the required files first!")
        purchase_model_loaded = False
        return False
    
    try:
        # Load model and scaler
        purchase_model = joblib.load(purchase_model_path)
        purchase_scaler = joblib.load(purchase_scaler_path)
        print("Purchase model and scaler loaded successfully")
        
        print(f"Purchase model expects {len(purchase_features)} features:")
        for i, feature in enumerate(purchase_features):
            print(f"  {i+1}. {feature}")
        
        purchase_model_loaded = True
        return True
        
    except Exception as e:
        print(f"Error loading purchase model files: {e}")
        purchase_model = None
        purchase_scaler = None
        purchase_model_loaded = False
        return False

def load_spending_model_files():
    global spending_model, spending_scaler, spending_model_loaded
    
    print("Checking for spending model files...")
    
    # Check if files exist
    spending_model_path = os.path.join('spending_clusters', 'model', 'spending_rf_v3_model.joblib')
    spending_scaler_path = os.path.join('spending_clusters', 'model', 'spending_scaler_v3.joblib')
    
    required_files = [spending_model_path, spending_scaler_path]
    missing_files = []

    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing spending model files: {missing_files}")
        print("Please run the spending model training script to create the required files first!")
        spending_model_loaded = False
        return False
    
    try:
        # Load model and scaler
        spending_model = joblib.load(spending_model_path)
        spending_scaler = joblib.load(spending_scaler_path)
        print("Spending model and scaler loaded successfully")
        
        print(f"Spending model expects {len(spending_features)} features:")
        for i, feature in enumerate(spending_features):
            print(f"  {i+1}. {feature}")
        
        spending_model_loaded = True
        return True
        
    except Exception as e:
        print(f"Error loading spending model files: {e}")
        spending_model = None
        spending_scaler = None
        spending_model_loaded = False
        return False

# Load models on startup
model_loaded = load_model_files()
lifecycle_model_loaded = load_lifecycle_model_files()
purchase_model_loaded = load_purchase_model_files()
spending_model_loaded = load_spending_model_files()

# Define cluster labels for demographic segmentation
cluster_labels = {
    0: "Young Singles/Couples", 
    1: "Middle-aged Small Families",
    2: "Senior Singles/Couples", 
    3: "Middle-aged Large Families"
}

# Define cluster labels for purchase behavior segmentation
purchase_cluster_labels = {
    0: "Deal Seekers",
    1: "Digital-First Shoppers", 
    2: "Premium Multi-Channel Buyers"
}

# Define cluster labels for spending behavior segmentation
spending_cluster_labels = {
    0: "Bronze",
    1: "Gold", 
    2: "Silver"
}

def predict_customer_segment(year_birth, teenhome, kidhome, income, education, marital_status='Single'):
    """Predict demographic cluster for a new customer"""
    if model is None:
        print("Demographic model is not loaded")
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

def predict_purchase_cluster(income, age, education, marital_status='Single', total_dependents=0):
    """Predict purchase cluster for a new customer"""
    if purchase_model is None:
        print("Purchase model is not loaded")
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
            'Income': income,
            'Age': age,
            'Education': education,
            'Total_Dependents': total_dependents,
            **marital_dummies
        }
        
        print(f"Input features: {features_dict}")
        
        # Create feature array in the same order as training
        features = [features_dict.get(feature, 0) for feature in purchase_features]
        features_array = np.array(features).reshape(1, -1)
        
        print(f"Feature array shape: {features_array.shape}")
        print(f"Feature values: {features}")
        
        # Apply scaling
        print("Applying scaling...")
        features_array = purchase_scaler.transform(features_array)
        
        # Make prediction
        print("Making prediction...")
        prediction = purchase_model.predict(features_array)[0]
        probabilities = purchase_model.predict_proba(features_array)[0]
        
        print(f"Prediction successful: Cluster {prediction}")
        print(f"Probabilities: {probabilities}")
        
        return int(prediction), probabilities.tolist()
        
    except Exception as e:
        print(f"Purchase prediction error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_spending_cluster(income, age, education, marital_status='Single', total_dependents=0):
    """Predict spending cluster for a new customer"""
    if spending_model is None:
        print("Spending model is not loaded")
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
            'Income': income,
            'Age': age,
            'Education': education,
            'Total_Dependents': total_dependents,
            **marital_dummies
        }
        
        print(f"Input features: {features_dict}")
        
        # Create feature array in the same order as training
        features = [features_dict.get(feature, 0) for feature in spending_features]
        features_array = np.array(features).reshape(1, -1)
        
        print(f"Feature array shape: {features_array.shape}")
        print(f"Feature values: {features}")
        
        # Apply scaling
        print("Applying scaling...")
        features_array = spending_scaler.transform(features_array)
        
        # Make prediction
        print("Making prediction...")
        prediction = spending_model.predict(features_array)[0]
        probabilities = spending_model.predict_proba(features_array)[0]
        
        print(f"Prediction successful: Cluster {prediction}")
        print(f"Probabilities: {probabilities}")
        
        return int(prediction), probabilities.tolist()
        
    except Exception as e:
        print(f"Spending prediction error: {e}")
        print(f"Error type: {type(e).__name__}")
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
    return render_template('spending.html', model_loaded=spending_model_loaded)

@app.route('/purchase')
def purchase():
    """Purchase behavior segmentation page"""
    return render_template('purchase.html', model_loaded=purchase_model_loaded)

@app.route('/customerlifecycle')
def lifecycle():
    """Customer lifecycle segmentation page"""
    return render_template('customerlifecycle.html', model_loaded=lifecycle_model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for demographic predictions"""
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

@app.route('/predict_purchase', methods=['POST'])
def predict_purchase():
    """API endpoint for purchase cluster predictions"""
    try:
        data = request.get_json() if request.is_json else request.form
        print(f"Received purchase prediction request: {dict(data)}")
        
        # Handle both form and JSON data
        if request.is_json:
            income = float(data['Income'])
            age = int(data.get('Age', 2024 - int(data.get('Year_Birth', 1980))))
            education = float(data['Education'])
            marital_status = data.get('Marital_Status', 'Single')
            total_dependents = int(data.get('Total_Dependents', int(data.get('Kidhome', 0)) + int(data.get('Teenhome', 0))))
        else:
            # Form data
            income = float(data['Income'])
            year_birth = int(data.get('Year_Birth', 1980))
            age = 2024 - year_birth
            education = float(data['Education'])
            marital_status = data.get('Marital_Status', 'Single')
            total_dependents = int(data.get('Kidhome', 0)) + int(data.get('Teenhome', 0))
        
        cluster, probabilities = predict_purchase_cluster(
            income, age, education, marital_status, total_dependents
        )
        
        if cluster is None:
            if request.is_json:
                return jsonify({'error': 'Model not loaded or prediction failed'}), 500
            else:
                error_msg = 'Model not loaded or prediction failed. Please check if model files exist.'
                return render_template('purchase.html', error=error_msg, model_loaded=purchase_model_loaded)
        
        response = {
            'cluster': cluster,
            'cluster_label': purchase_cluster_labels.get(cluster, 'Unknown'),
            'confidence': round(probabilities[cluster], 3),
            'all_probabilities': {f'Cluster {i}': round(prob, 3) for i, prob in enumerate(probabilities)},
            'input_data': {
                'income': income, 'age': age, 'education': education, 
                'marital_status': marital_status, 'total_dependents': total_dependents
            }
        }
        
        if request.is_json:
            return jsonify(response)
        else:
            # Form submission - return HTML page with results
            print(f"Purchase prediction result: {response}")
            return render_template('purchase.html', result=response, input_data=data, model_loaded=purchase_model_loaded)
    
    except Exception as e:
        print(f"Purchase prediction error: {e}")
        import traceback
        traceback.print_exc()
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        else:
            return render_template('purchase.html', error=str(e), model_loaded=purchase_model_loaded)

@app.route('/predict_spending', methods=['POST'])
def predict_spending():
    """API endpoint for spending cluster predictions"""
    try:
        data = request.get_json() if request.is_json else request.form
        print(f"Received spending prediction request: {dict(data)}")
        
        # Handle both form and JSON data
        if request.is_json:
            income = float(data['Income'])
            age = int(data.get('Age', 2024 - int(data.get('Year_Birth', 1980))))
            education = float(data['Education'])
            marital_status = data.get('Marital_Status', 'Single')
            total_dependents = int(data.get('Total_Dependents', int(data.get('Kidhome', 0)) + int(data.get('Teenhome', 0))))
        else:
            # Form data
            income = float(data['Income'])
            year_birth = int(data.get('Year_Birth', 1980))
            age = 2025 - year_birth  # Updated to current year
            education = float(data['Education'])
            marital_status = data.get('Marital_Status', 'Single')
            total_dependents = int(data.get('Kidhome', 0)) + int(data.get('Teenhome', 0))
        
        cluster, probabilities = predict_spending_cluster(
            income, age, education, marital_status, total_dependents
        )
        
        if cluster is None:
            if request.is_json:
                return jsonify({'error': 'Model not loaded or prediction failed'}), 500
            else:
                error_msg = 'Model not loaded or prediction failed. Please check if model files exist.'
                return render_template('spending.html', error=error_msg, model_loaded=spending_model_loaded)
        
        response = {
            'cluster': cluster,
            'cluster_label': spending_cluster_labels.get(cluster, 'Unknown'),
            'confidence': round(probabilities[cluster], 3),
            'all_probabilities': {f'Cluster {i}': round(prob, 3) for i, prob in enumerate(probabilities)},
            'input_data': {
                'income': income, 'age': age, 'education': education, 
                'marital_status': marital_status, 'total_dependents': total_dependents
            }
        }
        
        if request.is_json:
            return jsonify(response)
        else:
            # Form submission - return HTML page with results
            print(f"Spending prediction result: {response}")
            return render_template('spending.html', result=response, input_data=data, model_loaded=spending_model_loaded)
    
    except Exception as e:
        print(f"Spending prediction error: {e}")
        import traceback
        traceback.print_exc()
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        else:
            return render_template('spending.html', error=str(e), model_loaded=spending_model_loaded)

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Handle form submission from HTML for demographic predictions"""
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

@app.route('/api/info')
def model_info():
    """API endpoint for demographic model information"""
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
    """Reload demographic model files"""
    success = load_model_files()
    return jsonify({
        'success': success,
        'model_loaded': model is not None,
        'message': 'Demographic model reloaded successfully' if success else 'Failed to reload demographic model'
    })

@app.route('/fetch_lifecycle', methods=['POST'])
def fetch_lifecycle():
    """API endpoint for lifecycle predictions"""
    print(f"Accessing route: {request.url}")
    if not lifecycle_model_loaded:
        error_msg = 'Lifecycle model or data not loaded'
        print(f"Error: {error_msg}")
        return jsonify({'success': False, 'error': error_msg}), 500
    
    try:
        # Get Recency and Tenure_Days from form or JSON
        recency = request.form.get('Recency') if request.form.get('Recency') else request.get_json().get('Recency') if request.is_json else None
        tenure_days = request.form.get('Tenure_Days') if request.form.get('Tenure_Days') else request.get_json().get('Tenure_Days') if request.is_json else None
        
        if recency is None or tenure_days is None:
            error_msg = 'Recency and Tenure_Days are required'
            print(f"Error: {error_msg}")
            return jsonify({'success': False, 'error': error_msg}), 400
        
        # Convert inputs to float
        try:
            recency = float(recency)
            tenure_days = float(tenure_days)
            if recency < 0 or tenure_days < 0:
                raise ValueError("Recency and Tenure_Days must be non-negative")
        except ValueError as ve:
            error_msg = f"Invalid input: {str(ve)}"
            print(f"Error: {error_msg}")
            return jsonify({'success': False, 'error': error_msg}), 400
        
        # Prepare feature DataFrame
        input_data = {
            'Recency': recency,
            'Tenure_Days': tenure_days
        }
        # Add default values for other features
        for feature in lifecycle_features:
            if feature not in input_data:
                input_data[feature] = lifecycle_df[feature].median()
        
        X = pd.DataFrame([input_data], columns=lifecycle_features)
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(lifecycle_df[col].median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(lifecycle_df[col].median())
        
        # Cap extremely large values
        max_float32 = 3.4e38
        X = X.clip(upper=max_float32, lower=-max_float32)
        
        # Scale features, preserving column names
        X_scaled = pd.DataFrame(lifecycle_scaler.transform(X), columns=lifecycle_features)
        
        # Predict lifecycle stage
        prediction = lifecycle_model.predict(X_scaled)
        lifecycle_stage = lifecycle_label_encoder.inverse_transform(prediction)[0]
        
        # Prepare response
        result = {
            'lifecycle_stage': lifecycle_stage
        }
        print(f"Prediction successful: {result}")
        return jsonify({'success': True, **result}), 200
    
    except Exception as e:
        print(f"Lifecycle prediction error: {str(e)}")
        error_msg = f"Server error: {str(e)}"
        return jsonify({'success': False, 'error': error_msg}), 500


@app.route('/api/lifecycle_stats', methods=['GET'])
def lifecycle_stats():
    """API endpoint for lifecycle stage distribution"""
    if not lifecycle_model_loaded:
        return jsonify({'success': False, 'error': 'Lifecycle model or data not loaded'})
    
    try:
        # Calculate distribution of lifecycle stages
        stage_counts = lifecycle_df['Lifecycle_Stage'].value_counts()
        stages = stage_counts.index.tolist()
        counts = stage_counts.values.tolist()
        
        return jsonify({
            'success': True,
            'stages': stages,
            'counts': counts
        })
    
    except Exception as e:
        print(f"Lifecycle stats error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/lifecycle_info')
def lifecycle_model_info():
    """API endpoint for lifecycle model information"""
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'CustomerLifeCycle', 'model')
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'CustomerLifeCycle', 'lifecycle.csv')
    info = {
        'model_type': lifecycle_metadata.get('model_type', 'Unknown'),
        'features': lifecycle_features,
        'classes': lifecycle_metadata.get('classes', []),
        'model_loaded': lifecycle_model is not None,
        'needs_scaling': lifecycle_metadata.get('needs_scaling', False),
        'n_features': len(lifecycle_features),
        'files_exist': {
            'model': os.path.exists(os.path.join(MODEL_DIR, 'model.pkl')),
            'scaler': os.path.exists(os.path.join(MODEL_DIR, 'lifecycle_scaler.pkl')),
            'label_encoder': os.path.exists(os.path.join(MODEL_DIR, 'label_encoder.pkl')),
            'features': os.path.exists(os.path.join(MODEL_DIR, 'lifecycle_features.pkl')),
            'metadata': os.path.exists(os.path.join(MODEL_DIR, 'lifecycle_model_metadata.pkl')),
            'data': os.path.exists(DATA_PATH)
        }
    }
    return jsonify(info)

@app.route('/api/reload_lifecycle_model')
def reload_lifecycle_model():
    """Reload lifecycle model files"""
    success = load_lifecycle_model_files()
    return jsonify({
        'success': success,
        'model_loaded': lifecycle_model is not None,
        'message': 'Lifecycle model reloaded successfully' if success else 'Failed to reload lifecycle model'
    })

@app.route('/api/purchase_info')
def purchase_model_info():
    """API endpoint for purchase model information"""
    info = {
        'model_type': 'Random Forest Classifier',
        'features': purchase_features,
        'clusters': purchase_cluster_labels,
        'model_loaded': purchase_model is not None,
        'n_features': len(purchase_features),
        'files_exist': {
            'model': os.path.exists(os.path.join('purchase_clusters', 'model', 'random_forest_v3_model.joblib')),
            'scaler': os.path.exists(os.path.join('purchase_clusters', 'model', 'scaler_v3.joblib'))
        }
    }
    return jsonify(info)

@app.route('/api/reload_purchase_model')
def reload_purchase_model():
    """Reload purchase model files"""
    success = load_purchase_model_files()
    return jsonify({
        'success': success,
        'model_loaded': purchase_model is not None,
        'message': 'Purchase model reloaded successfully' if success else 'Failed to reload purchase model'
    })

@app.route('/api/spending_info')
def spending_model_info():
    """API endpoint for spending model information"""
    info = {
        'model_type': 'Random Forest Classifier',
        'features': spending_features,
        'clusters': spending_cluster_labels,
        'model_loaded': spending_model is not None,
        'n_features': len(spending_features),
        'files_exist': {
            'model': os.path.exists(os.path.join('spending_clusters', 'model', 'spending_rf_v3_model.joblib')),
            'scaler': os.path.exists(os.path.join('spending_clusters', 'model', 'spending_scaler_v3.joblib'))
        }
    }
    return jsonify(info)

@app.route('/api/reload_spending_model')
def reload_spending_model():
    """Reload spending model files"""
    success = load_spending_model_files()
    return jsonify({
        'success': success,
        'model_loaded': spending_model is not None,
        'message': 'Spending model reloaded successfully' if success else 'Failed to reload spending model'
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created templates directory")
    
    print("Starting Flask app...")
    print("Customer Segmentation API (Demographic, Purchase, Spending & Lifecycle)")
    print("Visit: http://localhost:5000")
    print(f"Demographic model loaded: {model_loaded}")
    print(f"Purchase model loaded: {purchase_model_loaded}")
    print(f"Spending model loaded: {spending_model_loaded}")
    print(f"Lifecycle model loaded: {lifecycle_model_loaded}")
    
    if not model_loaded:
        print("\nWARNING: Demographic model not loaded!")
        print("To fix this:")
        print("   1. Run your train_model.py script")
        print("   2. Make sure it creates the required files")
        print("   3. Restart this Flask app")
    
    if not purchase_model_loaded:
        print("\nWARNING: Purchase model not loaded!")
        print("To fix this:")
        print("   1. Run your purchase model training script")
        print("   2. Make sure it creates the required files")
        print("   3. Restart this Flask app")
    
    if not spending_model_loaded:
        print("\nWARNING: Spending model not loaded!")
        print("To fix this:")
        print("   1. Run your spending model training script")
        print("   2. Make sure it creates the required files")
        print("   3. Restart this Flask app")
    
    if not lifecycle_model_loaded:
        print("\nWARNING: Lifecycle model not loaded!")
        print("To fix this:")
        print("   1. Run your train_lifecycle_model.py script")
        print("   2. Make sure it creates the required files")
        print("   3. Restart this Flask app")
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except KeyboardInterrupt:
        print("\nFlask app stopped by user")
    except Exception as e:
        print(f"Error running Flask app: {e}")