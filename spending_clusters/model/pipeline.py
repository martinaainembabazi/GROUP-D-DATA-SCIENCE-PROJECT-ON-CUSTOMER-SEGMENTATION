import numpy as np
import pandas as pd
import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib

# ---- 1. Raw columns ----
raw_features = ['Year_Birth', 'Kidhome', 'Teenhome', 'Marital_Status', 'Education', 'Income']

# ---- 2. Custom transformation functions ----
def compute_age(X):
    current_year = datetime.datetime.now().year
    return current_year - X

def compute_dependents(X):
    if isinstance(X, pd.DataFrame):
        return (pd.to_numeric(X['Kidhome'], errors='coerce').values.reshape(-1, 1) +
                pd.to_numeric(X['Teenhome'], errors='coerce').values.reshape(-1, 1))
    # Ensure we convert to float to handle strings or other types
    return (pd.to_numeric(X[:, 0], errors='coerce').reshape(-1, 1) +
            pd.to_numeric(X[:, 1], errors='coerce').reshape(-1, 1))

# ---- 3. ColumnTransformer ----
preprocessor = ColumnTransformer(
    transformers=[
        # Log-transform Income
        ('log_income', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(np.log1p, validate=False)),
            ('scaler', StandardScaler())
        ]), ['Income']),

        # Age = current_year - Year_Birth
        ('age', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('age_calc', FunctionTransformer(compute_age, validate=False)),
            ('scaler', StandardScaler())
        ]), ['Year_Birth']),

        # Education is already ordinal encoded - just pass through with scaling
        ('education', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['Education']),

        # One-hot encode Marital_Status - place this first to handle categorical data properly
        # One-hot encode Marital_Status
        ('marital',
        OneHotEncoder(categories=[['Single', 'Together', 'Married', 'Divorced', 'Widow']], handle_unknown='ignore'),
         ['Marital_Status']),

        # Combine dependents
        ('dependents', Pipeline([
            ('combiner', FunctionTransformer(compute_dependents, validate=False)),
            ('scaler', StandardScaler())
        ]), ['Kidhome', 'Teenhome']),
    ],
    remainder='drop',  # Essential: drop any columns not explicitly transformed
    sparse_threshold=0  # Force dense output instead of sparse
)

# ---- 4. Full Pipeline with RandomForest ----
full_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42
    ))
])

# ---- 5. Helper Functions ----
def fit_pipeline_from_dict(data_dict, target):
    """
    Fit the pipeline using a dictionary of features

    Parameters:
    data_dict (dict): Dictionary with keys as feature names and values as lists
    target (list/array): Target cluster labels

    Returns:
    Pipeline: Fitted pipeline
    """
    df = pd.DataFrame(data_dict)
    X_raw = df[raw_features]
    y = np.array(target)
    full_pipeline.fit(X_raw, y)
    return full_pipeline

def predict_from_dict(pipeline, data_dict):
    """
    Make predictions using a fitted pipeline

    Parameters:
    pipeline (Pipeline): Fitted pipeline
    data_dict (dict): Dictionary with keys as feature names and values as lists

    Returns:
    array: Predictions
    """
    df = pd.DataFrame(data_dict)
    X_raw = df[raw_features]
    return pipeline.predict(X_raw)

def predict_with_confidence(pipeline, data_dict):
    """
    Make predictions with confidence scores

    Parameters:
    pipeline (Pipeline): Fitted pipeline
    data_dict (dict): Dictionary with keys as feature names and values as lists

    Returns:
    tuple: (predictions, confidence_scores)
    """
    try:
        df = pd.DataFrame(data_dict)
        X_raw = df[raw_features]

        # First get predictions using the full pipeline (which includes preprocessing)
        preds = pipeline.predict(X_raw)

        # For confidence scores, use the full pipeline correctly
        if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
            # Process data through preprocessing first
            X_processed = pipeline.named_steps['preprocessing'].transform(X_raw)

            # Convert to numpy array to match training format and avoid the feature names warning
            if hasattr(X_processed, 'values'):
                X_processed = X_processed.values

            # Then get probabilities from the classifier
            confs = pipeline.named_steps['classifier'].predict_proba(X_processed).max(axis=1)
        else:
            confs = [None] * len(preds)

        return preds, confs
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(f"Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None

# ---- 6. Main (Train + Save + Predict) ----
def main():
    """Test the pipeline with sample data and save the model"""
    # Example data dictionary with Education already ordinal encoded
    data_dict = {
        'Year_Birth': [1980, 2000],
        'Kidhome': [2, 0],
        'Teenhome': [2, 2],
        'Marital_Status': ['Married', 'Single'],
        'Education': [0, 2],  # Already ordinal encoded (0=Basic, 2=Graduation, etc.)
        'Income': [100000, 30000]
    }
    target = [0, 1]  # Example spending cluster labels

    pipeline = fit_pipeline_from_dict(data_dict, target)
    preds, confs = predict_with_confidence(pipeline, data_dict)

    print("Pipeline Testing Results:")
    for i, (pred, conf) in enumerate(zip(preds, confs)):
        print(f'Sample {i}: Prediction={pred}, Confidence={conf:.3f}')

    # Save the trained pipeline
    joblib.dump(pipeline, 'model/spending_rf_v3_model.joblib')
    print("\nModel saved to: model/spending_rf_v3_model.joblib")

# ---- 7. Entry Point ----
if __name__ == '__main__':
    import os
    os.makedirs("model", exist_ok=True)
    main()

# ---- 8. Use Pretrained Model ----
def load_pretrained_model(model_path='spending_rf_v3_model.joblib'):
    """
    Load a pretrained model from the specified path

    Parameters:
    model_path (str): Path to the saved model file

    Returns:
    Pipeline: Loaded pipeline model
    """
    try:
        pipeline = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        return pipeline
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_from_pretrained(data_dict, model_path='spending_rf_v3_model.joblib'):
    """
    Make predictions using the pretrained model

    Parameters:
    data_dict (dict): Dictionary with keys as feature names and values as lists
    model_path (str): Path to the saved model file

    Returns:
    tuple: (predictions, confidence_scores)
    """
    # Load the pretrained model
    pipeline = load_pretrained_model(model_path)

    if pipeline is None:
        return None, None

    # Make predictions with confidence
    return predict_with_confidence(pipeline, data_dict)

# Example usage of pretrained model
def demo_pretrained_model():
    """
    Demonstrate using the pretrained model for predictions
    """
    # Test data
    test_data = {
        'Year_Birth': [1975, 1985, 1995],
        'Kidhome': [1, 2, 0],
        'Teenhome': [1, 0, 1],
        'Marital_Status': ['Divorced', 'Married', 'Single'],
        'Education': [2, 3, 1],  # Ordinal encoded values
        'Income': [75000, 120000, 45000]
    }

    # Get predictions using pretrained model
    predictions, confidences = predict_from_pretrained(test_data)

    if predictions is not None:
        print("\nPredictions using pretrained model:")
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            print(f"Sample {i}: Cluster={pred}, Confidence={conf:.3f}")

# Run the demo if this file is executed directly
if __name__ == '__main__':
    import os
    os.makedirs("model", exist_ok=True)

    # Check if model file exists
    model_file = 'model/spending_rf_v3_model.joblib'
    if os.path.exists(model_file):
        print(f"Found existing model: {model_file}")
        demo_pretrained_model()
    else:
        print(f"No existing model found. Training new model...")
        main()
        print("\nNow using the newly trained model:")
        demo_pretrained_model()
