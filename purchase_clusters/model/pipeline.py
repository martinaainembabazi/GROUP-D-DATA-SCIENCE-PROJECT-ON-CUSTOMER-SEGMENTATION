import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import joblib
import os

# ---- 1. Raw columns ----
raw_features = ['Year_Birth', 'Kidhome', 'Teenhome', 'Marital_Status', 'Education', 'Income']

# ---- 2. Define transformer functions ----
def compute_age(X):
    current_year = datetime.datetime.now().year
    return current_year - X

def compute_dependents(X):
    if isinstance(X, pd.DataFrame):
        return (X['Kidhome'].values.reshape(-1, 1) + X['Teenhome'].values.reshape(-1, 1))
    return X[:, 0].reshape(-1, 1) + X[:, 1].reshape(-1, 1)

# ---- 3. Feature pipeline steps ----
preprocessor = ColumnTransformer(transformers=[
    # Log-transform Income
    ('log_income', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ]), ['Income']),

    ('age', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('age_calc', FunctionTransformer(compute_age, validate=False)),
        ('scaler', StandardScaler())
    ]), ['Year_Birth']),


    # One-hot encode Marital_Status
    ('marital', OneHotEncoder(categories=[['Single', 'Together', 'Married', 'Divorced', 'Widow']], handle_unknown='ignore'), ['Marital_Status']),

    # Combine dependents
    ('dependents', Pipeline([
        ('combiner', FunctionTransformer(compute_dependents, validate=False)),
        ('scaler', StandardScaler())
    ]), ['Kidhome', 'Teenhome']),
])

# ---- 4. Define full pipeline with your tuned Random Forest ----
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

# ---- 5. Fit on full dataset (raw input) ----
# Load the dataset
input_path = os.path.join(os.path.dirname(__file__), '../../featured_customer_segmentation_with_clusters.csv')
df = pd.read_csv(input_path)

print(df.head())

X_raw = df[raw_features]
y = df['PurchaseCluster']
full_pipeline.fit(X_raw, y)

# ---- 6. Save the pipeline ----
output_path = os.path.join(os.path.dirname(__file__), 'random_forest_pipeline_v3.joblib')
joblib.dump(full_pipeline, output_path)
print(f"Pipeline saved to {output_path}")

def predict_from_dict(input_dict, model_path=None):
    """
    Predict purchase cluster from a dictionary of features using the trained pipeline.
    Args:
        input_dict (dict): Dictionary with keys matching raw_features.
        model_path (str): Path to the trained pipeline. If None, uses default path.
    Returns:
        prediction (np.ndarray): Predicted cluster(s).
    """
    input_df = pd.DataFrame(input_dict)
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'random_forest_pipeline_v3.joblib')
    pipeline = joblib.load(model_path)
    return pipeline.predict(input_df)

if __name__ == "__main__":
    # Train and save pipeline
    full_pipeline.fit(X_raw, y)
    joblib.dump(full_pipeline, output_path)
    print(f"Pipeline saved to {output_path}")

    # Example usage: Predict from dictionary
    example_dict = {
        'Year_Birth': [1990],
        'Kidhome': [1],
        'Teenhome': [0],
        'Marital_Status': ['Married'],
        'Education': [0],
        'Income': [30000]
    }
    prediction = predict_from_dict(example_dict)
    print(example_dict)
    print(f"Predicted cluster: {prediction[0]}")