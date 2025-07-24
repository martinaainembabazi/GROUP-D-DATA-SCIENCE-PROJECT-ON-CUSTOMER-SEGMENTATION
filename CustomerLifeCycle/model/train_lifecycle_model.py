# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# import joblib
# import pickle
# import os

# # Define paths relative to the script location
# HERE = os.path.dirname(__file__)
# DATA_PATH = os.path.join(HERE, '..', 'lifecycle.csv')
# MODEL_DIR = os.path.join(HERE, 'model')

# print("Training customer lifecycle stage prediction model...")

# try:
#     # Create model directory if it doesn't exist
#     os.makedirs(MODEL_DIR, exist_ok=True)
    
#     # Load the dataset
#     df = pd.read_csv(DATA_PATH)
#     print("Lifecycle CSV Columns:", df.columns.tolist())
#     print(f"Dataset loaded: {df.shape}")

#     # Check if Lifecycle_Stage exists
#     if 'Lifecycle_Stage' not in df.columns:
#         print("Lifecycle_Stage column not found! Assigning stages based on logic...")
#         df['Lifecycle_Stage'] = df.apply(
#             lambda row: (
#                 "At-Risk" if row['Recency'] > 60 
#                 else "Loyal" if row['Tenure_Days'] > 365 
#                 else "Active"
#             ), axis=1
#         )
        
#     # Define lifecycle features
#     lifecycle_features = [
#         'Recency', 'Tenure_Days', 'Engagement_Ratio', 'Activity_Index', 'Total_Spending'
#     ]

#     # Filter available features
#     available_features = [f for f in lifecycle_features if f in df.columns]
#     print(f"Using {len(available_features)} features: {available_features}")

#     # Prepare data
#     X = df[available_features].copy()
#     y = df['Lifecycle_Stage'].copy()

#     # Handle missing values
#     for col in X.columns:
#         if X[col].isnull().sum() > 0:
#             print(f"Filling {X[col].isnull().sum()} missing values in {col}")
#             X[col] = X[col].fillna(X[col].median())

#     # Handle infinite values
#     print("Checking for infinite values in features:")
#     print(np.isinf(X).sum())
#     X = X.replace([np.inf, -np.inf], np.nan)
#     for col in X.columns:
#         if X[col].isnull().sum() > 0:  # Check again after replacing inf
#             print(f"Filling {X[col].isnull().sum()} missing values in {col} after inf replacement")
#             X[col] = X[col].fillna(X[col].median())

#     # Cap extremely large values to prevent float32 overflow
#     max_float32 = 3.4e38
#     X = X.clip(upper=max_float32, lower=-max_float32)

#     # Encode target
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#     )

#     print(f"Training set: {X_train.shape[0]} samples")
#     print(f"Test set: {X_test.shape[0]} samples")

#     # Train Random Forest
#     print("Training Random Forest model...")
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)

#     # Evaluate
#     train_score = model.score(X_train, y_train)
#     test_score = model.score(X_test, y_test)
#     print(f"Training accuracy: {train_score:.3f}")
#     print(f"Test accuracy: {test_score:.3f}")

#     # Create scaler (for consistency)
#     scaler = StandardScaler()
#     scaler.fit(X_train)

#     # Save model files
#     print("\nSaving model files...")
#     joblib.dump(model, os.path.join(MODEL_DIR, 'model.pkl'))
#     joblib.dump(scaler, os.path.join(MODEL_DIR, 'lifecycle_scaler.pkl'))
#     joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

#     # Save feature names
#     with open(os.path.join(MODEL_DIR, 'lifecycle_features.pkl'), 'wb') as f:
#         pickle.dump(available_features, f)

#     # Save metadata
#     metadata = {
#         'model_type': 'RandomForestClassifier',
#         'features': available_features,
#         'needs_scaling': True,
#         'n_classes': len(label_encoder.classes_),
#         'test_accuracy': test_score,
#         'feature_count': len(available_features),
#         'classes': list(label_encoder.classes_)
#     }
#     with open(os.path.join(MODEL_DIR, 'lifecycle_model_metadata.pkl'), 'wb') as f:
#         pickle.dump(metadata, f)

#     print("\nModel files saved successfully in:", MODEL_DIR)
#     print("Files created:")
#     print("  - model.pkl")
#     print("  - lifecycle_scaler.pkl")
#     print("  - label_encoder.pkl")
#     print("  - lifecycle_features.pkl")
#     print("  - lifecycle_model_metadata.pkl")

#     # Test the model
#     print("\nTesting saved model...")
#     loaded_model = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
#     loaded_label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
#     test_prediction = loaded_model.predict(X_test.iloc[0:1])
#     test_probabilities = loaded_model.predict_proba(X_test.iloc[0:1])
#     print(f"Test prediction: {loaded_label_encoder.inverse_transform(test_prediction)[0]}")
#     print(f"Test probabilities: {test_probabilities[0]}")

#     print("\nModel training completed successfully!")
#     print("You can now integrate with your Flask app!")

# except FileNotFoundError:
#     print(f"Error: {DATA_PATH} not found! Please ensure lifecycle.csv is in CustomerLifeCycle/")
# except Exception as e:
#     print(f"Error during training: {e}")
#     import traceback
#     traceback.print_exc()


import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define paths
LIFECYCLE_MODEL_DIR = os.path.join('CustomerLifeCycle', 'Model', 'model')
LIFECYCLE_DATA_PATH = os.path.join('CustomerLifeCycle', 'lifecycle.csv')

# Ensure model directory exists
os.makedirs(LIFECYCLE_MODEL_DIR, exist_ok=True)

# Load data
print("Loading lifecycle.csv...")
df = pd.read_csv(LIFECYCLE_DATA_PATH)

# Define features (based on your data; adjust as needed)
features = [
    'Age', 'Income', 'Total_Dependents', 'Is_Parent', 'Kidhome', 'Teenhome',
    'Total_Spending', 'Recency', 'Total_Purchases', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
    'Total_Accepted_Cmp', 'Engagement_Ratio', 'Activity_Index',
    'Recency_Score', 'Tenure_Score'
]
X = df[features]
y = df['Lifecycle_Stage']  # Assuming this is the target

# Encode target variable
print("Encoding target variable...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y_encoded)

# Metadata (e.g., model type, scaling requirement)
metadata = {
    'model_type': 'RandomForestClassifier',
    'needs_scaling': True,
    'features': features
}

# Save model and related files
print("Saving model and related files...")
joblib.dump(model, os.path.join(LIFECYCLE_MODEL_DIR, 'model.pkl'))
joblib.dump(scaler, os.path.join(LIFECYCLE_MODEL_DIR, 'lifecycle_scaler.pkl'))
joblib.dump(label_encoder, os.path.join(LIFECYCLE_MODEL_DIR, 'label_encoder.pkl'))
joblib.dump(features, os.path.join(LIFECYCLE_MODEL_DIR, 'lifecycle_features.pkl'))
joblib.dump(metadata, os.path.join(LIFECYCLE_MODEL_DIR, 'lifecycle_model_metadata.pkl'))

print("Model and files saved successfully!")