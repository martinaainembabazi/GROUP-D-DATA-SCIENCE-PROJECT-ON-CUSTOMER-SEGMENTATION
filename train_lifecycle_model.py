import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json
import os

print("Training customer lifecycle stage prediction model...")

try:
    # Load the dataset
    df = pd.read_csv('featured_customer_segmentation.csv')
    print(f"Dataset loaded: {df.shape}")

    # Check if Lifecycle_Stage exists
    if 'Lifecycle_Group' not in df.columns:
        print("Lifecycle_Stage column not found! Assigning stages based on logic...")
        # Calculate thresholds
        ENGAGEMENT_THRESHOLD = df['Engagement_Ratio'].quantile(0.85)
        ACTIVITY_THRESHOLD = df['Activity_Index'].quantile(0.85)

        # Assign Lifecycle_Stage
        df['Lifecycle_Group'] = df.apply(
            lambda row: (
                "Churned" if row['Recency'] > 90
                else "At-Risk" if row['Recency'] > 60
                else "Super-Engaged" if (
                    row['Engagement_Ratio'] > ENGAGEMENT_THRESHOLD * 5 and
                    row['Activity_Index'] > ACTIVITY_THRESHOLD * 5
                )
                else "Loyal-High-Value" if (
                    row['Engagement_Ratio'] > ENGAGEMENT_THRESHOLD and
                    row['Activity_Index'] > ACTIVITY_THRESHOLD
                )
                else "Loyal-Regular" if row['Tenure_Days'] > 365
                else "New"
            ), axis=1
        )
        print("Lifecycle stages assigned")

    # Define lifecycle features
    lifecycle_features = [
        'Recency', 'Tenure_Days', 'Engagement_Ratio', 'Activity_Index', 'Total_Spending'
    ]

    # Filter available features
    available_features = [f for f in lifecycle_features if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")

    # Prepare data
    X = df[available_features].copy()
    y = df['Lifecycle_Stage'].copy()

    # Handle missing values
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            print(f"Filling {X[col].isnull().sum()} missing values in {col}")
            X[col] = X[col].fillna(X[col].median())

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train Random Forest
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")

    # Create scaler (for consistency)
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Save model files
    print("\nSaving model files...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'lifecycle_scaler.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')

    # Save feature names
    with open('lifecycle_features.txt', 'w') as f:
        for feature in available_features:
            f.write(f"{feature}\n")

    # Save metadata
    metadata = {
        'model_type': 'Random Forest',
        'features': available_features,
        'needs_scaling': False,
        'n_classes': len(label_encoder.classes_),
        'test_accuracy': test_score,
        'feature_count': len(available_features),
        'classes': list(label_encoder.classes_)
    }

    with open('lifecycle_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Model files saved successfully!")
    print("Files created:")
    print("  - model.joblib")
    print("  - lifecycle_scaler.joblib")
    print("  - label_encoder.joblib")
    print("  - lifecycle_features.txt")
    print("  - lifecycle_model_metadata.json")

    # Test the model
    print("\nTesting saved model...")
    loaded_model = joblib.load('model.joblib')
    loaded_label_encoder = joblib.load('label_encoder.joblib')
    test_prediction = loaded_model.predict(X_test.iloc[0:1])
    test_probabilities = loaded_model.predict_proba(X_test.iloc[0:1])
    print(f"Test prediction: {loaded_label_encoder.inverse_transform(test_prediction)[0]}")
    print(f"Test probabilities: {test_probabilities[0]}")

    print("\nModel training completed successfully!")
    print("You can now integrate with your Flask app!")

except FileNotFoundError:
    print("featured_customer_segmentation.csv not found!")
    print("Please make sure the dataset file exists in the current directory")
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()