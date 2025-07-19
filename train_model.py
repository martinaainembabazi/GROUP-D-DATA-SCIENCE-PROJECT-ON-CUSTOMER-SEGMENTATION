# Quick model training script to ensure model files are created
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os

print(" Training demographic segmentation model...")

try:
    # Load the dataset
    df = pd.read_csv('featured_customer_segmentation.csv')
    print(f" Dataset loaded: {df.shape}")
    
    # Check if Demographics_Cluster exists
    if 'Demographics_Cluster' not in df.columns:
        print(" Demographics_Cluster column not found!")
        print(" Creating clusters using KMeans...")
        
        from sklearn.cluster import KMeans
        
        # Use basic demographic features for clustering
        demographic_features = ['Age', 'Education', 'Total_Dependents', 'Is_Parent']
        X_demo = df[demographic_features].copy()
        
        # Handle missing values
        for col in X_demo.columns:
            if X_demo[col].isnull().sum() > 0:
                X_demo[col] = X_demo[col].fillna(X_demo[col].median())
        
        # Scale and cluster
        scaler_demo = StandardScaler()
        X_demo_scaled = scaler_demo.fit_transform(X_demo)
        
        # Use 4 clusters
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
        df['Demographics_Cluster'] = kmeans.fit_predict(X_demo_scaled)
        print("Clusters created")
    
    # Create marital status dummy variables if they don't exist
    if 'Marital_Status' in df.columns and 'Marital_Single' not in df.columns:
        print("🔧 Creating marital status dummy variables...")
        marital_dummies = pd.get_dummies(df['Marital_Status'], prefix='Marital')
        df = pd.concat([df, marital_dummies], axis=1)
        print(f" Created: {list(marital_dummies.columns)}")
    
    # Define prediction features
    prediction_features = [
        'Age', 'Education', 'Income', 'Total_Dependents', 'Is_Parent',
        'Kidhome', 'Teenhome', 'Marital_Divorced', 'Marital_Married',
        'Marital_Single', 'Marital_Together', 'Marital_Widow'
    ]
    
    # Filter available features
    available_features = [f for f in prediction_features if f in df.columns]
    print(f" Using {len(available_features)} features: {available_features}")
    
    # Prepare data
    X = df[available_features].copy()
    y = df['Demographics_Cluster'].copy()
    
    # Handle missing values
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            print(f"🔧 Filling {X[col].isnull().sum()} missing values in {col}")
            X[col] = X[col].fillna(X[col].median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f" Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest (doesn't need scaling)
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f" Training accuracy: {train_score:.3f}")
    print(f" Test accuracy: {test_score:.3f}")
    
    # Create scaler (even though Random Forest doesn't need it, for consistency)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Save model files
    print("💾 Saving model files...")
    
    # Save model and scaler
    joblib.dump(model, 'demographic_cluster_model.pkl')
    joblib.dump(scaler, 'demographic_scaler.pkl')
    
    # Save feature names
    with open('model_features.txt', 'w') as f:
        for feature in available_features:
            f.write(f"{feature}\n")
    
    # Save metadata
    metadata = {
        'model_type': 'Random Forest',
        'features': available_features,
        'needs_scaling': False,  # Random Forest doesn't need scaling
        'n_clusters': len(y.unique()),
        'test_accuracy': test_score,
        'feature_count': len(available_features)
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(" Model files saved successfully!")
    print(" Files created:")
    print("   - demographic_cluster_model.pkl")
    print("   - demographic_scaler.pkl") 
    print("   - model_features.txt")
    print("   - model_metadata.json")
    
    # Test the model
    print("\nTesting saved model...")
    loaded_model = joblib.load('demographic_cluster_model.pkl')
    test_prediction = loaded_model.predict(X_test.iloc[0:1])
    test_probabilities = loaded_model.predict_proba(X_test.iloc[0:1])
    print(f"Test prediction: {test_prediction[0]}")
    print(f"Test probabilities: {test_probabilities[0]}")
    
    print("\n Model training completed successfully!")
    print(" You can now run your Flask app!")
    
except FileNotFoundError:
    print(" featured_customer_segmentation.csv not found!")
    print(" Please make sure the dataset file exists in the current directory")
except Exception as e:
    print(f" Error during training: {e}")
    import traceback
    traceback.print_exc()