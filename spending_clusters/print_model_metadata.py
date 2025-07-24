import joblib
import os
import json
import numpy as np
import pandas as pd

def print_model_metadata():
    # Load the trained model
    try:
        model_path = 'model/spending_rf_v3_model.joblib'
        model = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")

        # Extract model information
        metadata = {
            "model_type": str(type(model)),
            "parameters": model.get_params(),
            "feature_importances": None,
            "n_features": None,
            "creation_date": "2025-07-21"  # Current date
        }

        # If it's a RandomForestClassifier, extract feature importances
        if hasattr(model, 'feature_importances_'):
            metadata["feature_importances"] = model.feature_importances_.tolist()
            metadata["n_features"] = len(model.feature_importances_)

            # Print feature importances in a readable format
            print("\nFeature Importances:")
            for i, importance in enumerate(model.feature_importances_):
                print(f"Feature {i}: {importance:.4f}")

        # Print basic model info
        print(f"\nModel Type: {metadata['model_type']}")
        print(f"Number of Features: {metadata['n_features']}")
        print("\nModel Parameters:")
        for param, value in metadata['parameters'].items():
            print(f"  {param}: {value}")

        # Save metadata to JSON file
        output_path = 'model/spending_model_metadata.json'
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=lambda x: str(x))
        print(f"\nMetadata saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Model file not found. Please check if the model has been trained.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if __name__ == '__main__':
    # Make sure model directory exists
    os.makedirs('model', exist_ok=True)
    print_model_metadata()
