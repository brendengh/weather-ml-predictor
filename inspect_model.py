#!/usr/bin/env python3
"""
Script to inspect the saved machine learning model
"""

import joblib
import pandas as pd

def inspect_model():
    """Load and inspect the saved model"""
    try:
        # Load the model
        model = joblib.load('models/best_weather_model.joblib')
        
        print("Model Information:")
        print("=" * 50)
        print(f"Model type: {type(model)}")
        print(f"Model: {model}")
        
        # If it's a scikit-learn model, show some common attributes
        if hasattr(model, 'get_params'):
            print(f"\nModel parameters:")
            for param, value in model.get_params().items():
                print(f"  {param}: {value}")
        
        if hasattr(model, 'feature_importances_'):
            print(f"\nFeature importances:")
            for i, importance in enumerate(model.feature_importances_):
                print(f"  Feature {i}: {importance:.4f}")
        
        if hasattr(model, 'score'):
            print(f"\nModel has a score method available")
            
        if hasattr(model, 'predict'):
            print(f"Model has a predict method available")
            
        if hasattr(model, 'predict_proba'):
            print(f"Model has a predict_proba method available")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    inspect_model()
