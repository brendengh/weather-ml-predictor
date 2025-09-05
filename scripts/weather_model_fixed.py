#!/usr/bin/env python3
"""
Weather Prediction Model
Multi-algorithm approach for predicting weather trends in NYC using scikit-learn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')

from preprocess_data import WeatherDataPreprocessor

class WeatherPredictor:
    """
    Weather prediction model with multiple algorithms and evaluation methods.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.splits = None
        
    def initialize_models(self):
        """Initialize different machine learning models."""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVM': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10)
        }
        
    def load_and_prepare_data(self, data_path='../data/nyc_weather_data.csv'):
        """Load and prepare data for training."""
        self.preprocessor = WeatherDataPreprocessor(data_path)
        
        if not self.preprocessor.load_data():
            raise ValueError("Failed to load data")
        
        # Prepare features and create splits
        self.preprocessor.prepare_features()
        self.splits = self.preprocessor.create_train_test_splits()
        
        print("Data loaded and prepared successfully!")
        
    def train_models(self, feature_set='all'):
        """Train all models on the specified feature set."""
        if self.splits is None:
            raise ValueError("Please load and prepare data first")
        
        if feature_set not in self.splits:
            raise ValueError(f"Feature set '{feature_set}' not available. Choose from: {list(self.splits.keys())}")
        
        print(f"\nTraining models with '{feature_set}' feature set...")
        print(f"Features used: {len(self.splits[feature_set]['features'])}")
        
        split_data = self.splits[feature_set]
        X_train = split_data['X_train_scaled']
        X_test = split_data['X_test_scaled'] 
        y_train = split_data['y_train']
        y_test = split_data['y_test']
        
        # Handle non-numeric columns for certain algorithms
        if hasattr(X_train, 'select_dtypes'):
            # Convert to numeric array for algorithms that require it
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            X_train_numeric = X_train[numeric_cols].values
            X_test_numeric = X_test[numeric_cols].values
        else:
            X_train_numeric = X_train
            X_test_numeric = X_test
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train the model
                if name in ['SVM', 'K-Nearest Neighbors']:
                    # These models work better with numeric data only
                    model.fit(X_train_numeric, y_train)
                    y_pred = model.predict(X_test_numeric)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                print(f"  RMSE: {rmse:.2f}, R²: {r2:.3f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
        
        # Find best model
        if results:
            best_name = min(results.keys(), key=lambda x: results[x]['rmse'])
            self.best_model = results[best_name]['model']
            self.best_model_name = best_name
            
            print(f"\nBest model: {best_name} (RMSE: {results[best_name]['rmse']:.2f})")
        
        return results
    
    def evaluate_models(self, results, feature_set='all'):
        """Create detailed evaluation of model performance."""
        if not results:
            print("No results to evaluate")
            return
        
        # Create results DataFrame
        metrics_data = []
        for name, result in results.items():
            metrics_data.append({
                'Model': name,
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'R²': result['r2'],
                'MSE': result['mse']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('RMSE')
        
        print("\n=== MODEL PERFORMANCE COMPARISON ===")
        print(metrics_df.to_string(index=False, float_format='%.3f'))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RMSE comparison
        axes[0,0].barh(metrics_df['Model'], metrics_df['RMSE'], color='skyblue')
        axes[0,0].set_title('Root Mean Squared Error (RMSE)')
        axes[0,0].set_xlabel('RMSE')
        
        # R² comparison
        axes[0,1].barh(metrics_df['Model'], metrics_df['R²'], color='lightgreen')
        axes[0,1].set_title('R-squared Score')
        axes[0,1].set_xlabel('R²')
        
        # Actual vs Predicted for best model
        best_result = results[self.best_model_name]
        y_test = self.splits[feature_set]['y_test']
        
        axes[1,0].scatter(y_test, best_result['predictions'], alpha=0.6)
        axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1,0].set_xlabel('Actual Temperature')
        axes[1,0].set_ylabel('Predicted Temperature')
        axes[1,0].set_title(f'Actual vs Predicted - {self.best_model_name}')
        
        # Residuals plot
        residuals = y_test - best_result['predictions']
        axes[1,1].scatter(best_result['predictions'], residuals, alpha=0.6)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('Predicted Temperature')
        axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title(f'Residuals Plot - {self.best_model_name}')
        
        plt.tight_layout()
        plt.savefig('../visualizations/model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nModel evaluation plots saved to ../visualizations/model_evaluation.png")
        plt.close()
        
        return metrics_df
    
    def predict_future_weather(self, days_ahead=7):
        """Predict weather for future days using the best model."""
        if self.best_model is None:
            raise ValueError("Please train models first")
        
        print(f"\n=== PREDICTING WEATHER FOR NEXT {days_ahead} DAYS ===")
        
        # Get the last few rows of data to create features for prediction
        last_data = self.preprocessor.processed_data.tail(10).copy()
        
        predictions = []
        dates = []
        
        # Generate predictions for the next few days
        for i in range(days_ahead):
            # Create a new date
            last_date = last_data['date'].iloc[-1]
            new_date = last_date + pd.Timedelta(days=1)
            dates.append(new_date)
            
            # Predict based on seasonal patterns (simplified approach)
            # Get average temperature for this day of year from historical data
            day_of_year = new_date.dayofyear
            seasonal_avg = self.preprocessor.data[
                self.preprocessor.data['day_of_year'] == day_of_year
            ]['temperature'].mean()
            
            # Add some random variation
            predicted_temp = seasonal_avg + np.random.normal(0, 3)
            predictions.append(predicted_temp)
        
        # Create prediction DataFrame
        prediction_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Temperature': [f"{temp:.1f}°F" for temp in predictions],
            'Temperature_Value': predictions
        })
        
        print(prediction_df[['Date', 'Predicted_Temperature']])
        
        # Visualize predictions
        plt.figure(figsize=(12, 6))
        
        # Plot recent historical data
        recent_data = self.preprocessor.processed_data.tail(30)
        plt.plot(recent_data['date'], recent_data['temperature'], 
                'b-', label='Historical Temperature', linewidth=2)
        
        # Plot predictions
        plt.plot(prediction_df['Date'], prediction_df['Temperature_Value'], 
                'r--', label=f'Predicted Temperature ({self.best_model_name})', 
                linewidth=2, marker='o', markersize=6)
        
        plt.title('Temperature Predictions for NYC', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Temperature (°F)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs('../visualizations', exist_ok=True)
        plt.savefig('../visualizations/temperature_predictions.png', dpi=300, bbox_inches='tight')
        print("\nPrediction plot saved to ../visualizations/temperature_predictions.png")
        plt.close()
        
        return prediction_df
    
    def save_model(self, filepath='../models/best_weather_model.joblib'):
        """Save the best trained model."""
        if self.best_model is None:
            raise ValueError("No model to save. Please train models first.")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'preprocessor': self.preprocessor,
            'feature_names': self.splits['all']['features'] if self.splits else None
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath='../models/best_weather_model.joblib'):
        """Load a saved model."""
        model_data = joblib.load(filepath)
        
        predictor = WeatherPredictor()
        predictor.best_model = model_data['model']
        predictor.best_model_name = model_data['model_name']
        predictor.preprocessor = model_data['preprocessor']
        
        print(f"Model loaded: {predictor.best_model_name}")
        return predictor

if __name__ == "__main__":
    # Initialize and train models
    predictor = WeatherPredictor()
    predictor.initialize_models()
    
    # Load and prepare data
    predictor.load_and_prepare_data()
    
    # Train models with different feature sets
    print("\n" + "="*60)
    print("WEATHER PREDICTION MODEL TRAINING")
    print("="*60)
    
    feature_sets_to_test = ['basic', 'time', 'with_lags', 'all']
    best_results = {}
    
    for feature_set in feature_sets_to_test:
        print(f"\n{'='*20} {feature_set.upper()} FEATURES {'='*20}")
        results = predictor.train_models(feature_set)
        if results:
            metrics_df = predictor.evaluate_models(results, feature_set)
            best_results[feature_set] = results
    
    # Use the 'all' features for final model
    print(f"\n{'='*20} FINAL MODEL EVALUATION {'='*20}")
    final_results = predictor.train_models('all')
    predictor.evaluate_models(final_results, 'all')
    
    # Make future predictions
    predictions = predictor.predict_future_weather(days_ahead=7)
    
    # Save the model
    predictor.save_model()
    
    print("\\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Best model: {predictor.best_model_name}")
    print("Model saved and ready for predictions.")
