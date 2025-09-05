#!/usr/bin/env python3
"""
NYC Weather Prediction Tool
User-friendly script for making weather predictions using trained ML models.
"""

import sys
import os
sys.path.append('scripts')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

try:
    from scripts.weather_model_fixed import WeatherPredictor
    from scripts.preprocess_data import WeatherDataPreprocessor
except ImportError:
    print("Error: Could not import required modules. Please run from the project directory.")
    sys.exit(1)

class WeatherPredictionTool:
    """
    Main interface for weather prediction functionality.
    """
    
    def __init__(self):
        self.predictor = None
        
    def load_model(self, model_path='models/best_weather_model.joblib'):
        """Load a trained model."""
        try:
            self.predictor = WeatherPredictor.load_model(model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def train_new_model(self):
        """Train a new model with current data."""
        print("Training new weather prediction model...")
        print("This may take a few minutes...")
        
        try:
            self.predictor = WeatherPredictor()
            self.predictor.initialize_models()
            self.predictor.load_and_prepare_data('data/nyc_weather_data.csv')
            
            # Train with the best performing feature set
            results = self.predictor.train_models('with_lags')  # This performed best
            
            if results:
                self.predictor.save_model()
                print("\nModel training completed successfully!")
                return True
            else:
                print("Error: Model training failed")
                return False
                
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_temperature(self, days_ahead=7):
        """Make temperature predictions."""
        if self.predictor is None:
            print("Please load or train a model first")
            return None
        
        try:
            predictions = self.predictor.predict_future_weather(days_ahead)
            return predictions
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def show_model_info(self):
        """Display information about the current model."""
        if self.predictor is None:
            print("No model loaded")
            return
        
        print("\n=== MODEL INFORMATION ===")
        print(f"Model type: {self.predictor.best_model_name}")
        
        if hasattr(self.predictor, 'preprocessor') and self.predictor.preprocessor:
            data_shape = self.predictor.preprocessor.data.shape if self.predictor.preprocessor.data is not None else "Unknown"
            print(f"Training data size: {data_shape}")
            
            if self.predictor.preprocessor.data is not None:
                date_range = f"{self.predictor.preprocessor.data['date'].min()} to {self.predictor.preprocessor.data['date'].max()}"
                print(f"Data date range: {date_range}")
    
    def show_recent_weather(self, days=10):
        """Display recent weather data."""
        if self.predictor is None or self.predictor.preprocessor is None:
            print("No data available. Please load a model first.")
            return
        
        try:
            recent_data = self.predictor.preprocessor.data.tail(days)[
                ['date', 'temperature', 'humidity', 'weather_condition', 'precipitation']
            ]
            
            print(f"\n=== RECENT WEATHER (Last {days} days) ===")
            for _, row in recent_data.iterrows():
                date = row['date'].strftime('%Y-%m-%d')
                temp = row['temperature']
                humidity = row['humidity']
                condition = row['weather_condition']
                precip = row['precipitation']
                
                print(f"{date}: {temp:.1f}°F, {humidity:.0f}% humidity, {condition}")
                if precip > 0:
                    print(f"           Precipitation: {precip:.2f} inches")
                    
        except Exception as e:
            print(f"Error displaying recent weather: {e}")
    
    def interactive_mode(self):
        """Run the tool in interactive mode."""
        print("\n" + "="*60)
        print("NYC WEATHER PREDICTION TOOL")
        print("="*60)
        print("Using machine learning to predict weather trends in New York City")
        print()
        
        # Try to load existing model first
        if os.path.exists('models/best_weather_model.joblib'):
            print("Loading trained model...")
            if self.load_model():
                print("Model loaded successfully!")
                self.show_model_info()
            else:
                print("Failed to load model. Will train a new one.")
                if not self.train_new_model():
                    print("Failed to train model. Exiting.")
                    return
        else:
            print("No trained model found. Training new model...")
            if not self.train_new_model():
                print("Failed to train model. Exiting.")
                return
        
        while True:
            print("\n" + "-"*40)
            print("MENU:")
            print("1. Make weather predictions")
            print("2. Show recent weather")
            print("3. Show model information")
            print("4. Train new model")
            print("5. Exit")
            
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    try:
                        days = int(input("How many days ahead to predict (1-14)? "))
                        if 1 <= days <= 14:
                            predictions = self.predict_temperature(days)
                            if predictions is not None:
                                print(f"\n=== WEATHER PREDICTIONS FOR NEXT {days} DAYS ===")
                                for _, row in predictions.iterrows():
                                    date = row['Date'].strftime('%Y-%m-%d')
                                    temp = row['Predicted_Temperature']
                                    print(f"{date}: {temp}")
                        else:
                            print("Please enter a number between 1 and 14")
                    except ValueError:
                        print("Please enter a valid number")
                
                elif choice == '2':
                    try:
                        days = int(input("How many recent days to show (1-30)? "))
                        if 1 <= days <= 30:
                            self.show_recent_weather(days)
                        else:
                            print("Please enter a number between 1 and 30")
                    except ValueError:
                        print("Please enter a valid number")
                
                elif choice == '3':
                    self.show_model_info()
                
                elif choice == '4':
                    confirm = input("Train new model? This will overwrite the current model (y/n): ")
                    if confirm.lower() == 'y':
                        self.train_new_model()
                
                elif choice == '5':
                    print("Thank you for using the NYC Weather Prediction Tool!")
                    break
                
                else:
                    print("Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(
        description="NYC Weather Prediction Tool - Predict weather using machine learning"
    )
    parser.add_argument(
        '--days', '-d', type=int, default=7,
        help='Number of days to predict (default: 7)'
    )
    parser.add_argument(
        '--recent', '-r', type=int,
        help='Show recent weather for specified number of days'
    )
    parser.add_argument(
        '--train', '-t', action='store_true',
        help='Train a new model'
    )
    parser.add_argument(
        '--info', '-i', action='store_true',
        help='Show model information'
    )
    parser.add_argument(
        '--interactive', action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    tool = WeatherPredictionTool()
    
    # Interactive mode
    if args.interactive or len(sys.argv) == 1:
        tool.interactive_mode()
        return
    
    # Load model for other operations
    if not tool.load_model():
        if not args.train:
            print("No model found. Use --train to create one or --interactive for guided setup.")
            return
    
    # Command line operations
    if args.train:
        tool.train_new_model()
    
    if args.info:
        tool.show_model_info()
    
    if args.recent:
        tool.show_recent_weather(args.recent)
    
    if not args.train and not args.info and not args.recent:
        # Default: make predictions
        predictions = tool.predict_temperature(args.days)
        if predictions is not None:
            print(f"\nWeather predictions for the next {args.days} days:")
            for _, row in predictions.iterrows():
                date = row['Date'].strftime('%Y-%m-%d')
                temp = row['Predicted_Temperature']
                print(f"{date}: {temp}")

if __name__ == "__main__":
    main()
