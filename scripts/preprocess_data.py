#!/usr/bin/env python3
"""
Weather Data Preprocessing Script
Prepares weather data for machine learning analysis and model training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class WeatherDataPreprocessor:
    """
    Class to handle weather data preprocessing and exploration.
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load weather data from CSV file."""
        try:
            self.data = pd.read_csv(self.data_path, parse_dates=['date'])
            print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        if self.data is None:
            print("Please load data first using load_data()")
            return
        
        print("=== DATA OVERVIEW ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"Missing values: {self.data.isnull().sum().sum()}\n")
        
        print("=== BASIC STATISTICS ===")
        numerical_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation', 'cloud_cover']
        print(self.data[numerical_cols].describe().round(2))
        
        print("\n=== CATEGORICAL FEATURES ===")
        categorical_cols = ['weather_condition', 'season', 'temp_category']
        for col in categorical_cols:
            print(f"\n{col.upper()}:")
            print(self.data[col].value_counts().head())
    
    def create_visualizations(self, save_path='../visualizations/'):
        """Create exploratory data analysis visualizations."""
        if self.data is None:
            print("Please load data first using load_data()")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Temperature trends over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Temperature over time
        axes[0,0].plot(self.data['date'], self.data['temperature'], alpha=0.7, linewidth=0.5)
        axes[0,0].set_title('Temperature Trends Over Time')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Temperature (°F)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Seasonal temperature distribution
        sns.boxplot(data=self.data, x='season', y='temperature', ax=axes[0,1])
        axes[0,1].set_title('Temperature Distribution by Season')
        axes[0,1].set_ylabel('Temperature (°F)')
        
        # Weather conditions distribution
        weather_counts = self.data['weather_condition'].value_counts()
        axes[1,0].pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Weather Conditions Distribution')
        
        # Temperature vs Humidity relationship
        scatter = axes[1,1].scatter(self.data['temperature'], self.data['humidity'], 
                                   c=self.data['precipitation'], alpha=0.6, cmap='Blues')
        axes[1,1].set_xlabel('Temperature (°F)')
        axes[1,1].set_ylabel('Humidity (%)')
        axes[1,1].set_title('Temperature vs Humidity (color: precipitation)')
        plt.colorbar(scatter, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig(f'{save_path}weather_overview.png', dpi=300, bbox_inches='tight')
        print(f"Overview plots saved to {save_path}weather_overview.png")
        plt.close()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(10, 8))
        numerical_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation', 'cloud_cover']
        correlation_matrix = self.data[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f')
        plt.title('Weather Features Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{save_path}correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {save_path}correlation_heatmap.png")
        plt.close()
        
        # 3. Monthly patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        monthly_stats = self.data.groupby('month').agg({
            'temperature': 'mean',
            'humidity': 'mean',
            'precipitation': 'mean',
            'wind_speed': 'mean'
        })
        
        monthly_stats['temperature'].plot(kind='bar', ax=axes[0,0], color='orange')
        axes[0,0].set_title('Average Temperature by Month')
        axes[0,0].set_ylabel('Temperature (°F)')
        axes[0,0].set_xlabel('Month')
        
        monthly_stats['humidity'].plot(kind='bar', ax=axes[0,1], color='blue')
        axes[0,1].set_title('Average Humidity by Month')
        axes[0,1].set_ylabel('Humidity (%)')
        axes[0,1].set_xlabel('Month')
        
        monthly_stats['precipitation'].plot(kind='bar', ax=axes[1,0], color='green')
        axes[1,0].set_title('Average Precipitation by Month')
        axes[1,0].set_ylabel('Precipitation (inches)')
        axes[1,0].set_xlabel('Month')
        
        monthly_stats['wind_speed'].plot(kind='bar', ax=axes[1,1], color='purple')
        axes[1,1].set_title('Average Wind Speed by Month')
        axes[1,1].set_ylabel('Wind Speed (mph)')
        axes[1,1].set_xlabel('Month')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}monthly_patterns.png', dpi=300, bbox_inches='tight')
        print(f"Monthly patterns saved to {save_path}monthly_patterns.png")
        plt.close()
    
    def prepare_features(self):
        """Prepare features for machine learning."""
        if self.data is None:
            print("Please load data first using load_data()")
            return None
        
        # Create a copy for processing
        processed_df = self.data.copy()
        
        # Create additional time-based features
        processed_df['year'] = processed_df['date'].dt.year
        processed_df['month_sin'] = np.sin(2 * np.pi * processed_df['month'] / 12)
        processed_df['month_cos'] = np.cos(2 * np.pi * processed_df['month'] / 12)
        processed_df['day_of_year_sin'] = np.sin(2 * np.pi * processed_df['day_of_year'] / 365)
        processed_df['day_of_year_cos'] = np.cos(2 * np.pi * processed_df['day_of_year'] / 365)
        
        # Create lag features (previous day's weather)
        processed_df['temp_lag1'] = processed_df['temperature'].shift(1)
        processed_df['humidity_lag1'] = processed_df['humidity'].shift(1)
        processed_df['pressure_lag1'] = processed_df['pressure'].shift(1)
        
        # Create rolling averages (3-day and 7-day)
        processed_df['temp_roll3'] = processed_df['temperature'].rolling(window=3).mean()
        processed_df['temp_roll7'] = processed_df['temperature'].rolling(window=7).mean()
        
        # Encode categorical variables
        # One-hot encoding for weather conditions
        weather_encoded = pd.get_dummies(processed_df['weather_condition'], prefix='weather')
        processed_df = pd.concat([processed_df, weather_encoded], axis=1)
        
        # Label encoding for season
        le_season = LabelEncoder()
        processed_df['season_encoded'] = le_season.fit_transform(processed_df['season'])
        self.label_encoders['season'] = le_season
        
        # Label encoding for temperature category
        le_temp_cat = LabelEncoder()
        processed_df['temp_category_encoded'] = le_temp_cat.fit_transform(processed_df['temp_category'])
        self.label_encoders['temp_category'] = le_temp_cat
        
        # Drop rows with NaN values (from lag and rolling features)
        processed_df = processed_df.dropna()
        
        self.processed_data = processed_df
        print(f"Feature engineering completed. Final dataset shape: {processed_df.shape}")
        
        return processed_df
    
    def get_feature_sets(self):
        """Get different feature sets for modeling."""
        if self.processed_data is None:
            print("Please run prepare_features() first")
            return None
        
        # Basic weather features
        basic_features = ['humidity', 'pressure', 'wind_speed', 'precipitation', 'cloud_cover']
        
        # Time-based features
        time_features = ['month', 'day_of_year', 'month_sin', 'month_cos', 
                        'day_of_year_sin', 'day_of_year_cos', 'is_weekend']
        
        # Lag features
        lag_features = ['temp_lag1', 'humidity_lag1', 'pressure_lag1', 'temp_roll3', 'temp_roll7']
        
        # Encoded categorical features
        categorical_features = ['season_encoded'] + [col for col in self.processed_data.columns if col.startswith('weather_')]
        
        # All features combined
        all_features = basic_features + time_features + lag_features + categorical_features
        
        feature_sets = {
            'basic': basic_features,
            'time': basic_features + time_features,
            'with_lags': basic_features + time_features + lag_features,
            'all': all_features
        }
        
        return feature_sets
    
    def create_train_test_splits(self, target_column='temperature', test_size=0.2, random_state=42):
        """Create train/test splits for modeling."""
        if self.processed_data is None:
            print("Please run prepare_features() first")
            return None
        
        feature_sets = self.get_feature_sets()
        splits = {}
        
        y = self.processed_data[target_column]
        
        for name, features in feature_sets.items():
            X = self.processed_data[features]
            
            # Handle any remaining NaN values - only for numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X.loc[:, numeric_cols] = X.loc[:, numeric_cols].fillna(X.loc[:, numeric_cols].mean())
            
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=False
            )
            
            # Scale only numeric features
            scaler = StandardScaler()
            numeric_features = X_train.select_dtypes(include=[np.number]).columns
            
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            if len(numeric_features) > 0:
                X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
                X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
            else:
                # If no numeric features, use the data as-is
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            splits[name] = {
                'X_train': X_train,
                'X_test': X_test,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler,
                'features': features
            }
        
        print(f"Created train/test splits for {len(splits)} feature sets")
        print(f"Training size: {len(y_train)}, Test size: {len(y_test)}")
        
        return splits

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = WeatherDataPreprocessor('../data/nyc_weather_data.csv')
    
    # Load and explore data
    if preprocessor.load_data():
        print("\n" + "="*50)
        preprocessor.explore_data()
        
        print("\n" + "="*50)
        print("Creating visualizations...")
        preprocessor.create_visualizations()
        
        print("\n" + "="*50)
        print("Preparing features for ML...")
        preprocessor.prepare_features()
        
        print("\n" + "="*50)
        print("Creating train/test splits...")
        splits = preprocessor.create_train_test_splits()
        
        if splits:
            print("\nFeature sets created:")
            for name, split_data in splits.items():
                print(f"  {name}: {len(split_data['features'])} features")
            
            print("\nPreprocessing completed successfully!")
            print("Data is ready for model training.")
    else:
        print("Failed to load data. Please check the file path.")
