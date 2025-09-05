#!/usr/bin/env python3
"""
Weather Data Generator for NYC
Generates realistic mock weather data with seasonal patterns for machine learning.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_weather_data(start_date='2020-01-01', end_date='2024-12-31', random_seed=42):
    """
    Generate realistic weather data for NYC with seasonal patterns.
    
    Features:
    - Temperature (°F) with seasonal variation
    - Humidity (%) 
    - Pressure (hPa)
    - Wind Speed (mph)
    - Precipitation (inches)
    - Cloud Cover (%)
    - Weather condition categories
    """
    np.random.seed(random_seed)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Extract day of year for seasonal patterns
    day_of_year = date_range.dayofyear
    
    # Generate temperature with seasonal variation (NYC climate)
    # Base temperature follows sinusoidal pattern with peak in summer (day 200)
    base_temp = 50 + 25 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp_noise = np.random.normal(0, 8, n_days)  # Daily variation
    temperature = base_temp + temp_noise
    temperature = np.clip(temperature, 10, 95)  # Realistic bounds
    
    # Generate humidity (inversely related to temperature with noise)
    base_humidity = 65 - 0.3 * (temperature - 50)
    humidity_noise = np.random.normal(0, 10, n_days)
    humidity = base_humidity + humidity_noise
    humidity = np.clip(humidity, 20, 95)
    
    # Generate atmospheric pressure (slight seasonal variation)
    base_pressure = 1013 + 5 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
    pressure_noise = np.random.normal(0, 8, n_days)
    pressure = base_pressure + pressure_noise
    pressure = np.clip(pressure, 980, 1040)
    
    # Generate wind speed (higher in winter and spring)
    base_wind = 8 + 4 * np.sin(2 * np.pi * (day_of_year + 90) / 365)
    wind_noise = np.random.exponential(3, n_days)  # Wind has exponential distribution
    wind_speed = base_wind + wind_noise
    wind_speed = np.clip(wind_speed, 0, 40)
    
    # Generate precipitation (higher in summer, some winter storms)
    summer_factor = 0.5 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    winter_storms = 0.2 * (np.sin(2 * np.pi * (day_of_year + 30) / 365) > 0.8)
    precip_probability = summer_factor + winter_storms
    precipitation = np.where(
        np.random.random(n_days) < precip_probability * 0.3,
        np.random.exponential(0.5, n_days),
        0
    )
    precipitation = np.clip(precipitation, 0, 5)
    
    # Generate cloud cover (correlated with precipitation and humidity)
    base_clouds = 40 + 0.4 * humidity + 20 * (precipitation > 0)
    cloud_noise = np.random.normal(0, 15, n_days)
    cloud_cover = base_clouds + cloud_noise
    cloud_cover = np.clip(cloud_cover, 0, 100)
    
    # Generate weather conditions based on temperature, precipitation, and clouds
    conditions = []
    for i in range(n_days):
        if precipitation[i] > 0.5:
            if temperature[i] < 32:
                conditions.append('Snow')
            elif precipitation[i] > 2:
                conditions.append('Heavy Rain')
            else:
                conditions.append('Rain')
        elif cloud_cover[i] > 80:
            conditions.append('Overcast')
        elif cloud_cover[i] > 50:
            conditions.append('Partly Cloudy')
        elif cloud_cover[i] > 20:
            conditions.append('Mostly Sunny')
        else:
            conditions.append('Clear')
    
    # Create DataFrame
    weather_data = pd.DataFrame({
        'date': date_range,
        'temperature': np.round(temperature, 1),
        'humidity': np.round(humidity, 1),
        'pressure': np.round(pressure, 1),
        'wind_speed': np.round(wind_speed, 1),
        'precipitation': np.round(precipitation, 2),
        'cloud_cover': np.round(cloud_cover, 1),
        'weather_condition': conditions
    })
    
    # Add derived features
    weather_data['month'] = weather_data['date'].dt.month
    weather_data['day_of_year'] = weather_data['date'].dt.dayofyear
    weather_data['is_weekend'] = weather_data['date'].dt.weekday >= 5
    weather_data['season'] = weather_data['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Add temperature categories for classification tasks
    weather_data['temp_category'] = pd.cut(
        weather_data['temperature'],
        bins=[0, 32, 50, 70, 85, 100],
        labels=['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot']
    )
    
    return weather_data

def save_weather_data(data, filepath):
    """Save weather data to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath, index=False)
    print(f"Weather data saved to: {filepath}")
    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

if __name__ == "__main__":
    # Generate weather data
    print("Generating NYC weather data...")
    weather_df = generate_weather_data()
    
    # Display basic statistics
    print("\n=== Weather Data Summary ===")
    print(weather_df.describe())
    
    print("\n=== Weather Conditions Distribution ===")
    print(weather_df['weather_condition'].value_counts())
    
    print("\n=== Seasonal Temperature Averages ===")
    print(weather_df.groupby('season')['temperature'].mean().round(1))
    
    # Save to file
    save_weather_data(weather_df, '../data/nyc_weather_data.csv')
    
    print("\n=== Sample Data ===")
    print(weather_df.head(10))
