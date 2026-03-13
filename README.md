# NYC Weather Prediction Project

A comprehensive machine learning project for predicting weather trends in New York City using scikit-learn. This project demonstrates multiple ML algorithms, feature engineering, and model evaluation techniques applied to weather forecasting.

## Project Overview

This project uses historical weather data to train machine learning models capable of predicting temperature trends for NYC. It includes:

- **Mock Weather Data Generation**: Creates realistic synthetic weather data with seasonal patterns
- **Data Preprocessing**: Advanced feature engineering including lag variables and cyclical encoding
- **Multiple ML Models**: Comparison of 8 different algorithms (Linear Regression, Random Forest, SVM, etc.)
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **Future Predictions**: 7-day weather forecast capability
- **User-Friendly Interface**: Both command-line and interactive modes

## Project Structure

```
weather_ml_project/
├── data/
│   └── nyc_weather_data.csv          # Generated weather dataset
├── models/
│   └── best_weather_model.joblib     # Trained model file
├── scripts/
│   ├── generate_data.py              # Data generation script
│   ├── preprocess_data.py            # Data preprocessing and EDA
│   └── weather_model_fixed.py       # ML model training
├── visualizations/                   # Generated plots and charts
│   ├── weather_overview.png
│   ├── correlation_heatmap.png
│   ├── monthly_patterns.png
│   ├── model_evaluation.png
│   └── temperature_predictions.png
├── predict_weather.py               # Main prediction interface
└── README.md
```

## Start

### Prerequisites

- Python 3.7+
- Virtual environment (recommended)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd weather_ml_project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv weather_prediction_env
   source weather_prediction_env/bin/activate  # On Windows: weather_prediction_env\\Scripts\\activate
   ```

3. **Install required packages:**
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn joblib
   ```

### Quick Start

1. **Run the main prediction tool:**
   ```bash
   python predict_weather.py
   ```

2. **Or use command-line options:**
   ```bash
   # Predict weather for next 5 days
   python predict_weather.py --days 5
   
   # Show recent weather
   python predict_weather.py --recent 10
   
   # Show model information
   python predict_weather.py --info
   
   # Train a new model
   python predict_weather.py --train
   ```

## Features

### Data Generation
- **Realistic Weather Patterns**: Generates 5 years of synthetic NYC weather data
- **Seasonal Variations**: Temperature follows sinusoidal patterns with appropriate seasonal changes
- **Correlated Variables**: Humidity, pressure, wind speed, and precipitation are realistically correlated
- **Weather Conditions**: Categorical weather states based on temperature and precipitation

### Feature Engineering
- **Lag Variables**: Previous day's weather conditions
- **Rolling Averages**: 3-day and 7-day temperature moving averages  
- **Cyclical Encoding**: Seasonal patterns using sine/cosine transformations
- **Categorical Encoding**: One-hot encoding for weather conditions
- **Derived Features**: Weekend indicators, temperature categories, etc.

### Machine Learning Models
The project compares 8 different algorithms:

1. **Linear Regression**
2. **Ridge Regression** 
3. **Lasso Regression**
4. **Random Forest**
5. **Gradient Boosting**
6. **Support Vector Machine (SVM)**
7. **K-Nearest Neighbors**
8. **Decision Tree**

### Model Performance
Best results achieved with **Linear Regression** on lag features:
- **RMSE**: 5.66°F
- **R² Score**: 0.918
- **MAE**: 4.55°F

## Results & Visualizations

The project generates several informative visualizations:

1. **Weather Overview**: Temperature trends, seasonal distributions, weather conditions
2. **Correlation Heatmap**: Feature relationships and multicollinearity analysis  
3. **Monthly Patterns**: Average temperature, humidity, precipitation, and wind by month
4. **Model Evaluation**: Performance comparisons, actual vs predicted plots, residual analysis
5. **Future Predictions**: 7-day forecast with historical context

## 🔧 Usage Examples

### Interactive Mode
```bash
python predict_weather.py --interactive
```
Launches a user-friendly menu system for exploring the model and making predictions.

### Command Line Examples
```bash
# Quick 7-day prediction
python predict_weather.py

# Custom prediction period
python predict_weather.py --days 14

# View recent weather history
python predict_weather.py --recent 20

# Get model details
python predict_weather.py --info

# Retrain the model
python predict_weather.py --train
```

### Programmatic Usage
```python
from scripts.weather_model_fixed import WeatherPredictor

# Load trained model
predictor = WeatherPredictor.load_model('models/best_weather_model.joblib')

# Make predictions
predictions = predictor.predict_future_weather(days_ahead=7)
print(predictions)
```

## Model Development Process

1. **Data Generation**: Create synthetic but realistic weather dataset
2. **Exploratory Data Analysis**: Understand patterns and relationships
3. **Feature Engineering**: Create predictive features from raw data
4. **Model Training**: Train and compare multiple algorithms
5. **Model Evaluation**: Assess performance using multiple metrics
6. **Hyperparameter Tuning**: Optimize best-performing models
7. **Model Deployment**: Save and create prediction interface

## 📈 Performance Metrics

The models are evaluated using multiple metrics:

- **RMSE** (Root Mean Squared Error): Measures prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction error magnitude  
- **R² Score**: Proportion of variance explained by the model
- **Residual Analysis**: Checks for patterns in prediction errors

## Future Enhancements

Potential improvements and extensions:

- **Real Weather Data**: Integration with weather APIs for live data
- **Additional Features**: Wind direction, atmospheric pressure trends, UV index
- **Deep Learning**: LSTM networks for time series forecasting  
- **Ensemble Methods**: Combining multiple models for better predictions
- **Weather Classification**: Predicting categorical weather conditions
- **Climate Analysis**: Long-term trend analysis and climate modeling
- **Web Interface**: Flask/Django web application for broader access

## Technical Details

### Dataset Features
- **Date Range**: 2020-2024 (5 years, 1,827 observations)
- **Features**: Temperature, humidity, pressure, wind speed, precipitation, cloud cover
- **Derived Variables**: 26+ engineered features including lags and cyclical encodings

### Model Architecture
- **Preprocessing**: StandardScaler for numeric features
- **Feature Selection**: Multiple feature sets tested (basic, time, lags, all)
- **Cross-Validation**: 5-fold CV for hyperparameter tuning
- **Train/Test Split**: 80/20 split with temporal ordering preserved

