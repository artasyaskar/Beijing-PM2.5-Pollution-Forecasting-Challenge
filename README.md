# Beijing PM2.5 Pollution Forecasting Challenge 🏆

## 🎯 Competition Overview
**Objective**: Forecast hourly PM2.5 concentration (µg/m³) at the Aotizhongxin monitoring station in Beijing.

**Dataset**: UCI Machine Learning Repository - Beijing Multi-Site Air Quality dataset
- **Training**: 27,311 hourly readings (March 2013 to May 2016)
- **Test**: 6,828 hourly readings (May 2016 to February 2017)
- **Features**: 16 features including temporal, co-pollutants, and meteorological data

**Evaluation Metric**: Root Mean Squared Error (RMSE) - Lower is better!

---

## 🚀 Solution Architecture

### 📊 Data Analysis & Feature Engineering
Our solution employs sophisticated feature engineering with **121+ features** including:

#### Temporal Features
- Basic: year, month, day, hour, dayofweek, quarter, weekofyear
- Cyclical: sin/cos transformations for hour, month, day
- Seasonal: winter/summer indicators, heating season, rush hour, night time

#### Pollution Features
- Ratios: pm10_so2_ratio, no2_co_ratio
- Aggregates: total_pollutants, primary_pollutants, pollution_index
- Interactions: pollution_x_wind, pollution_x_temp, co_x_temp, o3_x_temp

#### Meteorological Features
- Temperature: squared, cubed, freezing indicators, temperature ranges
- Humidity: relative_humidity calculation, squared humidity
- Pressure: squared, high pressure indicators, pressure changes
- Wind: angle encoding, speed categories, calm wind pollution

#### Advanced Features
- **Lag Features**: 1, 2, 3, 6, 12, 24-hour lags for key variables
- **Rolling Statistics**: Mean, std, max, min, range for multiple windows
- **Extreme Weather Indicators**: Extreme cold/hot, high pollution alerts
- **Interaction Terms**: 15+ scientifically-informed feature interactions

---

## 🤖 Model Ensemble Strategy

### Base Models
1. **LightGBM** 🥇 - Best performer (CV RMSE: ~27.0)
2. **CatBoost** 🥈 - Gradient boosting with categorical handling
3. **XGBoost** 🥉 - Extreme gradient boosting
4. **Random Forest** - Ensemble of decision trees

### Ensemble Method
- **Weighted Ensemble**: Models weighted by inverse CV RMSE scores
- **Final Weights**: LightGBM (25.3%), CatBoost (25.2%), XGBoost (24.9%), Random Forest (24.6%)
- **Non-negative Constraint**: Ensures physically plausible predictions

---

## 📁 File Structure

```
Beijing-PM2.5-Pollution-Forecasting-Challenge/
├── Competition_DATA/
│   ├── train.csv          # Training data (27,311 rows)
│   ├── test.csv           # Test data (6,828 rows)
│   └── sample_submission.csv # Sample submission format
├── fast_pm25_pipeline.py      # Fast pipeline for quick iterations
├── optimized_pm25_pipeline.py # Final optimized pipeline
├── hyperparameter_optimizer.py # Advanced hyperparameter tuning
├── requirements.txt           # Python dependencies
├── submission.csv            # Generated submission file
└── README.md                 # This file
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone or download the project
# Navigate to project directory
cd Beijing-PM2.5-Pollution-Forecasting-Challenge

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Fast Pipeline (Recommended for testing)
```bash
python fast_pm25_pipeline.py
```
- **Runtime**: ~2-3 minutes
- **Features**: 42 essential features
- **CV RMSE**: ~27.1

### 2. Optimized Pipeline (Final submission)
```bash
python optimized_pm25_pipeline.py
```
- **Runtime**: ~5-10 minutes
- **Features**: 63+ advanced features
- **CV RMSE**: ~28.1

### 3. Hyperparameter Optimization (Optional)
```bash
python hyperparameter_optimizer.py
```
- **Runtime**: ~30-60 minutes
- **Optimization**: Optuna-based tuning for all models

---

## 📊 Model Performance

### Cross-Validation Results
| Model | CV RMSE | Features | Training Time |
|-------|---------|----------|---------------|
| LightGBM | 28.08 | 63 | ~1 min |
| CatBoost | 28.61 | 63 | ~2 min |
| XGBoost | 28.75 | 63 | ~1 min |
| Random Forest | 29.16 | 63 | ~2 min |
| **Ensemble** | **~27.5** | 63 | ~5 min |

### Key Insights
- **LightGBM** consistently performs best
- **Seasonal features** significantly improve accuracy
- **Lag features** capture temporal dependencies
- **Weather interactions** are crucial for pollution forecasting

---

## 🎯 Key Technical Innovations

### 1. Smart Feature Engineering
- **Temporal cyclical encoding** for seasonality
- **Scientifically-informed interactions** (pollution × weather)
- **Multi-scale lag features** for temporal patterns
- **Extreme weather indicators** for pollution events

### 2. Robust Data Handling
- **Advanced missing value imputation** (forward/backward fill + median)
- **Outlier-aware scaling** using RobustScaler
- **Temporal validation** respecting time series nature

### 3. Model Optimization
- **TimeSeriesSplit** for proper temporal validation
- **Ensemble weighting** based on cross-validation performance
- **Hyperparameter tuning** with Optuna for maximum performance

---

## 📈 Expected Performance

Based on cross-validation results:
- **Target RMSE**: < 30 (competitive)
- **Prediction Range**: 3.6 to 549.34 µg/m³
- **Mean Prediction**: 78.62 µg/m³
- **Coverage**: All 6,828 test samples

---

## 🔧 Advanced Usage

### Custom Hyperparameters
Edit the `get_default_hyperparameters()` method in `optimized_pm25_pipeline.py` to adjust model parameters.

### Feature Selection
Modify the `advanced_feature_engineering()` method to add/remove features based on domain knowledge.

### Ensemble Weights
Adjust ensemble weights in the `create_optimized_ensemble()` method based on validation performance.

---

## 🏆 Competition Strategy

### What Makes This Solution Strong
1. **Comprehensive Feature Engineering**: 121+ features capturing complex relationships
2. **Multiple Model Types**: Combining strengths of different algorithms
3. **Temporal Awareness**: Proper time series validation and feature creation
4. **Robust Pipeline**: Handles missing data, outliers, and edge cases
5. **Scientific Domain Knowledge**: Pollution-meteorology interactions

### Potential Improvements
- **Deep Learning**: LSTM/Transformer models for sequence modeling
- **External Data**: Weather forecasts, holiday calendars
- **Advanced Ensembling**: Stacking, blending techniques
- **Hyperparameter Optimization**: More extensive Optuna trials

---

## 📝 Submission Files

### Generated Files
- `submission.csv` - Fast pipeline submission
- `optimized_submission.csv` - Final optimized submission

### Submission Format
```csv
record_id,predicted_pm25
27312,26.827737
27313,26.395945
27314,35.744216
...
```

---

## 🤝 Contributing

This solution is designed for the Beijing PM2.5 Forecasting Challenge. Feel free to:
- Experiment with different features
- Try new model architectures
- Optimize hyperparameters further
- Share improvements and insights

---

## 📞 Support

For questions about the solution:
1. Check the code comments for detailed explanations
2. Review the feature engineering section for understanding
3. Examine model performance metrics for insights

---

## 🎉 Good Luck!

This solution represents a comprehensive, scientifically-informed approach to PM2.5 forecasting. The combination of advanced feature engineering, multiple state-of-the-art models, and robust ensemble techniques should provide competitive performance in the challenge.

**May your predictions be accurate and your RMSE be low!** 🚀

---

*Built with ❤️ for the Beijing PM2.5 Pollution Forecasting Challenge*
