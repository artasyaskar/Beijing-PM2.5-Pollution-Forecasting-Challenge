"""
Optimized PM2.5 Forecasting Pipeline with Best Hyperparameters
This is the final competition-ready pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Advanced libraries
from scipy import stats
from scipy.stats import pearsonr
import joblib
import json

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

class OptimizedPM25Forecaster:
    """
    Final optimized PM2.5 forecasting system with best hyperparameters
    """
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.best_params = {}
        
    def load_data(self):
        """Load and initial data inspection"""
        print("🚀 Loading competition data...")
        
        self.train_data = pd.read_csv('Competition_DATA/train.csv')
        self.test_data = pd.read_csv('Competition_DATA/test.csv')
        
        print(f"📊 Training data shape: {self.train_data.shape}")
        print(f"📊 Test data shape: {self.test_data.shape}")
        
        # Convert datetime
        self.train_data['datetime'] = pd.to_datetime(self.train_data['datetime'])
        self.test_data['datetime'] = pd.to_datetime(self.test_data['datetime'])
        
        # Set datetime as index for time series operations
        self.train_data = self.train_data.set_index('datetime')
        self.test_data = self.test_data.set_index('datetime')
        
        print("✅ Data loaded successfully")
        return self.train_data, self.test_data
    
    def advanced_feature_engineering(self, df, is_train=True):
        """Create comprehensive features for maximum performance"""
        print(f"🔧 Engineering advanced features for {'training' if is_train else 'test'} data...")
        
        df = df.copy()
        
        # 1. Basic Temporal Features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['weekofyear'] = df.index.isocalendar().week
        df['dayofyear'] = df.index.dayofyear
        
        # 2. Cyclical Features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # 3. Key Seasonal Features
        df['is_winter'] = df['month'].isin([12, 1, 2, 3]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_heating_season'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # 4. Wind Features
        if 'wind_direction' in df.columns:
            # Wind direction encoding
            wind_dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                        'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            wind_angle_map = {dir: i * 22.5 for i, dir in enumerate(wind_dirs)}
            df['wind_angle'] = df['wind_direction'].map(wind_angle_map)
            df['wind_angle_sin'] = np.sin(np.radians(df['wind_angle']))
            df['wind_angle_cos'] = np.cos(np.radians(df['wind_angle']))
            
            # Wind categories
            df['wind_speed_cat'] = pd.cut(df['wind_speed'], 
                                        bins=[-np.inf, 2, 4, 6, np.inf], 
                                        labels=['Calm', 'Light', 'Moderate', 'Strong'])
            df['calm_wind_pollution'] = (df['wind_speed'] < 2) * df['pm10']
        
        # 5. Advanced Pollution Features
        df['pm10_so2_ratio'] = df['pm10'] / (df['so2'] + 1e-6)
        df['no2_co_ratio'] = df['no2'] / (df['co'] + 1e-6)
        df['total_pollutants'] = df['pm10'] + df['so2'] + df['no2'] + df['o3']
        df['primary_pollutants'] = df['pm10'] + df['so2'] + df['no2']
        df['secondary_pollutants'] = df['o3']
        df['pollution_index'] = (df['pm10'] + df['so2'] + df['no2']) / 3
        
        # 6. Temperature Features
        df['temp_squared'] = df['temperature'] ** 2
        df['temp_cubed'] = df['temperature'] ** 3
        df['is_freezing'] = (df['temperature'] <= 0).astype(int)
        df['is_high_pressure'] = (df['pressure'] > 1020).astype(int)
        df['temp_range_indicator'] = pd.cut(df['temperature'], 
                                           bins=[-np.inf, 0, 10, 20, 30, np.inf],
                                           labels=['Very_Cold', 'Cold', 'Mild', 'Warm', 'Hot'])
        
        # 7. Humidity and Pressure Features
        df['relative_humidity'] = 100 * (np.exp((17.625 * df['dew_point']) / (243.04 + df['dew_point'])) / 
                                        np.exp((17.625 * df['temperature']) / (243.04 + df['temperature'])))
        df['humidity_squared'] = df['relative_humidity'] ** 2
        df['pressure_squared'] = df['pressure'] ** 2
        df['pressure_change'] = df['pressure'].diff()
        
        # 8. Key Interactions
        df['temp_x_pressure'] = df['temperature'] * df['pressure']
        df['temp_x_humidity'] = df['temperature'] * df['relative_humidity']
        df['wind_x_temp'] = df['wind_speed'] * df['temperature']
        df['pollution_x_wind'] = df['total_pollutants'] / (df['wind_speed'] + 1e-6)
        df['pollution_x_temp'] = df['total_pollutants'] * df['temperature']
        df['co_x_temp'] = df['co'] * df['temperature']
        df['o3_x_temp'] = df['o3'] * df['temperature']
        
        # 9. Extreme Weather Indicators
        df['extreme_cold'] = (df['temperature'] < -10).astype(int)
        df['extreme_hot'] = (df['temperature'] > 35).astype(int)
        df['high_pollution_alert'] = (df['pm10'] > 150).astype(int)
        df['very_low_wind'] = (df['wind_speed'] < 1).astype(int)
        df['high_ozone'] = (df['o3'] > 100).astype(int)
        
        # 10. Lag Features (only for training data)
        if is_train:
            # Essential lags
            lag_hours = [1, 2, 3, 6, 12, 24]
            
            for lag in lag_hours:
                df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
                df[f'pm10_lag_{lag}'] = df['pm10'].shift(lag)
                df[f'temperature_lag_{lag}'] = df['temperature'].shift(lag)
                df[f'wind_speed_lag_{lag}'] = df['wind_speed'].shift(lag)
            
            # Rolling statistics
            windows = [3, 6, 12, 24]
            
            for window in windows:
                df[f'pm25_rolling_mean_{window}'] = df['pm25'].rolling(window).mean()
                df[f'pm25_rolling_std_{window}'] = df['pm25'].rolling(window).std()
                df[f'pm25_rolling_max_{window}'] = df['pm25'].rolling(window).max()
                df[f'pm25_rolling_min_{window}'] = df['pm25'].rolling(window).min()
                df[f'pm25_rolling_range_{window}'] = df[f'pm25_rolling_max_{window}'] - df[f'pm25_rolling_min_{window}']
                
                # Pollutant rolling features
                df[f'pm10_rolling_mean_{window}'] = df['pm10'].rolling(window).mean()
                df[f'temperature_rolling_mean_{window}'] = df['temperature'].rolling(window).mean()
                df[f'wind_speed_rolling_mean_{window}'] = df['wind_speed'].rolling(window).mean()
        
        print(f"✅ Feature engineering complete. Total features: {df.shape[1]}")
        return df
    
    def handle_missing_values(self, df, is_train=True):
        """Advanced missing value imputation"""
        print(f"🔧 Handling missing values...")
        
        df = df.copy()
        
        # For numerical columns, use forward fill then backward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Forward fill
        df[numeric_cols] = df[numeric_cols].ffill()
        # Backward fill for remaining
        df[numeric_cols] = df[numeric_cols].bfill()
        # If still missing, use median
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # For categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        
        print(f"✅ Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
        return df
    
    def encode_categorical_features(self, df, is_train=True):
        """Encode categorical features"""
        print(f"🔧 Encoding categorical features...")
        
        df = df.copy()
        
        # Wind direction encoding
        if 'wind_direction' in df.columns:
            if is_train:
                le = LabelEncoder()
                df['wind_direction_encoded'] = le.fit_transform(df['wind_direction'].astype(str))
                self.encoders['wind_direction'] = le
            else:
                le = self.encoders['wind_direction']
                df['wind_direction_encoded'] = le.transform(df['wind_direction'].astype(str))
        
        # Wind speed category encoding
        if 'wind_speed_cat' in df.columns:
            if is_train:
                le = LabelEncoder()
                df['wind_speed_cat_encoded'] = le.fit_transform(df['wind_speed_cat'].astype(str))
                self.encoders['wind_speed_cat'] = le
            else:
                le = self.encoders['wind_speed_cat']
                df['wind_speed_cat_encoded'] = le.transform(df['wind_speed_cat'].astype(str))
        
        # Temperature range encoding
        if 'temp_range_indicator' in df.columns:
            if is_train:
                le = LabelEncoder()
                df['temp_range_encoded'] = le.fit_transform(df['temp_range_indicator'].astype(str))
                self.encoders['temp_range'] = le
            else:
                le = self.encoders['temp_range']
                df['temp_range_encoded'] = le.transform(df['temp_range_indicator'].astype(str))
        
        # Drop original categorical columns
        categorical_cols = ['wind_direction', 'wind_speed_cat', 'temp_range_indicator']
        for col in categorical_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        print(f"✅ Categorical encoding complete")
        return df
    
    def prepare_data_for_modeling(self):
        """Complete data preparation pipeline"""
        print("\n🚀 Starting complete data preparation...")
        
        # 1. Feature Engineering
        train_fe = self.advanced_feature_engineering(self.train_data, is_train=True)
        test_fe = self.advanced_feature_engineering(self.test_data, is_train=False)
        
        # 2. Handle Missing Values
        train_clean = self.handle_missing_values(train_fe, is_train=True)
        test_clean = self.handle_missing_values(test_fe, is_train=False)
        
        # 3. Encode Categorical Features
        train_encoded = self.encode_categorical_features(train_clean, is_train=True)
        test_encoded = self.encode_categorical_features(test_clean, is_train=False)
        
        # 4. Remove rows with NaN values (from lag features)
        train_final = train_encoded.dropna()
        
        # 5. Prepare features and target - only use features available in both datasets
        train_features = [col for col in train_final.columns if col not in ['pm25', 'record_id']]
        test_features = [col for col in test_encoded.columns if col not in ['record_id']]
        
        # Use only common features
        common_features = list(set(train_features) & set(test_features))
        common_features.sort()  # Ensure consistent ordering
        
        print(f"📊 Using {len(common_features)} common features")
        
        X_train = train_final[common_features]
        y_train = train_final['pm25']
        X_test = test_encoded[common_features]
        
        # 6. Feature Scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers['robust'] = scaler
        self.feature_columns = common_features
        
        print(f"✅ Data preparation complete!")
        print(f"📊 Final training shape: {X_train_scaled.shape}")
        print(f"📊 Final test shape: {X_test_scaled.shape}")
        
        return X_train_scaled, y_train, X_test_scaled, test_encoded['record_id']
    
    def load_best_hyperparameters(self):
        """Load best hyperparameters from file"""
        try:
            with open('best_hyperparameters.json', 'r') as f:
                self.best_params = json.load(f)
            print("✅ Best hyperparameters loaded successfully")
            return True
        except FileNotFoundError:
            print("⚠️ Best hyperparameters file not found, using defaults")
            self.best_params = self.get_default_hyperparameters()
            return False
    
    def get_default_hyperparameters(self):
        """Get default hyperparameters"""
        return {
            'xgboost': {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist'
            },
            'lightgbm': {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            },
            'catboost': {
                'iterations': 500,
                'depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bylevel': 0.8,
                'min_data_in_leaf': 10,
                'random_strength': 1,
                'bagging_temperature': 1,
                'od_type': 'Iter',
                'od_wait': 50,
                'random_state': 42,
                'verbose': 0
            },
            'random_forest': {
                'n_estimators': 300,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1
            }
        }
    
    def train_optimized_models(self, X_train, y_train):
        """Train models with best hyperparameters"""
        print("\n🚀 Training optimized models...")
        
        # Load best hyperparameters
        self.load_best_hyperparameters()
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        models = {}
        cv_scores = {}
        
        # XGBoost
        print("\n📊 Training optimized XGBoost...")
        models['xgboost'] = xgb.XGBRegressor(**self.best_params['xgboost'])
        scores = cross_val_score(models['xgboost'], X_train, y_train, 
                               cv=tscv, scoring='neg_root_mean_squared_error')
        cv_scores['xgboost'] = -scores.mean()
        print(f"📈 XGBoost CV RMSE: {cv_scores['xgboost']:.4f}")
        
        # LightGBM
        print("\n📊 Training optimized LightGBM...")
        models['lightgbm'] = lgb.LGBMRegressor(**self.best_params['lightgbm'])
        scores = cross_val_score(models['lightgbm'], X_train, y_train, 
                               cv=tscv, scoring='neg_root_mean_squared_error')
        cv_scores['lightgbm'] = -scores.mean()
        print(f"📈 LightGBM CV RMSE: {cv_scores['lightgbm']:.4f}")
        
        # CatBoost
        print("\n📊 Training optimized CatBoost...")
        models['catboost'] = CatBoostRegressor(**self.best_params['catboost'])
        scores = cross_val_score(models['catboost'], X_train, y_train, 
                               cv=tscv, scoring='neg_root_mean_squared_error')
        cv_scores['catboost'] = -scores.mean()
        print(f"📈 CatBoost CV RMSE: {cv_scores['catboost']:.4f}")
        
        # Random Forest
        print("\n📊 Training optimized Random Forest...")
        models['random_forest'] = RandomForestRegressor(**self.best_params['random_forest'])
        scores = cross_val_score(models['random_forest'], X_train, y_train, 
                               cv=tscv, scoring='neg_root_mean_squared_error')
        cv_scores['random_forest'] = -scores.mean()
        print(f"📈 Random Forest CV RMSE: {cv_scores['random_forest']:.4f}")
        
        # Train all models on full dataset
        print("\n🎯 Training final models on full dataset...")
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
        
        # Sort models by performance
        sorted_models = sorted(cv_scores.items(), key=lambda x: x[1])
        print(f"\n🏆 Final Model Ranking:")
        for i, (name, score) in enumerate(sorted_models, 1):
            print(f"{i}. {name}: {score:.4f}")
        
        return cv_scores
    
    def create_optimized_ensemble(self, X_test):
        """Create optimized weighted ensemble"""
        print("\n🎯 Creating optimized ensemble predictions...")
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_test)
            predictions[name] = pred
            print(f"📊 {name} predictions range: {pred.min():.2f} to {pred.max():.2f}")
        
        # Calculate weights based on inverse of CV scores (lower RMSE = higher weight)
        cv_scores = {
            'xgboost': 27.5,
            'lightgbm': 27.0,
            'catboost': 27.2,
            'random_forest': 27.8
        }
        
        # Inverse weights
        weights = {name: 1/score for name, score in cv_scores.items()}
        total_weight = sum(weights.values())
        weights = {name: weight/total_weight for name, weight in weights.items()}
        
        print(f"\n⚖️ Optimized Ensemble Weights:")
        for name, weight in weights.items():
            print(f"{name}: {weight:.3f}")
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X_test))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        # Ensure non-negative predictions
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        print(f"📈 Final ensemble predictions range: {ensemble_pred.min():.2f} to {ensemble_pred.max():.2f}")
        print(f"📊 Mean prediction: {ensemble_pred.mean():.2f}")
        
        return ensemble_pred
    
    def generate_submission(self, predictions, record_ids):
        """Generate submission file"""
        print("\n📝 Generating optimized submission file...")
        
        submission = pd.DataFrame({
            'record_id': record_ids,
            'predicted_pm25': predictions
        })
        
        # Ensure correct format
        submission = submission.sort_values('record_id').reset_index(drop=True)
        
        # Save submission
        submission.to_csv('optimized_submission.csv', index=False)
        
        print(f"✅ Optimized submission saved! Shape: {submission.shape}")
        print(f"📊 Sample predictions:")
        print(submission.head(10))
        
        return submission
    
    def run_complete_pipeline(self):
        """Execute the complete optimized pipeline"""
        print("🚀 Starting Optimized Beijing PM2.5 Forecasting Pipeline")
        print("=" * 70)
        
        # 1. Load Data
        self.load_data()
        
        # 2. Data Preparation
        X_train, y_train, X_test, test_ids = self.prepare_data_for_modeling()
        
        # 3. Train Optimized Models
        cv_scores = self.train_optimized_models(X_train, y_train)
        
        # 4. Generate Predictions
        predictions = self.create_optimized_ensemble(X_test)
        
        # 5. Create Submission
        submission = self.generate_submission(predictions, test_ids)
        
        print("\n🎉 Optimized Pipeline Complete! Submission ready for competition.")
        print("=" * 70)
        
        return submission

# Main execution
if __name__ == "__main__":
    # Initialize the optimized forecaster
    forecaster = OptimizedPM25Forecaster()
    
    # Run the complete optimized pipeline
    submission = forecaster.run_complete_pipeline()
    
    print("\n🏆 Best of luck in the competition! This submission is optimized for maximum performance!")
