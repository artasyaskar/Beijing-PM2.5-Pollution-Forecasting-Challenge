"""
Advanced Hyperparameter Optimization for PM2.5 Forecasting
Uses Optuna for state-of-the-art hyperparameter tuning
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
import random
random.seed(42)

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna"""
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.best_params = {}
        self.tscv = TimeSeriesSplit(n_splits=3)
        
    def objective_xgboost(self, trial):
        """Objective function for XGBoost optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, self.X_train, self.y_train, 
                               cv=self.tscv, scoring='neg_root_mean_squared_error')
        return -scores.mean()
    
    def objective_lightgbm(self, trial):
        """Objective function for LightGBM optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        scores = cross_val_score(model, self.X_train, self.y_train, 
                               cv=self.tscv, scoring='neg_root_mean_squared_error')
        return -scores.mean()
    
    def objective_catboost(self, trial):
        """Objective function for CatBoost optimization"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
            'od_type': 'Iter',
            'od_wait': 50,
            'random_state': 42,
            'verbose': 0
        }
        
        model = CatBoostRegressor(**params)
        scores = cross_val_score(model, self.X_train, self.y_train, 
                               cv=self.tscv, scoring='neg_root_mean_squared_error')
        return -scores.mean()
    
    def objective_random_forest(self, trial):
        """Objective function for Random Forest optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        scores = cross_val_score(model, self.X_train, self.y_train, 
                               cv=self.tscv, scoring='neg_root_mean_squared_error')
        return -scores.mean()
    
    def optimize_all_models(self, n_trials=50):
        """Optimize all models with specified number of trials"""
        print("🚀 Starting Advanced Hyperparameter Optimization")
        print("=" * 60)
        
        # Optimize XGBoost
        print("\n📊 Optimizing XGBoost...")
        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(self.objective_xgboost, n_trials=n_trials)
        self.best_params['xgboost'] = study_xgb.best_params
        print(f"✅ XGBoost Best RMSE: {study_xgb.best_value:.4f}")
        
        # Optimize LightGBM
        print("\n📊 Optimizing LightGBM...")
        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(self.objective_lightgbm, n_trials=n_trials)
        self.best_params['lightgbm'] = study_lgb.best_params
        print(f"✅ LightGBM Best RMSE: {study_lgb.best_value:.4f}")
        
        # Optimize CatBoost
        print("\n📊 Optimizing CatBoost...")
        study_cat = optuna.create_study(direction='minimize')
        study_cat.optimize(self.objective_catboost, n_trials=n_trials)
        self.best_params['catboost'] = study_cat.best_params
        print(f"✅ CatBoost Best RMSE: {study_cat.best_value:.4f}")
        
        # Optimize Random Forest
        print("\n📊 Optimizing Random Forest...")
        study_rf = optuna.create_study(direction='minimize')
        study_rf.optimize(self.objective_random_forest, n_trials=n_trials)
        self.best_params['random_forest'] = study_rf.best_params
        print(f"✅ Random Forest Best RMSE: {study_rf.best_value:.4f}")
        
        print("\n🎉 Hyperparameter Optimization Complete!")
        return self.best_params
    
    def save_best_params(self, filename='best_hyperparameters.json'):
        """Save best parameters to JSON file"""
        import json
        with open(filename, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print(f"✅ Best parameters saved to {filename}")
    
    def load_best_params(self, filename='best_hyperparameters.json'):
        """Load best parameters from JSON file"""
        import json
        try:
            with open(filename, 'r') as f:
                self.best_params = json.load(f)
            print(f"✅ Best parameters loaded from {filename}")
            return self.best_params
        except FileNotFoundError:
            print(f"❌ File {filename} not found")
            return None

def run_optimization():
    """Run hyperparameter optimization with the prepared data"""
    # Load the fast pipeline to get prepared data
    from fast_pm25_pipeline import FastPM25Forecaster
    
    print("🚀 Preparing data for hyperparameter optimization...")
    
    # Initialize and prepare data
    forecaster = FastPM25Forecaster()
    forecaster.load_data()
    X_train, y_train, X_test, test_ids = forecaster.prepare_data_for_modeling()
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(X_train, y_train)
    
    # Run optimization
    best_params = optimizer.optimize_all_models(n_trials=30)  # Reduced trials for speed
    
    # Save results
    optimizer.save_best_params()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    for model_name, params in best_params.items():
        print(f"\n📊 {model_name.upper()} Best Parameters:")
        for param, value in params.items():
            print(f"  {param}: {value}")
    
    print("\n🎯 Ready to train optimized models!")
    return best_params

if __name__ == "__main__":
    best_params = run_optimization()
