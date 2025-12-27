"""
Machine Learning Models Module
Implements XGBoost regression model for water consumption forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

import xgboost as xgb


class WaterDemandModels:
    """XGBoost regression model for water demand forecasting"""

    def __init__(self):
        self.model = None
        self.scalers = {}
        self.feature_names = []
        self.results = {}

    # --------------------------------------------------
    # 1. Initialize model
    # --------------------------------------------------
    def initialize_models(self):
        """Initialize the XGBoost model"""
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            verbosity=0
        )

    # --------------------------------------------------
    # 2. Prepare data
    # --------------------------------------------------
    def prepare_data(self, df: pd.DataFrame, target_col="Total_Liters"):
        exclude_cols = [
            "date", "Total_Liters", "Bathroom_Liters", "Kitchen_Liters",
            "Laundry_Liters", "Gardening_Liters", "festival_name",
            "Household_ID", "temp_category", "rainfall_category",
            "humidity_category", "season"
        ]

        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and df[c].dtype in ["int64", "float64"]
        ]

        X = df[feature_cols].copy()
        y = df[target_col].copy()
        self.feature_names = feature_cols

        return X, y

    # --------------------------------------------------
    # 3. Scaling
    # --------------------------------------------------
    def scale_features(self, X_train, X_val=None, X_test=None):
        scaler = StandardScaler()

        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        self.scalers["standard"] = scaler
        outputs = [X_train_s]

        if X_val is not None:
            outputs.append(pd.DataFrame(
                scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            ))

        if X_test is not None:
            outputs.append(pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            ))

        return tuple(outputs)

    # --------------------------------------------------
    # 4. Train model
    # --------------------------------------------------
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model"""
        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100

        self.results = {
            "xgboost": {
                "train_rmse": float(train_rmse),
                "train_mae": float(train_mae),
                "train_r2": float(train_r2),
                "train_mape": float(train_mape)
            }
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_mape = mean_absolute_percentage_error(y_val, val_pred) * 100
            
            self.results["xgboost"].update({
                "val_rmse": float(val_rmse),
                "val_mae": float(val_mae),
                "val_r2": float(val_r2),
                "val_mape": float(val_mape)
            })

        return pd.DataFrame(self.results).T

    # --------------------------------------------------
    # 5. Predict
    # --------------------------------------------------
    def predict(self, model_name, X):
        """Make predictions using the XGBoost model"""
        return self.model.predict(X)

    # --------------------------------------------------
    # 6. Evaluation
    # --------------------------------------------------
    def evaluate_on_test(self, model_name, X_test, y_test):
        """Evaluate model on test set"""
        preds = self.predict(model_name, X_test)

        return {
            "test_rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "test_mae": mean_absolute_error(y_test, preds),
            "test_r2": r2_score(y_test, preds),
            "test_mape": mean_absolute_percentage_error(y_test, preds) * 100
        }

    # --------------------------------------------------
    # 7. Save model
    # --------------------------------------------------
    def save_model(self, model_name, path):
        """Save the trained model to disk"""
        joblib.dump(self.model, path)
    
    # --------------------------------------------------
    # 8. Get feature importance
    # --------------------------------------------------
    def get_feature_importance(self, model_name, top_n=15):
        """Get feature importance from XGBoost model"""
        if self.model is None:
            return None
        
        importance_dict = self.model.get_booster().get_score(importance_type='weight')
        importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df


# --------------------------------------------------
# Utility metrics function
# --------------------------------------------------
def calculate_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100
    }
