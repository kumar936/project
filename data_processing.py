"""
Data Processing and Feature Engineering Module
Handles data loading, cleaning, and advanced feature engineering for water consumption forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class WaterDataProcessor:
    """Advanced data processor for water consumption forecasting"""
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and perform initial data preparation"""
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive temporal features"""
        df = df.copy()
        
        # Basic temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # Cyclical encoding for periodic features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Week patterns
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Season
        df['season'] = df['month'].apply(self._get_season)
        
        return df
    
    def _get_season(self, month: int) -> str:
        """Determine season from month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'Total_Liters', 
                           lags: List[int] = [1, 2, 3, 7, 14, 30]) -> pd.DataFrame:
        """Create lag features for time series"""
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'Total_Liters',
                               windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """Create rolling window statistics"""
        df = df.copy()
        
        for window in windows:
            # Rolling mean
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(
                window=window, min_periods=1).mean()
            
            # Rolling std
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(
                window=window, min_periods=1).std()
            
            # Rolling min/max
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(
                window=window, min_periods=1).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(
                window=window, min_periods=1).max()
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-based features"""
        df = df.copy()
        
        # Temperature bins
        df['temp_category'] = pd.cut(df['temperature'], 
                                     bins=[0, 15, 20, 25, 100],
                                     labels=['Cold', 'Mild', 'Warm', 'Hot'])
        
        # Rainfall categories
        df['rainfall_category'] = pd.cut(df['rainfall'],
                                         bins=[-1, 0, 5, 10, 100],
                                         labels=['None', 'Light', 'Moderate', 'Heavy'])
        
        # Humidity categories
        df['humidity_category'] = pd.cut(df['humidity'],
                                         bins=[0, 40, 60, 80, 100],
                                         labels=['Low', 'Moderate', 'High', 'Very High'])
        
        # Weather interactions
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        df['temp_squared'] = df['temperature'] ** 2
        
        # Heat index approximation
        df['heat_index'] = df['temperature'] + 0.5 * df['humidity']
        
        # Rainfall lag features
        df['rainfall_lag_1'] = df['rainfall'].shift(1)
        df['rainfall_lag_2'] = df['rainfall'].shift(2)
        df['rainfall_3day_sum'] = df['rainfall'].rolling(window=3, min_periods=1).sum()
        
        return df
    
    def create_consumption_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create consumption pattern features"""
        df = df.copy()
        
        # Usage ratios
        df['bathroom_ratio'] = df['Bathroom_Liters'] / (df['Total_Liters'] + 1e-6)
        df['kitchen_ratio'] = df['Kitchen_Liters'] / (df['Total_Liters'] + 1e-6)
        df['laundry_ratio'] = df['Laundry_Liters'] / (df['Total_Liters'] + 1e-6)
        df['gardening_ratio'] = df['Gardening_Liters'] / (df['Total_Liters'] + 1e-6)
        
        # Indoor vs outdoor usage
        df['indoor_usage'] = df['Bathroom_Liters'] + df['Kitchen_Liters'] + df['Laundry_Liters']
        df['outdoor_usage'] = df['Gardening_Liters']
        df['indoor_outdoor_ratio'] = df['indoor_usage'] / (df['outdoor_usage'] + 1e-6)
        
        return df
    
    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced holiday features"""
        df = df.copy()
        
        # Days to/from holiday
        holiday_dates = df[df['is_public_holiday'] == 1.0]['date']
        
        def days_to_nearest_holiday(date):
            if len(holiday_dates) == 0:
                return 365
            diffs = (holiday_dates - date).dt.days
            future_holidays = diffs[diffs >= 0]
            if len(future_holidays) > 0:
                return future_holidays.min()
            return 365
        
        def days_from_last_holiday(date):
            if len(holiday_dates) == 0:
                return 365
            diffs = (date - holiday_dates).dt.days
            past_holidays = diffs[diffs >= 0]
            if len(past_holidays) > 0:
                return past_holidays.min()
            return 365
        
        df['days_to_holiday'] = df['date'].apply(days_to_nearest_holiday)
        df['days_from_holiday'] = df['date'].apply(days_from_last_holiday)
        
        # Weekend + holiday interaction
        df['is_long_weekend'] = ((df['is_weekend'] == 1.0) & 
                                 ((df['days_to_holiday'] <= 1) | 
                                  (df['days_from_holiday'] <= 1))).astype(int)
        
        return df
    
    def detect_anomalies(self, df: pd.DataFrame, target_col: str = 'Total_Liters',
                        method: str = 'iqr', contamination: float = 0.05) -> pd.DataFrame:
        """Detect anomalies in water consumption"""
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df['is_anomaly'] = ((df[target_col] < lower_bound) | 
                               (df[target_col] > upper_bound)).astype(int)
        
        elif method == 'zscore':
            mean = df[target_col].mean()
            std = df[target_col].std()
            df['zscore'] = (df[target_col] - mean) / std
            df['is_anomaly'] = (np.abs(df['zscore']) > 3).astype(int)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        print("Creating temporal features...")
        df = self.create_temporal_features(df)
        
        print("Creating weather features...")
        df = self.create_weather_features(df)
        
        print("Creating consumption features...")
        df = self.create_consumption_features(df)
        
        print("Creating holiday features...")
        df = self.create_holiday_features(df)
        
        if for_training:
            print("Creating lag features...")
            df = self.create_lag_features(df)
            
            print("Creating rolling features...")
            df = self.create_rolling_features(df)
            
            print("Detecting anomalies...")
            df = self.detect_anomalies(df)
        
        # Fill any NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Time-based train/validation/test split"""
        n = len(df)
        train_size = int(n * (1 - test_size - val_size))
        val_size_idx = int(n * (1 - test_size))
        
        train_df = df[:train_size].copy()
        val_df = df[train_size:val_size_idx].copy()
        test_df = df[val_size_idx:].copy()
        
        return train_df, val_df, test_df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return feature groups for analysis"""
        return {
            'temporal': ['month', 'day_of_week', 'day_of_year', 'quarter', 
                        'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos'],
            'weather': ['temperature', 'rainfall', 'humidity', 'heat_index',
                       'temp_humidity_interaction', 'temp_squared'],
            'lag': [col for col in self.feature_columns if 'lag' in col],
            'rolling': [col for col in self.feature_columns if 'rolling' in col],
            'holiday': ['is_weekend', 'is_public_holiday', 'days_to_holiday', 
                       'days_from_holiday', 'is_long_weekend'],
            'consumption': ['bathroom_ratio', 'kitchen_ratio', 'laundry_ratio',
                          'gardening_ratio', 'indoor_outdoor_ratio']
        }
