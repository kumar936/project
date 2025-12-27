"""
Complete Example: Water Demand Analytics End-to-End
This script demonstrates all major features of the system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import WaterDataProcessor
from models import WaterDemandModels, calculate_metrics
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("WATER DEMAND ANALYTICS - COMPLETE EXAMPLE")
print("=" * 80)
print()


# ============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# ============================================================================
print("PART 1: Data Loading and Exploration")
print("-" * 80)

processor = WaterDataProcessor()
df = processor.load_data('../data/water_consumption.csv')

print(f"✓ Loaded {len(df)} records")
print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
print(f"✓ Duration: {(df['date'].max() - df['date'].min()).days} days")
print()

print("Basic Statistics:")
print(df[['Total_Liters', 'temperature', 'rainfall', 'humidity']].describe())
print()


# ============================================================================
# PART 2: FEATURE ENGINEERING
# ============================================================================
print("\nPART 2: Feature Engineering")
print("-" * 80)

print("Creating features...")
df = processor.prepare_features(df, for_training=True)

print(f"✓ Total features: {len(df.columns)}")
print(f"✓ Feature groups:")
groups = processor.get_feature_importance_groups()
for group, features in groups.items():
    print(f"  - {group}: {len(features)} features")
print()


# ============================================================================
# PART 3: DATA ANALYSIS
# ============================================================================
print("\nPART 3: Data Analysis")
print("-" * 80)

# Seasonal analysis
print("\nMonthly Average Consumption:")
monthly_avg = df.groupby('month')['Total_Liters'].mean().round(2)
for month, avg in monthly_avg.items():
    month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
    print(f"  {month_name}: {avg:.2f} L")
print()

# Day of week analysis
print("Day of Week Analysis:")
dow_avg = df.groupby('day_of_week')['Total_Liters'].mean().round(2)
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for dow, avg in dow_avg.items():
    print(f"  {dow_names[dow]}: {avg:.2f} L")
print()

# Holiday impact
if 'is_public_holiday' in df.columns:
    holiday_avg = df[df['is_public_holiday'] == 1]['Total_Liters'].mean()
    regular_avg = df[df['is_public_holiday'] == 0]['Total_Liters'].mean()
    pct_change = ((holiday_avg - regular_avg) / regular_avg) * 100
    print(f"Holiday Impact: {pct_change:+.1f}% change")
    print(f"  Regular days: {regular_avg:.2f} L")
    print(f"  Holidays: {holiday_avg:.2f} L")
    print()

# Weather correlations
print("Weather Correlations with Consumption:")
correlations = df[['temperature', 'rainfall', 'humidity', 'Total_Liters']].corr()['Total_Liters'].drop('Total_Liters')
for weather, corr in correlations.items():
    print(f"  {weather}: {corr:.3f}")
print()


# ============================================================================
# PART 4: ANOMALY DETECTION
# ============================================================================
print("\nPART 4: Anomaly Detection")
print("-" * 80)

if 'is_anomaly' in df.columns:
    anomalies = df[df['is_anomaly'] == 1]
    print(f"✓ Detected {len(anomalies)} anomalies ({len(anomalies)/len(df)*100:.2f}%)")
    print(f"✓ Anomaly consumption range: {anomalies['Total_Liters'].min():.2f} - {anomalies['Total_Liters'].max():.2f} L")
    print(f"✓ Average anomaly value: {anomalies['Total_Liters'].mean():.2f} L")
    
    # Show some examples
    print("\nExample Anomalies:")
    sample_anomalies = anomalies.head(3)[['date', 'Total_Liters', 'temperature', 'rainfall', 'is_weekend']]
    print(sample_anomalies.to_string(index=False))
    print()
else:
    print("No anomaly detection available")
    print()


# ============================================================================
# PART 5: MODEL TRAINING
# ============================================================================
print("\nPART 5: Model Training")
print("-" * 80)

# Split data
train_df, val_df, test_df = processor.split_data(df, test_size=0.2, val_size=0.1)
print(f"✓ Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")
print()

# Initialize models
models = WaterDemandModels()
models.initialize_models()

# Prepare features
X_train, y_train = models.prepare_data(train_df)
X_val, y_val = models.prepare_data(val_df)
X_test, y_test = models.prepare_data(test_df)

print(f"✓ Feature dimensions: {X_train.shape}")
print()

# Scale features
X_train_scaled, X_val_scaled, X_test_scaled = models.scale_features(X_train, X_val, X_test)
print("✓ Features scaled")
print()

# Train selected models (faster for demo)
print("Training models...")
selected_models = ['xgboost', 'lightgbm', 'random_forest']

for model_name in selected_models:
    models.train_model(model_name, X_train_scaled, y_train, X_val_scaled, y_val)

print()


# ============================================================================
# PART 6: MODEL EVALUATION
# ============================================================================
print("\nPART 6: Model Evaluation")
print("-" * 80)

# Compare models
results_df = pd.DataFrame(models.results).T
print("Model Comparison:")
print(results_df[['val_rmse', 'val_mae', 'val_r2', 'val_mape']].to_string())
print()

# Best model
best_model = results_df.nsmallest(1, 'val_rmse').index[0]
print(f"🏆 Best Model: {best_model}")
print()

# Test set evaluation
test_metrics = models.evaluate_on_test(best_model, X_test_scaled, y_test)
print(f"Test Set Performance:")
print(f"  RMSE: {test_metrics['test_rmse']:.2f} liters")
print(f"  MAE: {test_metrics['test_mae']:.2f} liters")
print(f"  R²: {test_metrics['test_r2']:.4f}")
print(f"  MAPE: {test_metrics['test_mape']:.2f}%")
print()


# ============================================================================
# PART 7: FEATURE IMPORTANCE
# ============================================================================
print("\nPART 7: Feature Importance Analysis")
print("-" * 80)

importance_df = models.get_feature_importance(best_model, top_n=15)
if importance_df is not None:
    print("Top 15 Most Important Features:")
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")
    print()


# ============================================================================
# PART 8: FORECASTING EXAMPLES
# ============================================================================
print("\nPART 8: Forecasting Examples")
print("-" * 80)

# Make predictions on test set
y_pred = models.predict(best_model, X_test_scaled)

print("Sample Predictions:")
print(f"{'Date':<12} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
print("-" * 45)
for i in range(min(10, len(y_test))):
    date = test_df.iloc[i]['date'].strftime('%Y-%m-%d')
    actual = y_test.iloc[i]
    pred = y_pred[i]
    error = actual - pred
    print(f"{date:<12} {actual:<10.2f} {pred:<10.2f} {error:>10.2f}")
print()


# ============================================================================
# PART 9: SCENARIO SIMULATION
# ============================================================================
print("\nPART 9: What-If Scenario Simulation")
print("-" * 80)

# Baseline
recent_data = test_df.head(7).copy()
X_baseline, _ = models.prepare_data(recent_data)
X_baseline_scaled = models.scalers['standard'].transform(X_baseline)
baseline_pred = models.predict(best_model, pd.DataFrame(X_baseline_scaled, columns=X_baseline.columns))

print("Baseline Scenario:")
print(f"  Average consumption: {baseline_pred.mean():.2f} L/day")
print(f"  Total (7 days): {baseline_pred.sum():.2f} L")
print()

# Scenario 1: Heat wave (+5°C)
scenario1 = recent_data.copy()
scenario1['temperature'] += 5
scenario1['temp_humidity_interaction'] = scenario1['temperature'] * scenario1['humidity']
scenario1['temp_squared'] = scenario1['temperature'] ** 2
scenario1['heat_index'] = scenario1['temperature'] + 0.5 * scenario1['humidity']

X_scenario1, _ = models.prepare_data(scenario1)
X_scenario1_scaled = models.scalers['standard'].transform(X_scenario1)
scenario1_pred = models.predict(best_model, pd.DataFrame(X_scenario1_scaled, columns=X_scenario1.columns))

print("Scenario 1: Heat Wave (+5°C)")
print(f"  Average consumption: {scenario1_pred.mean():.2f} L/day")
print(f"  Total (7 days): {scenario1_pred.sum():.2f} L")
print(f"  Impact: +{(scenario1_pred.mean() - baseline_pred.mean()):.2f} L/day (+{((scenario1_pred.mean() - baseline_pred.mean())/baseline_pred.mean()*100):.1f}%)")
print()

# Scenario 2: Rainy week (+10mm rainfall)
scenario2 = recent_data.copy()
scenario2['rainfall'] += 10
scenario2['rainfall_3day_sum'] = scenario2['rainfall'].rolling(window=3, min_periods=1).sum()

X_scenario2, _ = models.prepare_data(scenario2)
X_scenario2_scaled = models.scalers['standard'].transform(X_scenario2)
scenario2_pred = models.predict(best_model, pd.DataFrame(X_scenario2_scaled, columns=X_scenario2.columns))

print("Scenario 2: Rainy Week (+10mm)")
print(f"  Average consumption: {scenario2_pred.mean():.2f} L/day")
print(f"  Total (7 days): {scenario2_pred.sum():.2f} L")
print(f"  Impact: {(scenario2_pred.mean() - baseline_pred.mean()):+.2f} L/day ({((scenario2_pred.mean() - baseline_pred.mean())/baseline_pred.mean()*100):+.1f}%)")
print()


# ============================================================================
# PART 10: DEMAND PLANNING INSIGHTS
# ============================================================================
print("\nPART 10: Demand Planning Insights")
print("-" * 80)

# Weekly patterns
weekly_avg = df.groupby('day_of_week')['Total_Liters'].mean()
peak_day = weekly_avg.idxmax()
low_day = weekly_avg.idxmin()

print("Weekly Patterns:")
print(f"  Peak demand day: {dow_names[peak_day]} ({weekly_avg[peak_day]:.2f} L)")
print(f"  Lowest demand day: {dow_names[low_day]} ({weekly_avg[low_day]:.2f} L)")
print(f"  Weekly variation: {((weekly_avg[peak_day] - weekly_avg[low_day])/weekly_avg.mean()*100):.1f}%")
print()

# Seasonal recommendations
summer_avg = df[df['season'] == 'Summer']['Total_Liters'].mean()
winter_avg = df[df['season'] == 'Winter']['Total_Liters'].mean()

print("Seasonal Planning:")
print(f"  Summer average: {summer_avg:.2f} L (+{((summer_avg - df['Total_Liters'].mean())/df['Total_Liters'].mean()*100):.1f}% vs annual)")
print(f"  Winter average: {winter_avg:.2f} L ({((winter_avg - df['Total_Liters'].mean())/df['Total_Liters'].mean()*100):+.1f}% vs annual)")
print()

# Recommendations
print("💡 Key Recommendations:")
print(f"  1. Increase reserves by 15% on {dow_names[peak_day]}s")
print(f"  2. Schedule maintenance on {dow_names[low_day]}s")
print(f"  3. Prepare for {((summer_avg - winter_avg)/winter_avg*100):.0f}% increase in summer")
print(f"  4. Monitor for anomalies exceeding {df['Total_Liters'].mean() + 2*df['Total_Liters'].std():.0f} L")
print()


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - SUMMARY")
print("=" * 80)
print(f"✓ Analyzed {len(df)} days of water consumption data")
print(f"✓ Created {len(df.columns)} features using advanced engineering")
print(f"✓ Trained {len(selected_models)} machine learning models")
print(f"✓ Best model: {best_model} with R² = {test_metrics['test_r2']:.4f}")
print(f"✓ Prediction accuracy: ±{test_metrics['test_mae']:.2f} liters (MAPE: {test_metrics['test_mape']:.2f}%)")
print(f"✓ Detected {len(anomalies) if 'is_anomaly' in df.columns else 0} anomalies")
print()
print("Next Steps:")
print("  1. Launch dashboard: streamlit run ../dashboards/app.py")
print("  2. Start API: python ../deployment/app.py")
print("  3. Review model outputs in ../data/predictions.csv")
print("  4. Integrate with existing systems using REST API")
print("=" * 80)
