"""
Model Training Pipeline
Complete training workflow for water demand forecasting models
"""

import pandas as pd
import numpy as np
import sys
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================
# SAFE PROJECT PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "water_consumption.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

from data_processing import WaterDataProcessor
from models import WaterDemandModels


def main():
    print("=" * 80)
    print("WATER DEMAND FORECASTING - MODEL TRAINING PIPELINE")
    print("=" * 80)
    print()

    # 1. Load and Process Data
    print("STEP 1: Loading and Processing Data")
    print("-" * 80)

    processor = WaterDataProcessor()
    df = processor.load_data(DATA_FILE)

    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    # 2. Feature Engineering
    print("STEP 2: Feature Engineering")
    print("-" * 80)

    df = processor.prepare_features(df, for_training=True)

    print(f"Total features created: {len(df.columns)}")
    print(f"Records after processing: {len(df)}")
    print()

    # 3. Data Splitting
    print("STEP 3: Data Splitting")
    print("-" * 80)

    train_df, val_df, test_df = processor.split_data(df, test_size=0.2, val_size=0.1)

    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print()

    # 4. Prepare Features
    print("STEP 4: Preparing Features")
    print("-" * 80)

    models = WaterDemandModels()
    models.initialize_models()

    X_train, y_train = models.prepare_data(train_df)
    X_val, y_val = models.prepare_data(val_df)
    X_test, y_test = models.prepare_data(test_df)

    print(f"Feature dimensions: {X_train.shape}")
    print()

    # 5. Scaling
    print("STEP 5: Feature Scaling")
    print("-" * 80)

    X_train_s, X_val_s, X_test_s = models.scale_features(
        X_train, X_val, X_test
    )
    print("Scaling completed")
    print()

    # 6. Train Models
    print("STEP 6: Training Models")
    print("-" * 80)

    results_df = models.train_all_models(X_train_s, y_train, X_val_s, y_val)

    print(results_df[['train_rmse', 'val_rmse', 'val_r2']])
    print()

    # 7. Ensemble
    print("STEP 7: Creating Ensemble")
    print("-" * 80)

    models.create_ensemble(X_train_s, y_train, X_val_s, y_val)
    print()

    # 8. Final Evaluation
    print("STEP 8: Final Evaluation")
    print("-" * 80)

    best_model = results_df.nsmallest(1, 'val_rmse').index[0]
    print(f"Best model: {best_model}")

    test_best = models.evaluate_on_test(best_model, X_test_s, y_test)
    test_ens = models.evaluate_on_test('ensemble', X_test_s, y_test)

    print("Best Model Test RMSE:", test_best['test_rmse'])
    print("Ensemble Test RMSE:", test_ens['test_rmse'])
    print()

    # 9. Save Models
    print("STEP 9: Saving Models")
    print("-" * 80)

    models.save_model(best_model, os.path.join(MODELS_DIR, f"best_{best_model}.pkl"))
    models.save_model('ensemble', os.path.join(MODELS_DIR, "ensemble.pkl"))

    joblib.dump(processor, os.path.join(MODELS_DIR, "data_processor.pkl"))

    metadata = {
        "trained_on": datetime.now().isoformat(),
        "best_model": best_model,
        "test_metrics": test_best,
        "ensemble_metrics": test_ens
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))

    print("Models saved successfully")
    print()

    # 10. Predictions
    print("STEP 10: Generating Predictions")
    print("-" * 80)

    X_full, y_full = models.prepare_data(df)
    X_full_s = models.scalers['standard'].transform(X_full)

    df["prediction_best"] = models.predict(best_model, X_full_s)
    df["prediction_ensemble"] = models.predict("ensemble", X_full_s)

    output_file = os.path.join(OUTPUTS_DIR, "predictions.csv")
    df.to_csv(output_file, index=False)

    print("Predictions saved to:", output_file)
    print()

    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
