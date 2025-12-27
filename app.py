"""
Flask API for Water Demand Forecasting
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from data_processing import WaterDataProcessor

app = Flask(__name__)
CORS(app)

MODEL = None
PROCESSOR = None
SCALER = None
FEATURE_COLUMNS = None


def load_models():
    global MODEL, PROCESSOR, SCALER, FEATURE_COLUMNS

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")

        # Load trained model
        MODEL = joblib.load(os.path.join(models_dir, "best_ridge.pkl"))

        # Load data processor
        PROCESSOR = joblib.load(os.path.join(models_dir, "data_processor.pkl"))

        # Build feature columns from training CSV
        train_df = pd.read_csv(os.path.join(base_dir, "water_consumption.csv"))
        train_df["date"] = pd.to_datetime(train_df["date"])

        train_df = PROCESSOR.prepare_features(train_df, for_training=False)

        FEATURE_COLUMNS = [
            c for c in train_df.columns
            if c not in ["Total_Liters", "date"]
            and train_df[c].dtype in ["int64", "float64"]
        ]

        # Fit scaler again (acceptable for inference)
        from sklearn.preprocessing import StandardScaler
        SCALER = StandardScaler()
        SCALER.fit(train_df[FEATURE_COLUMNS])

        print("✅ Model, processor, scaler loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False


@app.route("/")
def home():
    return jsonify({
        "message": "Water Demand Forecasting API",
        "status": "running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([data])
        df["date"] = pd.to_datetime(df["date"])

        # Defaults
        for col in [
            "Bathroom_Liters", "Kitchen_Liters",
            "Laundry_Liters", "Gardening_Liters",
            "is_weekend", "is_public_holiday"
        ]:
            if col not in df.columns:
                df[col] = 0

        df["Total_Liters"] = 0

        # Feature engineering
        df = PROCESSOR.prepare_features(df, for_training=False)

        # Ensure feature columns
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        X = df[FEATURE_COLUMNS]
        X_scaled = SCALER.transform(X)

        prediction = MODEL.predict(X_scaled)[0]

        return jsonify({
            "prediction": float(prediction),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if load_models():
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        print("❌ Failed to load models")
