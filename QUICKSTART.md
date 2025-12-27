# 🚀 Quick Start Guide

Get your Water Demand Analytics system up and running in 5 minutes!

## Step 1: Installation (2 minutes)

```bash
# Navigate to project directory
cd water_demand_analytics

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, sklearn, xgboost, lightgbm, streamlit; print('✅ All packages installed successfully!')"
```

## Step 2: Train Models (3 minutes)

```bash
# Run the training pipeline
cd notebooks
python train_models.py
```

**What happens:**
- Loads 10,000+ records of water consumption data
- Creates 50+ engineered features
- Trains 7 different ML models
- Saves best model (typically XGBoost with R² > 0.88)

**Output:**
```
Training All Models
====================
Training xgboost...
  Train RMSE: 42.10
  Val RMSE: 48.92
  Val R²: 0.8891

🏆 Best Model: xgboost
Test RMSE: 49.35 liters
Test R²: 0.8856

Models saved in ../models/
```

## Step 3: Launch Dashboard

```bash
# Start the interactive dashboard
cd ../dashboards
streamlit run app.py
```

**Access:** Open browser to `http://localhost:8501`

## Step 4: Explore Features

### 📊 Dashboard Tab
- View current consumption metrics
- See recent trends (last 90 days)
- Analyze consumption by category
- Check weather correlations

### 🔮 Forecasting Tab
- Select forecast horizon (7-90 days)
- View predictions with confidence intervals
- Get forecast summaries
- Export predictions

### 📈 Trends & Patterns Tab
- Monthly seasonal patterns
- Weekly consumption cycles
- Holiday vs. regular day impact
- Category breakdowns over time

### 🔍 Anomaly Detection Tab
- View detected anomalies
- Analyze anomaly statistics
- Identify potential causes
- Review recent unusual patterns

### 🎯 What-If Scenarios Tab
- Simulate temperature changes
- Test rainfall impact
- Adjust humidity levels
- Get demand planning recommendations

### 📉 Model Performance Tab
- Compare all trained models
- View evaluation metrics
- Analyze feature importance
- Retrain models with new data

## Step 5: Make API Predictions (Optional)

```bash
# Start the REST API
cd ../deployment
python app.py
```

**Test the API:**

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-01-15",
    "temperature": 28.5,
    "rainfall": 2.3,
    "humidity": 65.0
  }'
```

**Response:**
```json
{
  "prediction": 687.45,
  "confidence_interval": {
    "lower": 591.23,
    "upper": 783.67
  }
}
```

---

## Common Use Cases

### 1. Daily Demand Forecast
**Goal:** Predict tomorrow's water consumption

**Steps:**
1. Go to Forecasting tab
2. Set horizon to 1 day
3. View prediction
4. Plan supply accordingly

### 2. Weekly Planning
**Goal:** Plan next week's supply schedule

**Steps:**
1. Go to Forecasting tab
2. Set horizon to 7 days
3. Note peak demand days
4. Schedule pumping operations

### 3. Scenario Analysis
**Goal:** Understand heat wave impact

**Steps:**
1. Go to What-If Scenarios
2. Increase temperature by +5°C
3. View predicted demand increase
4. Prepare contingency plans

### 4. Anomaly Investigation
**Goal:** Identify unusual consumption patterns

**Steps:**
1. Go to Anomaly Detection tab
2. Review detected anomalies
3. Check correlating weather events
4. Investigate potential causes (leaks, errors)

---

## Troubleshooting

### Issue: Package Installation Fails

**Solution:**
```bash
# Upgrade pip
pip install --upgrade pip

# Install packages one by one
pip install pandas numpy scikit-learn
pip install xgboost lightgbm
pip install plotly streamlit flask
```

### Issue: Model Training Error

**Solution:**
- Verify data file exists: `data/water_consumption.csv`
- Check data format matches expected structure
- Ensure sufficient RAM (2GB+)

### Issue: Dashboard Won't Load

**Solution:**
```bash
# Check Streamlit is installed
streamlit --version

# Run from correct directory
cd dashboards
streamlit run app.py

# Try different port
streamlit run app.py --server.port 8502
```

### Issue: API Returns Errors

**Solution:**
- Verify models are trained (check `models/` directory)
- Check Flask is running: `curl http://localhost:5000/health`
- Review API logs for specific error messages

---

## Next Steps

Once you're comfortable with the basics:

1. **Customize Features**: Edit `src/data_processing.py` to add domain-specific features
2. **Tune Models**: Adjust hyperparameters in `src/models.py`
3. **Deploy to Cloud**: Follow deployment guide in README.md
4. **Integrate with Systems**: Use REST API to connect with existing infrastructure
5. **Schedule Retraining**: Set up automated model updates with new data

---

## Support Resources

- 📖 **Full Documentation**: See `README.md`
- 💻 **Code Reference**: Check inline comments in source files
- 📊 **Example Output**: Review `data/predictions.csv`
- 🔧 **API Docs**: Visit `http://localhost:5000/` after starting API

---

## Quick Reference Commands

```bash
# Train models
cd notebooks && python train_models.py

# Start dashboard
cd dashboards && streamlit run app.py

# Start API
cd deployment && python app.py

# Run tests (if available)
cd tests && pytest

# Check model performance
python -c "import joblib; meta=joblib.load('models/metadata.pkl'); print(meta['test_metrics'])"
```

---

**🎉 Congratulations! You now have a fully functional water demand forecasting system!**

Start exploring the dashboard and making predictions! 💧
