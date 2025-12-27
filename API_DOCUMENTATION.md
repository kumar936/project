# 🔌 API Documentation

Complete REST API reference for Water Demand Forecasting System

---

## Base URL

```
http://localhost:5000
```

For production: `https://your-domain.com/api`

---

## Authentication

Currently, no authentication is required. For production deployment, implement:
- API keys
- OAuth 2.0
- JWT tokens

---

## Endpoints

### 1. Health Check

Check if the API is running and models are loaded.

**Endpoint:** `GET /health`

**Request:**
```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

---

### 2. Model Information

Get details about the trained model.

**Endpoint:** `GET /model_info`

**Request:**
```bash
curl http://localhost:5000/model_info
```

**Response:**
```json
{
  "best_model": "xgboost",
  "training_date": "2024-01-14T15:30:00",
  "test_metrics": {
    "test_rmse": 49.35,
    "test_mae": 37.82,
    "test_r2": 0.8856,
    "test_mape": 6.82
  },
  "features_count": 47
}
```

---

### 3. Single Prediction

Make a prediction for a single date.

**Endpoint:** `POST /predict`

**Required Fields:**
- `date` (string): Date in YYYY-MM-DD format
- `temperature` (float): Temperature in °C
- `rainfall` (float): Rainfall in mm
- `humidity` (float): Humidity percentage (0-100)

**Optional Fields:**
- `Bathroom_Liters` (float): Historical bathroom usage
- `Kitchen_Liters` (float): Historical kitchen usage
- `Laundry_Liters` (float): Historical laundry usage
- `Gardening_Liters` (float): Historical gardening usage
- `is_weekend` (int): 0 or 1
- `is_public_holiday` (int): 0 or 1
- `Household_ID` (int): Household identifier

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-01-15",
    "temperature": 28.5,
    "rainfall": 2.3,
    "humidity": 65.0,
    "is_weekend": 0,
    "is_public_holiday": 0
  }'
```

**Response:**
```json
{
  "prediction": 687.45,
  "confidence_interval": {
    "lower": 591.23,
    "upper": 783.67,
    "confidence_level": 0.95
  },
  "input_data": {
    "date": "2024-01-15",
    "temperature": 28.5,
    "rainfall": 2.3,
    "humidity": 65.0,
    "is_weekend": 0,
    "is_public_holiday": 0
  },
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

---

### 4. Batch Predictions

Make predictions for multiple dates at once.

**Endpoint:** `POST /predict_batch`

**Request:**
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "date": "2024-01-15",
        "temperature": 28.5,
        "rainfall": 2.3,
        "humidity": 65.0
      },
      {
        "date": "2024-01-16",
        "temperature": 29.0,
        "rainfall": 0.0,
        "humidity": 62.0
      },
      {
        "date": "2024-01-17",
        "temperature": 27.5,
        "rainfall": 5.5,
        "humidity": 70.0
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "date": "2024-01-15",
      "prediction": 687.45,
      "input": {
        "date": "2024-01-15",
        "temperature": 28.5,
        "rainfall": 2.3,
        "humidity": 65.0
      }
    },
    {
      "date": "2024-01-16",
      "prediction": 705.32,
      "input": {
        "date": "2024-01-16",
        "temperature": 29.0,
        "rainfall": 0.0,
        "humidity": 62.0
      }
    },
    {
      "date": "2024-01-17",
      "prediction": 645.18,
      "input": {
        "date": "2024-01-17",
        "temperature": 27.5,
        "rainfall": 5.5,
        "humidity": 70.0
      }
    }
  ],
  "count": 3,
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

---

### 5. Multi-Day Forecast

Generate a forecast for multiple consecutive days.

**Endpoint:** `POST /forecast`

**Required Fields:**
- `start_date` (string): Starting date in YYYY-MM-DD format
- `days` (int): Number of days to forecast (1-90)

**Optional Fields:**
- `weather_forecast` (array): Array of weather predictions for each day

**Request (with weather forecast):**
```bash
curl -X POST http://localhost:5000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2024-01-15",
    "days": 7,
    "weather_forecast": [
      {"date": "2024-01-15", "temperature": 28.5, "rainfall": 0, "humidity": 65},
      {"date": "2024-01-16", "temperature": 29.0, "rainfall": 0, "humidity": 62},
      {"date": "2024-01-17", "temperature": 27.5, "rainfall": 5.5, "humidity": 70},
      {"date": "2024-01-18", "temperature": 26.0, "rainfall": 10.0, "humidity": 75},
      {"date": "2024-01-19", "temperature": 25.5, "rainfall": 3.0, "humidity": 72},
      {"date": "2024-01-20", "temperature": 27.0, "rainfall": 0, "humidity": 68},
      {"date": "2024-01-21", "temperature": 28.0, "rainfall": 0, "humidity": 64}
    ]
  }'
```

**Request (without weather forecast - uses defaults):**
```bash
curl -X POST http://localhost:5000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2024-01-15",
    "days": 7
  }'
```

**Response:**
```json
{
  "forecast": [
    {
      "date": "2024-01-15",
      "predicted_consumption": 687.45,
      "weather": {
        "date": "2024-01-15",
        "temperature": 28.5,
        "rainfall": 0,
        "humidity": 65.0,
        "is_weekend": false,
        "is_public_holiday": 0
      }
    },
    {
      "date": "2024-01-16",
      "predicted_consumption": 705.32,
      "weather": {
        "date": "2024-01-16",
        "temperature": 29.0,
        "rainfall": 0,
        "humidity": 62.0,
        "is_weekend": false,
        "is_public_holiday": 0
      }
    }
    // ... more days
  ],
  "summary": {
    "total_predicted": 4823.45,
    "average_daily": 689.06,
    "peak_day": "2024-01-16",
    "peak_consumption": 705.32
  },
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

---

## Error Responses

All endpoints return appropriate HTTP status codes and error messages.

### 400 Bad Request
Missing or invalid input parameters.

```json
{
  "error": "Missing required fields: ['temperature', 'humidity']"
}
```

### 500 Internal Server Error
Server-side error during prediction.

```json
{
  "error": "Model prediction failed: Invalid feature dimensions"
}
```

---

## Python Client Examples

### Example 1: Single Prediction

```python
import requests
import json

url = "http://localhost:5000/predict"

payload = {
    "date": "2024-01-15",
    "temperature": 28.5,
    "rainfall": 2.3,
    "humidity": 65.0,
    "is_weekend": 0
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Predicted consumption: {result['prediction']:.2f} liters")
print(f"95% CI: [{result['confidence_interval']['lower']:.2f}, "
      f"{result['confidence_interval']['upper']:.2f}]")
```

**Output:**
```
Predicted consumption: 687.45 liters
95% CI: [591.23, 783.67]
```

---

### Example 2: Weekly Forecast

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

url = "http://localhost:5000/forecast"

# Generate weather forecast for next 7 days
start_date = datetime.now()
weather_forecast = []

for i in range(7):
    date = start_date + timedelta(days=i)
    weather_forecast.append({
        "date": date.strftime("%Y-%m-%d"),
        "temperature": 25.0 + i * 0.5,  # Gradual temperature increase
        "rainfall": 0 if i % 2 == 0 else 3.0,  # Rain every other day
        "humidity": 65.0 - i * 1.0
    })

payload = {
    "start_date": start_date.strftime("%Y-%m-%d"),
    "days": 7,
    "weather_forecast": weather_forecast
}

response = requests.post(url, json=payload)
result = response.json()

# Convert to DataFrame
df = pd.DataFrame(result['forecast'])
print("\nWeekly Forecast:")
print(df[['date', 'predicted_consumption']])

print(f"\nSummary:")
print(f"Total weekly demand: {result['summary']['total_predicted']:.2f} liters")
print(f"Average daily: {result['summary']['average_daily']:.2f} liters")
print(f"Peak day: {result['summary']['peak_day']}")
```

**Output:**
```
Weekly Forecast:
         date  predicted_consumption
0  2024-01-15                 687.45
1  2024-01-16                 705.32
2  2024-01-17                 645.18
3  2024-01-18                 658.92
4  2024-01-19                 672.45
5  2024-01-20                 695.78
6  2024-01-21                 710.23

Summary:
Total weekly demand: 4775.33 liters
Average daily: 682.19 liters
Peak day: 2024-01-21
```

---

### Example 3: Batch Processing

```python
import requests
import pandas as pd

url = "http://localhost:5000/predict_batch"

# Load data from CSV
df = pd.read_csv('future_weather.csv')

# Prepare batch payload
data_list = df.to_dict('records')
payload = {"data": data_list}

# Make batch prediction
response = requests.post(url, json=payload)
result = response.json()

# Process results
predictions_df = pd.DataFrame(result['predictions'])
predictions_df['date'] = pd.to_datetime(predictions_df['date'])

print(f"Processed {result['count']} predictions")
print("\nFirst 10 predictions:")
print(predictions_df.head(10))

# Save results
predictions_df.to_csv('predictions_output.csv', index=False)
```

---

## JavaScript Client Example

```javascript
// Single prediction
async function predictWaterDemand(date, temperature, rainfall, humidity) {
  const url = 'http://localhost:5000/predict';
  
  const payload = {
    date: date,
    temperature: temperature,
    rainfall: rainfall,
    humidity: humidity,
    is_weekend: new Date(date).getDay() >= 5 ? 1 : 0
  };
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    
    const result = await response.json();
    
    console.log(`Predicted consumption: ${result.prediction.toFixed(2)} liters`);
    console.log(`Confidence interval: [${result.confidence_interval.lower.toFixed(2)}, ${result.confidence_interval.upper.toFixed(2)}]`);
    
    return result;
  } catch (error) {
    console.error('Prediction failed:', error);
  }
}

// Usage
predictWaterDemand('2024-01-15', 28.5, 2.3, 65.0);
```

---

## Rate Limiting

**Current:** No rate limiting

**Recommended for Production:**
- 100 requests per minute per IP
- 1000 requests per hour per API key

---

## Response Time

- Single prediction: <50ms
- Batch (100 records): <500ms
- Forecast (30 days): <200ms

---

## Best Practices

1. **Use batch predictions** for multiple dates to reduce API calls
2. **Cache results** for identical inputs
3. **Handle errors gracefully** with retry logic
4. **Validate inputs** before sending to API
5. **Monitor response times** and adjust request patterns

---

## Integration Examples

### Cron Job for Daily Forecasts

```bash
#!/bin/bash
# daily_forecast.sh

DATE=$(date +%Y-%m-%d)
DAYS=7

curl -X POST http://localhost:5000/forecast \
  -H "Content-Type: application/json" \
  -d "{\"start_date\": \"$DATE\", \"days\": $DAYS}" \
  > /var/log/water_forecast_$(date +%Y%m%d).json
```

Schedule with cron:
```bash
0 6 * * * /path/to/daily_forecast.sh
```

---

### Integration with Excel/Power BI

Use Power Query to fetch predictions:

```m
let
    url = "http://localhost:5000/forecast",
    body = "{""start_date"": ""2024-01-15"", ""days"": 30}",
    response = Json.Document(Web.Contents(url, [
        Headers=[#"Content-Type"="application/json"],
        Content=Text.ToBinary(body)
    ])),
    forecast = response[forecast],
    table = Table.FromList(forecast, Splitter.SplitByNothing(), null, null, ExtraValues.Error)
in
    table
```

---

## Support & Contact

- **Issues**: Review error logs in `/var/log/`
- **Documentation**: See `README.md`
- **Updates**: Check GitHub repository for latest version

---

**API Version:** 1.0  
**Last Updated:** January 2024
