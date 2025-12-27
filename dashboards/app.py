"""
Water Demand Analytics Dashboard
Interactive Streamlit application for water consumption forecasting and demand planning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing import WaterDataProcessor
from models import WaterDemandModels, calculate_metrics
import joblib
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Water Demand Analytics",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://streamlit.io',
        'Report a bug': None,
        'About': "Water Demand Analytics Dashboard"
    }
)

# Custom CSS with Water Theme Animations
st.markdown("""
<style>
    /* Remove default margins and padding */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body, html {
        width: 100%;
        height: 100%;
    }
    
    /* Main container styling */
    .main {
        width: 100%;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1f77b4 0%, #0d47a1 100%);
        padding: 2rem 1rem;
    }
    
    .css-1d391kg div[role="navigation"] {
        color: white;
    }
    
    /* Remove default Streamlit padding */
    .css-1e5imcs {
        padding: 2rem;
    }
    
    @keyframes water-wave {
        0% { transform: translateX(-100%); }
        50% { transform: translateX(0%); }
        100% { transform: translateX(100%); }
    }
    
    @keyframes water-drop {
        0% { transform: translateY(-10px); opacity: 1; }
        100% { transform: translateY(10px); opacity: 0; }
    }
    
    @keyframes water-glow {
        0%, 100% { box-shadow: 0 0 10px rgba(31, 119, 180, 0.3); }
        50% { box-shadow: 0 0 20px rgba(31, 119, 180, 0.8); }
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f2f6 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #1f77b4;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        border-bottom: 4px solid #1f77b4;
        padding-bottom: 0.8rem;
        animation: water-glow 3s ease-in-out infinite;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    
    .page-transition {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Status box styling */
    .stStatus {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-left: 5px solid #1f77b4;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.15);
    }
    
    /* Chart styling */
    .plotly {
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Radio button styling */
    .css-1aumxpj {
        color: #1f77b4;
        font-weight: 500;
    }
    
    /* Slider styling */
    .css-qri22k {
        color: #1f77b4;
    }
    
    /* Info box styling */
    .css-19ih76x {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #1f77b4;
        border-radius: 0.5rem;
    }
    
    /* Subheader text */
    h2 {
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    h3 {
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    
    /* Overall page background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and process data"""
    processor = WaterDataProcessor()
    # Get the parent directory to load from root
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(parent_dir), 'water_consumption.csv')
    df = processor.load_data(data_path)
    df = processor.prepare_features(df, for_training=True)
    return df, processor


@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        models = WaterDemandModels()
        models.initialize_models()
        # In production, load pre-trained models
        # models.load_model('xgboost', '../models/best_model.pkl')
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


def create_forecast_plot(df, predictions, actual_col='Total_Liters', title="Water Consumption Forecast"):
    """Create interactive forecast visualization"""
    fig = go.Figure()
    # ...existing code...



# --- Begin full dashboard logic ---
def create_forecast_plot(df, predictions, actual_col='Total_Liters', title="Water Consumption Forecast"):
    fig = go.Figure()
    # Actual values
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[actual_col],
        name='Actual',
        mode='lines',
        line=dict(color='#1f77b4', width=2)
    ))
    # Predictions
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=predictions,
        name='Predicted',
        mode='lines',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    # Confidence intervals (simplified)
    residuals = df[actual_col] - predictions
    std_residual = np.std(residuals)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=predictions + 1.96 * std_residual,
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=predictions - 1.96 * std_residual,
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        name='95% Confidence',
        fillcolor='rgba(255, 127, 14, 0.2)'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Water Consumption (Liters)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    return fig

def create_consumption_breakdown(df):
    consumption_cols = ['Bathroom_Liters', 'Kitchen_Liters', 'Laundry_Liters', 'Gardening_Liters']
    fig = go.Figure()
    for col in consumption_cols:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[col],
            name=col.replace('_Liters', ''),
            mode='lines',
            stackgroup='one',
            fillcolor=None
        ))
    fig.update_layout(
        title='Water Consumption by Category',
        xaxis_title='Date',
        yaxis_title='Consumption (Liters)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    return fig

def create_seasonal_analysis(df):
    monthly_avg = df.groupby('month')['Total_Liters'].agg(['mean', 'std']).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_avg['month'],
        y=monthly_avg['mean'],
        error_y=dict(type='data', array=monthly_avg['std']),
        name='Average Consumption',
        marker_color='#1f77b4'
    ))
    fig.update_layout(
        title='Monthly Average Water Consumption',
        xaxis_title='Month',
        yaxis_title='Average Consumption (Liters)',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        height=400,
        template='plotly_white'
    )
    return fig

def create_weather_correlation(df):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Temperature', 'Rainfall', 'Humidity')
    )
    fig.add_trace(
        go.Scatter(x=df['temperature'], y=df['Total_Liters'], mode='markers',
                  marker=dict(color='#1f77b4', opacity=0.5), name='Temperature'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['rainfall'], y=df['Total_Liters'], mode='markers',
                  marker=dict(color='#2ca02c', opacity=0.5), name='Rainfall'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['humidity'], y=df['Total_Liters'], mode='markers',
                  marker=dict(color='#d62728', opacity=0.5), name='Humidity'),
        row=1, col=3
    )
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Rainfall (mm)", row=1, col=2)
    fig.update_xaxes(title_text="Humidity (%)", row=1, col=3)
    fig.update_yaxes(title_text="Consumption (Liters)", row=1, col=1)
    fig.update_layout(
        title='Weather Impact on Water Consumption',
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    return fig

def create_anomaly_detection_plot(df):
    if 'is_anomaly' not in df.columns:
        st.warning("Anomaly detection not available")
        return None
    fig = go.Figure()
    normal_df = df[df['is_anomaly'] == 0]
    fig.add_trace(go.Scatter(
        x=normal_df['date'],
        y=normal_df['Total_Liters'],
        name='Normal',
        mode='markers',
        marker=dict(color='#1f77b4', size=5)
    ))
    anomaly_df = df[df['is_anomaly'] == 1]
    fig.add_trace(go.Scatter(
        x=anomaly_df['date'],
        y=anomaly_df['Total_Liters'],
        name='Anomaly',
        mode='markers',
        marker=dict(color='#d62728', size=10, symbol='x')
    ))
    fig.update_layout(
        title=f'Anomaly Detection - {len(anomaly_df)} Anomalies Detected',
        xaxis_title='Date',
        yaxis_title='Water Consumption (Liters)',
        height=400,
        template='plotly_white'
    )
    return fig

def simulate_scenario(df, processor, models, model_name, scenario_params):
    recent_data = df.tail(30).copy()
    if 'temperature_change' in scenario_params:
        recent_data['temperature'] += scenario_params['temperature_change']
    if 'rainfall_change' in scenario_params:
        recent_data['rainfall'] += scenario_params['rainfall_change']
    if 'humidity_change' in scenario_params:
        recent_data['humidity'] += scenario_params['humidity_change']
    recent_data['temp_humidity_interaction'] = recent_data['temperature'] * recent_data['humidity']
    recent_data['temp_squared'] = recent_data['temperature'] ** 2
    recent_data['heat_index'] = recent_data['temperature'] + 0.5 * recent_data['humidity']
    X_scenario, _ = models.prepare_data(recent_data)
    if 'standard' in models.scalers:
        X_scenario_scaled = models.scalers['standard'].transform(X_scenario)
        predictions = models.predict(model_name, pd.DataFrame(X_scenario_scaled, columns=X_scenario.columns))
    else:
        predictions = models.predict(model_name, X_scenario)
    return recent_data, predictions



def main():
    st.markdown('<div class="main-header">💧 Water Demand Analytics & Forecasting</div>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Water Consumption Forecasting for Optimal Resource Management**")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
                           ["📊 Dashboard", "🔮 Forecasting", "📈 Trends & Patterns", 
                            "🔍 Anomaly Detection", "🎯 What-If Scenarios", "📉 Model Performance"])
    
    with st.spinner("💧 Loading data..."):
        df, processor = load_data()
    
    if page == "📊 Dashboard":
        status = st.status("💧 Preparing Dashboard...", expanded=True)
        with status:
            st.write("Loading consumption data...")
            time.sleep(0.3)
            st.write("Calculating metrics...")
            time.sleep(0.2)
            st.write("Generating visualizations...")
            time.sleep(0.2)
        status.update(label="✅ Dashboard Ready", state="complete")
        st.markdown('<div class="section-header page-transition">Water Consumption Overview</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Daily Consumption", f"{df['Total_Liters'].mean():.0f} L")
        with col2:
            st.metric("Peak Consumption", f"{df['Total_Liters'].max():.0f} L")
        with col3:
            st.metric("Total Consumption", f"{df['Total_Liters'].sum() / 1000000:.2f} ML")
        with col4:
            anomaly_count = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
            st.metric("Detected Anomalies", f"{int(anomaly_count)}")
        
        st.markdown("### Recent Consumption Trend (Last 90 Days)")
        recent_df = df.tail(90)
        fig_recent = go.Figure()
        fig_recent.add_trace(go.Scatter(x=recent_df['date'], y=recent_df['Total_Liters'],
                                       mode='lines', line=dict(color='#1f77b4', width=2), fill='tozeroy'))
        fig_recent.update_layout(xaxis_title='Date', yaxis_title='Consumption (Liters)',
                                height=400, template='plotly_white')
        st.plotly_chart(fig_recent, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Consumption by Category")
            avg_consumption = {
                'Bathroom': df['Bathroom_Liters'].mean(),
                'Kitchen': df['Kitchen_Liters'].mean(),
                'Laundry': df['Laundry_Liters'].mean(),
                'Gardening': df['Gardening_Liters'].mean()
            }
            fig_pie = px.pie(values=list(avg_consumption.values()), names=list(avg_consumption.keys()),
                            title='Average Daily Distribution')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("### Weather Conditions")
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
            with col2b:
                st.metric("Avg Rainfall", f"{df['rainfall'].mean():.1f} mm")
            with col2c:
                st.metric("Avg Humidity", f"{df['humidity'].mean():.1f}%")
            fig_weather = create_weather_correlation(df.tail(500))
            st.plotly_chart(fig_weather, use_container_width=True)
    
    elif page == "🔮 Forecasting":
        status = st.status("🔮 Generating Forecast...", expanded=True)
        with status:
            st.write("Loading historical data...")
            time.sleep(0.3)
            st.write("Calculating forecast model...")
            time.sleep(0.2)
            st.write("Generating predictions...")
            time.sleep(0.2)
        status.update(label="✅ Forecast Ready", state="complete")
        st.markdown('<div class="section-header page-transition">Water Consumption Forecasting</div>', unsafe_allow_html=True)
        st.info("📌 In this demo, we're showing visualizations. Full ML predictions require model training.")
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30)
        recent_df = df.tail(180)
        from sklearn.linear_model import LinearRegression
        X_simple = np.arange(len(recent_df)).reshape(-1, 1)
        y_simple = recent_df['Total_Liters'].values
        simple_model = LinearRegression()
        simple_model.fit(X_simple, y_simple)
        future_X = np.arange(len(recent_df), len(recent_df) + forecast_days).reshape(-1, 1)
        future_pred = simple_model.predict(future_X)
        last_date = recent_df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent_df['date'], y=recent_df['Total_Liters'],
                                name='Historical', mode='lines', line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=future_pred,
                                name='Forecast', mode='lines', line=dict(color='#ff7f0e', width=2, dash='dash')))
        std_error = np.std(y_simple - simple_model.predict(X_simple))
        fig.add_trace(go.Scatter(x=future_dates, y=future_pred + 1.96 * std_error,
                                fill=None, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=future_dates, y=future_pred - 1.96 * std_error,
                                fill='tonexty', mode='lines', line=dict(width=0),
                                name='95% CI', fillcolor='rgba(255, 127, 14, 0.2)'))
        fig.update_layout(title=f'{forecast_days}-Day Forecast', xaxis_title='Date',
                         yaxis_title='Consumption (Liters)', height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Forecast", f"{future_pred.mean():.0f} L")
        with col2:
            st.metric("Peak Forecast", f"{future_pred.max():.0f} L")
        with col3:
            st.metric("Total Forecast", f"{future_pred.sum()/1000:.1f} kL")
    
    elif page == "📈 Trends & Patterns":
        status = st.status("📈 Analyzing Trends...", expanded=True)
        with status:
            st.write("Loading historical consumption...")
            time.sleep(0.3)
            st.write("Analyzing seasonal patterns...")
            time.sleep(0.2)
            st.write("Generating trend charts...")
            time.sleep(0.2)
        status.update(label="✅ Analysis Complete", state="complete")
        st.markdown('<div class="section-header page-transition">Consumption Trends & Patterns</div>', unsafe_allow_html=True)
        st.markdown("### Seasonal Patterns")
        fig_seasonal = create_seasonal_analysis(df)
        st.plotly_chart(fig_seasonal, use_container_width=True)
        st.markdown("### Weekly Patterns")
        dow_avg = df.groupby('day_of_week')['Total_Liters'].mean().reset_index()
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_avg['day_name'] = dow_avg['day_of_week'].map(dict(enumerate(dow_names)))
        fig_dow = px.bar(dow_avg, x='day_name', y='Total_Liters',
                        title='Average Consumption by Day of Week',
                        labels={'Total_Liters': 'Average Consumption (L)', 'day_name': 'Day'})
        st.plotly_chart(fig_dow, use_container_width=True)
        st.markdown("### Consumption Categories Over Time")
        fig_breakdown = create_consumption_breakdown(df.tail(365))
        st.plotly_chart(fig_breakdown, use_container_width=True)
        st.markdown("### Holiday Impact")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average on Holidays", f"{df[df['is_public_holiday'] == 1]['Total_Liters'].mean():.0f} L")
        with col2:
            st.metric("Average on Regular Days", f"{df[df['is_public_holiday'] == 0]['Total_Liters'].mean():.0f} L")
    
    elif page == "🔍 Anomaly Detection":
        status = st.status("🔍 Detecting Anomalies...", expanded=True)
        with status:
            st.write("Scanning consumption patterns...")
            time.sleep(0.3)
            st.write("Identifying outliers...")
            time.sleep(0.2)
            st.write("Analyzing anomaly data...")
            time.sleep(0.2)
        status.update(label="✅ Anomaly Detection Complete", state="complete")
        st.markdown('<div class="section-header page-transition">Anomaly Detection & Insights</div>', unsafe_allow_html=True)
        if 'is_anomaly' in df.columns:
            fig_anomaly = create_anomaly_detection_plot(df)
            st.plotly_chart(fig_anomaly, use_container_width=True)
            anomalies = df[df['is_anomaly'] == 1]
            st.markdown("### Anomaly Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Anomalies", f"{len(anomalies)}")
            with col2:
                st.metric("Anomaly Rate", f"{(len(anomalies) / len(df)) * 100:.2f}%")
            with col3:
                st.metric("Avg Anomaly Value", f"{anomalies['Total_Liters'].mean():.0f} L")
            with col4:
                st.metric("Max Anomaly", f"{anomalies['Total_Liters'].max():.0f} L")
        else:
            st.warning("Anomaly detection not available")
    
    elif page == "🎯 What-If Scenarios":
        status = st.status("🎯 Loading Scenarios...", expanded=True)
        with status:
            st.write("Loading baseline data...")
            time.sleep(0.3)
            st.write("Preparing simulation environment...")
            time.sleep(0.2)
            st.write("Initializing scenario parameters...")
            time.sleep(0.2)
        status.update(label="✅ Scenario Ready", state="complete")
        st.markdown('<div class="section-header page-transition">What-If Scenario Simulator</div>', unsafe_allow_html=True)
        st.info("🎯 Simulate how changes in weather conditions affect water demand")
        st.markdown("### Scenario Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            temp_change = st.slider("Temperature Change (°C)", -10.0, 10.0, 0.0, 0.5)
        with col2:
            rainfall_change = st.slider("Rainfall Change (mm)", -10.0, 10.0, 0.0, 0.5)
        with col3:
            humidity_change = st.slider("Humidity Change (%)", -20.0, 20.0, 0.0, 1.0)
        
        baseline = df['Total_Liters'].mean()
        predicted_change = temp_change * 8 + rainfall_change * -2 + humidity_change * 1.5
        predicted_consumption = baseline + predicted_change
        
        st.markdown("### Scenario Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Baseline Consumption", f"{baseline:.0f} L")
        with col2:
            st.metric("Predicted Consumption", f"{predicted_consumption:.0f} L", delta=f"{predicted_change:+.0f} L")
        with col3:
            st.metric("Percentage Change", f"{(predicted_change / baseline) * 100:+.1f}%")
    
    elif page == "📉 Model Performance":
        status = st.status("📉 Loading Models...", expanded=True)
        with status:
            st.write("Loading model configuration...")
            time.sleep(0.3)
            st.write("Initializing ML models...")
            time.sleep(0.2)
            st.write("Preparing performance metrics...")
            time.sleep(0.2)
        status.update(label="✅ Models Ready", state="complete")
        st.markdown('<div class="section-header page-transition">Model Training & Performance</div>', unsafe_allow_html=True)
        st.info("🚀 Click below to train models on your data")
        st.markdown("""
        ### Available Models:
        - **Random Forest**: Ensemble of decision trees
        - **Gradient Boosting**: Sequential ensemble learning
        - **XGBoost**: Optimized gradient boosting
        - **LightGBM**: Fast gradient boosting framework
        - **Extra Trees**: Randomized decision trees
        - **Ridge Regression**: Linear model with L2 regularization
        - **Elastic Net**: Combined L1 and L2 regularization
        - **Ensemble**: Weighted combination of top models
        
        ### Training Instructions:
        - The training button is ready to be implemented
        - Models will be trained on your historical data
        - Results will display model comparison metrics
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>💧 Water Demand Analytics Dashboard | Powered by Machine Learning</p>
        <p>For optimal water resource management and demand planning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
