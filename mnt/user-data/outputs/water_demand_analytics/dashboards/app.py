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

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

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
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and process data"""
    processor = WaterDataProcessor()
    df = processor.load_data('../data/water_consumption.csv')
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
    """Create consumption breakdown visualization"""
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
    """Create seasonal pattern analysis"""
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
    """Create weather vs consumption correlation plot"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Temperature', 'Rainfall', 'Humidity')
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(x=df['temperature'], y=df['Total_Liters'], mode='markers',
                  marker=dict(color='#1f77b4', opacity=0.5), name='Temperature'),
        row=1, col=1
    )
    
    # Rainfall
    fig.add_trace(
        go.Scatter(x=df['rainfall'], y=df['Total_Liters'], mode='markers',
                  marker=dict(color='#2ca02c', opacity=0.5), name='Rainfall'),
        row=1, col=2
    )
    
    # Humidity
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
    """Create anomaly detection visualization"""
    if 'is_anomaly' not in df.columns:
        st.warning("Anomaly detection not available")
        return None
    
    fig = go.Figure()
    
    # Normal data
    normal_df = df[df['is_anomaly'] == 0]
    fig.add_trace(go.Scatter(
        x=normal_df['date'],
        y=normal_df['Total_Liters'],
        name='Normal',
        mode='markers',
        marker=dict(color='#1f77b4', size=5)
    ))
    
    # Anomalies
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
    """Simulate what-if scenarios"""
    # Create a copy of recent data
    recent_data = df.tail(30).copy()
    
    # Apply scenario changes
    if 'temperature_change' in scenario_params:
        recent_data['temperature'] += scenario_params['temperature_change']
    
    if 'rainfall_change' in scenario_params:
        recent_data['rainfall'] += scenario_params['rainfall_change']
    
    if 'humidity_change' in scenario_params:
        recent_data['humidity'] += scenario_params['humidity_change']
    
    # Recalculate weather features
    recent_data['temp_humidity_interaction'] = recent_data['temperature'] * recent_data['humidity']
    recent_data['temp_squared'] = recent_data['temperature'] ** 2
    recent_data['heat_index'] = recent_data['temperature'] + 0.5 * recent_data['humidity']
    
    # Prepare features
    X_scenario, _ = models.prepare_data(recent_data)
    
    # Make predictions
    if 'standard' in models.scalers:
        X_scenario_scaled = models.scalers['standard'].transform(X_scenario)
        predictions = models.predict(model_name, pd.DataFrame(X_scenario_scaled, columns=X_scenario.columns))
    else:
        predictions = models.predict(model_name, X_scenario)
    
    return recent_data, predictions


def main():
    # Header
    st.markdown('<div class="main-header">💧 Water Demand Analytics & Forecasting</div>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Water Consumption Forecasting for Optimal Resource Management**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
                           ["📊 Dashboard", "🔮 Forecasting", "📈 Trends & Patterns", 
                            "🔍 Anomaly Detection", "🎯 What-If Scenarios", "📉 Model Performance"])
    
    # Load data
    with st.spinner("Loading data..."):
        df, processor = load_data()
    
    # Load models (in production)
    # models = load_models()
    
    # ==================== DASHBOARD PAGE ====================
    if page == "📊 Dashboard":
        st.markdown('<div class="section-header">Water Consumption Overview</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_daily = df['Total_Liters'].mean()
            st.metric("Average Daily Consumption", f"{avg_daily:.0f} L")
        
        with col2:
            max_consumption = df['Total_Liters'].max()
            st.metric("Peak Consumption", f"{max_consumption:.0f} L")
        
        with col3:
            total_consumption = df['Total_Liters'].sum() / 1000000  # Convert to millions
            st.metric("Total Consumption", f"{total_consumption:.2f} ML")
        
        with col4:
            anomaly_count = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
            st.metric("Detected Anomalies", f"{int(anomaly_count)}")
        
        # Recent trend
        st.markdown("### Recent Consumption Trend (Last 90 Days)")
        recent_df = df.tail(90)
        fig_recent = go.Figure()
        fig_recent.add_trace(go.Scatter(
            x=recent_df['date'],
            y=recent_df['Total_Liters'],
            mode='lines',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy'
        ))
        fig_recent.update_layout(
            xaxis_title='Date',
            yaxis_title='Consumption (Liters)',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_recent, use_container_width=True)
        
        # Consumption breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Consumption by Category")
            avg_consumption = {
                'Bathroom': df['Bathroom_Liters'].mean(),
                'Kitchen': df['Kitchen_Liters'].mean(),
                'Laundry': df['Laundry_Liters'].mean(),
                'Gardening': df['Gardening_Liters'].mean()
            }
            fig_pie = px.pie(
                values=list(avg_consumption.values()),
                names=list(avg_consumption.keys()),
                title='Average Daily Distribution'
            )
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
            
            # Weather trends
            fig_weather = create_weather_correlation(df.tail(500))
            st.plotly_chart(fig_weather, use_container_width=True)
    
    # ==================== FORECASTING PAGE ====================
    elif page == "🔮 Forecasting":
        st.markdown('<div class="section-header">Water Consumption Forecasting</div>', unsafe_allow_html=True)
        
        st.info("📌 **Note**: In this demo, we're showing visualizations. Full ML predictions require model training (see Model Performance tab).")
        
        # Forecast horizon selection
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30)
        
        # Show recent data with trend
        recent_df = df.tail(180)
        
        # Simple trend projection for demo
        from sklearn.linear_model import LinearRegression
        X_simple = np.arange(len(recent_df)).reshape(-1, 1)
        y_simple = recent_df['Total_Liters'].values
        simple_model = LinearRegression()
        simple_model.fit(X_simple, y_simple)
        
        # Create forecast
        future_X = np.arange(len(recent_df), len(recent_df) + forecast_days).reshape(-1, 1)
        future_pred = simple_model.predict(future_X)
        
        # Create dates
        last_date = recent_df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # Visualization
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=recent_df['date'],
            y=recent_df['Total_Liters'],
            name='Historical',
            mode='lines',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_pred,
            name='Forecast',
            mode='lines',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Confidence interval
        std_error = np.std(y_simple - simple_model.predict(X_simple))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_pred + 1.96 * std_error,
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_pred - 1.96 * std_error,
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            name='95% CI',
            fillcolor='rgba(255, 127, 14, 0.2)'
        ))
        
        fig.update_layout(
            title=f'{forecast_days}-Day Water Consumption Forecast',
            xaxis_title='Date',
            yaxis_title='Consumption (Liters)',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast statistics
        st.markdown("### Forecast Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Forecast", f"{future_pred.mean():.0f} L")
        with col2:
            st.metric("Peak Forecast", f"{future_pred.max():.0f} L")
        with col3:
            st.metric("Total Forecast", f"{future_pred.sum()/1000:.1f} kL")
    
    # ==================== TRENDS PAGE ====================
    elif page == "📈 Trends & Patterns":
        st.markdown('<div class="section-header">Consumption Trends & Patterns</div>', unsafe_allow_html=True)
        
        # Seasonal analysis
        st.markdown("### Seasonal Patterns")
        fig_seasonal = create_seasonal_analysis(df)
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Day of week analysis
        st.markdown("### Weekly Patterns")
        dow_avg = df.groupby('day_of_week')['Total_Liters'].mean().reset_index()
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_avg['day_name'] = dow_avg['day_of_week'].map(dict(enumerate(dow_names)))
        
        fig_dow = px.bar(dow_avg, x='day_name', y='Total_Liters',
                        title='Average Consumption by Day of Week',
                        labels={'Total_Liters': 'Average Consumption (L)', 'day_name': 'Day'})
        st.plotly_chart(fig_dow, use_container_width=True)
        
        # Category breakdown over time
        st.markdown("### Consumption Categories Over Time")
        fig_breakdown = create_consumption_breakdown(df.tail(365))
        st.plotly_chart(fig_breakdown, use_container_width=True)
        
        # Holiday vs Non-holiday
        st.markdown("### Holiday Impact")
        col1, col2 = st.columns(2)
        with col1:
            holiday_avg = df[df['is_public_holiday'] == 1]['Total_Liters'].mean()
            st.metric("Average on Holidays", f"{holiday_avg:.0f} L")
        with col2:
            regular_avg = df[df['is_public_holiday'] == 0]['Total_Liters'].mean()
            st.metric("Average on Regular Days", f"{regular_avg:.0f} L")
            
        pct_change = ((holiday_avg - regular_avg) / regular_avg) * 100
        st.info(f"💡 Holidays show a **{pct_change:+.1f}%** change in water consumption compared to regular days")
    
    # ==================== ANOMALY DETECTION PAGE ====================
    elif page == "🔍 Anomaly Detection":
        st.markdown('<div class="section-header">Anomaly Detection & Insights</div>', unsafe_allow_html=True)
        
        if 'is_anomaly' in df.columns:
            # Anomaly visualization
            fig_anomaly = create_anomaly_detection_plot(df)
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Anomaly statistics
            anomalies = df[df['is_anomaly'] == 1]
            st.markdown("### Anomaly Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Anomalies", f"{len(anomalies)}")
            with col2:
                pct_anomalies = (len(anomalies) / len(df)) * 100
                st.metric("Anomaly Rate", f"{pct_anomalies:.2f}%")
            with col3:
                st.metric("Avg Anomaly Value", f"{anomalies['Total_Liters'].mean():.0f} L")
            with col4:
                st.metric("Max Anomaly", f"{anomalies['Total_Liters'].max():.0f} L")
            
            # Recent anomalies table
            st.markdown("### Recent Anomalies")
            recent_anomalies = anomalies.tail(10)[['date', 'Total_Liters', 'temperature', 'rainfall', 'humidity', 'is_weekend', 'is_public_holiday']]
            st.dataframe(recent_anomalies, use_container_width=True)
            
            # Anomaly causes analysis
            st.markdown("### Potential Anomaly Drivers")
            col1, col2 = st.columns(2)
            
            with col1:
                temp_anom = anomalies['temperature'].mean()
                temp_normal = df[df['is_anomaly'] == 0]['temperature'].mean()
                st.metric("Temperature (Anomalies)", f"{temp_anom:.1f}°C", 
                         delta=f"{temp_anom - temp_normal:.1f}°C")
            
            with col2:
                rainfall_anom = anomalies['rainfall'].mean()
                rainfall_normal = df[df['is_anomaly'] == 0]['rainfall'].mean()
                st.metric("Rainfall (Anomalies)", f"{rainfall_anom:.1f} mm",
                         delta=f"{rainfall_anom - rainfall_normal:.1f} mm")
        else:
            st.warning("Anomaly detection not available in the current dataset")
    
    # ==================== WHAT-IF SCENARIOS PAGE ====================
    elif page == "🎯 What-If Scenarios":
        st.markdown('<div class="section-header">What-If Scenario Simulator</div>', unsafe_allow_html=True)
        
        st.info("🎯 **Simulate how changes in weather conditions affect water demand**")
        
        # Scenario inputs
        st.markdown("### Scenario Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temp_change = st.slider("Temperature Change (°C)", -10.0, 10.0, 0.0, 0.5)
        
        with col2:
            rainfall_change = st.slider("Rainfall Change (mm)", -10.0, 10.0, 0.0, 0.5)
        
        with col3:
            humidity_change = st.slider("Humidity Change (%)", -20.0, 20.0, 0.0, 1.0)
        
        # Calculate impact (simplified model)
        baseline_consumption = df['Total_Liters'].mean()
        
        # Simple linear relationships (can be replaced with actual model predictions)
        temp_impact = temp_change * 8  # ~8L per degree
        rainfall_impact = rainfall_change * -2  # negative correlation
        humidity_impact = humidity_change * 1.5
        
        predicted_change = temp_impact + rainfall_impact + humidity_impact
        predicted_consumption = baseline_consumption + predicted_change
        
        # Display results
        st.markdown("### Scenario Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Baseline Consumption", f"{baseline_consumption:.0f} L")
        
        with col2:
            st.metric("Predicted Consumption", f"{predicted_consumption:.0f} L",
                     delta=f"{predicted_change:+.0f} L")
        
        with col3:
            pct_change = (predicted_change / baseline_consumption) * 100
            st.metric("Percentage Change", f"{pct_change:+.1f}%")
        
        # Visualization
        st.markdown("### Impact Breakdown")
        
        impact_df = pd.DataFrame({
            'Factor': ['Temperature', 'Rainfall', 'Humidity', 'Total'],
            'Impact (L)': [temp_impact, rainfall_impact, humidity_impact, predicted_change]
        })
        
        fig_impact = px.bar(impact_df[:-1], x='Factor', y='Impact (L)',
                           title='Impact by Weather Factor',
                           color='Impact (L)',
                           color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig_impact, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Demand Planning Recommendations")
        
        if predicted_change > 50:
            st.warning(f"⚠️ **High Demand Alert**: Expected increase of {predicted_change:.0f}L ({pct_change:+.1f}%)")
            st.markdown("""
            **Recommended Actions:**
            - Increase water reserves by 15-20%
            - Activate auxiliary pumping stations
            - Issue conservation advisories
            - Monitor reservoir levels closely
            """)
        elif predicted_change < -50:
            st.info(f"ℹ️ **Lower Demand Expected**: Expected decrease of {abs(predicted_change):.0f}L ({pct_change:.1f}%)")
            st.markdown("""
            **Recommended Actions:**
            - Reduce pumping capacity
            - Schedule maintenance activities
            - Optimize energy consumption
            """)
        else:
            st.success("✅ **Normal Operations**: Consumption within expected range")
    
    # ==================== MODEL PERFORMANCE PAGE ====================
    elif page == "📉 Model Performance":
        st.markdown('<div class="section-header">Model Training & Performance</div>', unsafe_allow_html=True)
        
        st.info("🚀 **Click below to train models on your data**")
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training multiple regression models..."):
                # Initialize models
                models = WaterDemandModels()
                models.initialize_models()
                
                # Prepare data
                train_df, val_df, test_df = processor.split_data(df, test_size=0.2, val_size=0.1)
                
                X_train, y_train = models.prepare_data(train_df)
                X_val, y_val = models.prepare_data(val_df)
                X_test, y_test = models.prepare_data(test_df)
                
                # Scale features
                X_train_scaled, X_val_scaled, X_test_scaled = models.scale_features(X_train, X_val, X_test)
                
                # Train all models
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                model_names = list(models.models.keys())
                for i, model_name in enumerate(model_names):
                    status_text.text(f"Training {model_name}...")
                    models.train_model(model_name, X_train_scaled, y_train, X_val_scaled, y_val)
                    progress_bar.progress((i + 1) / len(model_names))
                
                status_text.text("Creating ensemble model...")
                models.create_ensemble(X_train_scaled, y_train, X_val_scaled, y_val)
                
                progress_bar.progress(1.0)
                status_text.text("Training complete!")
                
                # Display results
                results_df = pd.DataFrame(models.results).T
                st.markdown("### Model Comparison")
                st.dataframe(results_df.style.format("{:.2f}"), use_container_width=True)
                
                # Best model
                best_model = results_df.nsmallest(1, 'val_rmse').index[0]
                st.success(f"🏆 **Best Model**: {best_model}")
                
                # Feature importance
                st.markdown("### Feature Importance")
                importance_df = models.get_feature_importance(best_model if best_model != 'ensemble' else 'xgboost')
                
                if importance_df is not None:
                    fig_importance = px.bar(importance_df.head(15), x='importance', y='feature',
                                          orientation='h',
                                          title='Top 15 Most Important Features')
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # Save models
                st.markdown("### Save Trained Models")
                if st.button("Save Best Model"):
                    models.save_model(best_model, f'../models/{best_model}_model.pkl')
                    st.success(f"Model saved successfully!")
        
        else:
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
            
            Click **Train Models** to start the training process!
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>💧 Water Demand Analytics Dashboard | Powered by Machine Learning</p>
        <p>For optimal water resource management and demand planning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
