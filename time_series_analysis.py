"""
Time-Series Analysis Module for Cybersecurity Attack Prediction
Features: Trend analysis, seasonal patterns, forecasting, and anomaly detection in time series
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Some advanced time series features will be limited.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Facebook Prophet forecasting will be disabled.")

class CyberAttackTimeSeriesAnalyzer:
    """Advanced time-series analysis for cybersecurity attack patterns"""
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.forecasts = {}
        self.scalers = {}
        self.seasonal_components = {}
        
    def load_and_prepare_data(self, df, date_column='Year', target_columns=None):
        """Load and prepare time series data"""
        try:
            self.data = df.copy()
            
            # Convert date column to datetime if needed
            if date_column in df.columns:
                if df[date_column].dtype == 'object':
                    self.data[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                elif df[date_column].dtype in ['int64', 'float64']:
                    # Assume it's a year
                    self.data[date_column] = pd.to_datetime(df[date_column], format='%Y')
            
            # Set date as index
            self.data.set_index(date_column, inplace=True)
            self.data.sort_index(inplace=True)
            
            # Define target columns for analysis
            if target_columns is None:
                numeric_columns = self.data.select_dtypes(include=[np.number]).columns
                self.target_columns = [col for col in numeric_columns 
                                     if col not in ['Year', 'Unnamed: 0']]
            else:
                self.target_columns = target_columns
            
            return True
            
        except Exception as e:
            print(f"Error preparing time series data: {e}")
            return False
    
    def create_time_series_aggregations(self, groupby_cols=['Country', 'Attack Type'], 
                                      agg_level='yearly'):
        """Create various time series aggregations"""
        try:
            results = {}
            
            # Overall time series (all attacks aggregated)
            if agg_level == 'yearly':
                overall_ts = self.data.groupby(self.data.index.year).agg({
                    'Financial Loss (in Million $)': ['sum', 'mean', 'count'],
                    'Number of Affected Users': ['sum', 'mean'],
                    'Incident Resolution Time (in Hours)': 'mean'
                }).round(2)
            else:
                overall_ts = self.data.resample('M').agg({
                    'Financial Loss (in Million $)': ['sum', 'mean', 'count'],
                    'Number of Affected Users': ['sum', 'mean'],
                    'Incident Resolution Time (in Hours)': 'mean'
                }).round(2)
            
            results['overall'] = overall_ts
            
            # Country-wise time series
            if 'Country' in self.data.columns:
                country_ts = {}
                top_countries = self.data['Country'].value_counts().head(10).index
                
                for country in top_countries:
                    country_data = self.data[self.data['Country'] == country]
                    if len(country_data) > 1:
                        if agg_level == 'yearly':
                            ts = country_data.groupby(country_data.index.year).agg({
                                'Financial Loss (in Million $)': 'sum',
                                'Number of Affected Users': 'sum'
                            }).round(2)
                        else:
                            ts = country_data.resample('M').agg({
                                'Financial Loss (in Million $)': 'sum',
                                'Number of Affected Users': 'sum'
                            }).round(2)
                        country_ts[country] = ts
                
                results['country'] = country_ts
            
            # Attack type time series
            if 'Attack Type' in self.data.columns:
                attack_ts = {}
                attack_types = self.data['Attack Type'].value_counts().head(8).index
                
                for attack_type in attack_types:
                    attack_data = self.data[self.data['Attack Type'] == attack_type]
                    if len(attack_data) > 1:
                        if agg_level == 'yearly':
                            ts = attack_data.groupby(attack_data.index.year).agg({
                                'Financial Loss (in Million $)': 'sum',
                                'Number of Affected Users': 'sum'
                            }).round(2)
                        else:
                            ts = attack_data.resample('M').agg({
                                'Financial Loss (in Million $)': 'sum',
                                'Number of Affected Users': 'sum'
                            }).round(2)
                        attack_ts[attack_type] = ts
                
                results['attack_type'] = attack_ts
            
            return results
            
        except Exception as e:
            print(f"Error creating time series aggregations: {e}")
            return {}
    
    def perform_seasonal_decomposition(self, ts_data, column, period=12):
        """Perform seasonal decomposition on time series"""
        try:
            if not STATSMODELS_AVAILABLE:
                return None
            
            # Ensure we have enough data points
            if len(ts_data) < 2 * period:
                print(f"Not enough data for seasonal decomposition (need at least {2*period} points)")
                return None
            
            # Handle missing values
            ts_clean = ts_data[column].fillna(method='forward').fillna(method='backward')
            
            # Perform decomposition
            decomposition = seasonal_decompose(ts_clean, model='additive', period=period)
            
            self.seasonal_components[f"{column}_decomposition"] = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }
            
            return decomposition
            
        except Exception as e:
            print(f"Error in seasonal decomposition: {e}")
            return None
    
    def forecast_with_arima(self, ts_data, column, steps=12, order=(1,1,1)):
        """Forecast using ARIMA model"""
        try:
            if not STATSMODELS_AVAILABLE:
                return None, None
            
            # Prepare data
            ts_clean = ts_data[column].fillna(method='forward').fillna(method='backward')
            
            # Check stationarity
            result = adfuller(ts_clean.dropna())
            is_stationary = result[1] < 0.05
            
            # Fit ARIMA model
            model = ARIMA(ts_clean, order=order)
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast = fitted_model.forecast(steps=steps)
            forecast_index = pd.date_range(
                start=ts_clean.index[-1] + pd.DateOffset(years=1),
                periods=steps,
                freq='YS'
            )
            
            forecast_series = pd.Series(forecast, index=forecast_index)
            
            # Store model
            self.models[f"{column}_arima"] = fitted_model
            self.forecasts[f"{column}_arima"] = forecast_series
            
            return fitted_model, forecast_series
            
        except Exception as e:
            print(f"Error in ARIMA forecasting: {e}")
            return None, None
    
    def forecast_with_exponential_smoothing(self, ts_data, column, steps=12):
        """Forecast using Exponential Smoothing"""
        try:
            if not STATSMODELS_AVAILABLE:
                return None, None
            
            # Prepare data
            ts_clean = ts_data[column].fillna(method='forward').fillna(method='backward')
            
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(
                ts_clean,
                trend='add',
                seasonal='add' if len(ts_clean) >= 24 else None,
                seasonal_periods=12 if len(ts_clean) >= 24 else None
            )
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast = fitted_model.forecast(steps=steps)
            forecast_index = pd.date_range(
                start=ts_clean.index[-1] + pd.DateOffset(years=1),
                periods=steps,
                freq='YS'
            )
            
            forecast_series = pd.Series(forecast, index=forecast_index)
            
            # Store model
            self.models[f"{column}_exp_smooth"] = fitted_model
            self.forecasts[f"{column}_exp_smooth"] = forecast_series
            
            return fitted_model, forecast_series
            
        except Exception as e:
            print(f"Error in Exponential Smoothing forecasting: {e}")
            return None, None
    
    def forecast_with_prophet(self, ts_data, column, steps=12):
        """Forecast using Facebook Prophet"""
        try:
            if not PROPHET_AVAILABLE:
                return None, None
            
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': ts_data.index,
                'y': ts_data[column].fillna(ts_data[column].mean())
            })
            
            # Initialize and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_data)
            
            # Create future dates
            future = model.make_future_dataframe(periods=steps, freq='YS')
            forecast = model.predict(future)
            
            # Extract forecast series
            forecast_series = pd.Series(
                forecast['yhat'].iloc[-steps:].values,
                index=pd.date_range(
                    start=ts_data.index[-1] + pd.DateOffset(years=1),
                    periods=steps,
                    freq='YS'
                )
            )
            
            # Store model and forecast
            self.models[f"{column}_prophet"] = model
            self.forecasts[f"{column}_prophet"] = forecast_series
            
            return model, forecast_series
            
        except Exception as e:
            print(f"Error in Prophet forecasting: {e}")
            return None, None
    
    def detect_time_series_anomalies(self, ts_data, column, method='iqr', window=12):
        """Detect anomalies in time series data"""
        try:
            ts_clean = ts_data[column].fillna(method='forward').fillna(method='backward')
            anomalies = []
            
            if method == 'iqr':
                # IQR method
                Q1 = ts_clean.quantile(0.25)
                Q3 = ts_clean.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies = ts_clean[(ts_clean < lower_bound) | (ts_clean > upper_bound)]
            
            elif method == 'rolling_std':
                # Rolling standard deviation method
                rolling_mean = ts_clean.rolling(window=window).mean()
                rolling_std = ts_clean.rolling(window=window).std()
                
                upper_bound = rolling_mean + 2 * rolling_std
                lower_bound = rolling_mean - 2 * rolling_std
                
                anomalies = ts_clean[(ts_clean > upper_bound) | (ts_clean < lower_bound)]
            
            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                
                # Reshape data for sklearn
                X = ts_clean.values.reshape(-1, 1)
                
                # Fit Isolation Forest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = iso_forest.fit_predict(X)
                
                # Get anomalies
                anomaly_indices = ts_clean.index[anomaly_labels == -1]
                anomalies = ts_clean[anomaly_indices]
            
            return anomalies
            
        except Exception as e:
            print(f"Error detecting time series anomalies: {e}")
            return pd.Series()
    
    def calculate_trend_metrics(self, ts_data, column):
        """Calculate trend metrics for time series"""
        try:
            ts_clean = ts_data[column].fillna(method='forward').fillna(method='backward')
            
            # Calculate basic trend metrics
            start_value = ts_clean.iloc[0]
            end_value = ts_clean.iloc[-1]
            total_change = end_value - start_value
            percent_change = (total_change / start_value) * 100 if start_value != 0 else 0
            
            # Calculate year-over-year growth rates
            yoy_growth = ts_clean.pct_change() * 100
            avg_growth_rate = yoy_growth.mean()
            
            # Calculate volatility
            volatility = ts_clean.std()
            coefficient_of_variation = (volatility / ts_clean.mean()) * 100 if ts_clean.mean() != 0 else 0
            
            # Trend direction
            if avg_growth_rate > 5:
                trend_direction = "Strongly Increasing"
            elif avg_growth_rate > 0:
                trend_direction = "Increasing"
            elif avg_growth_rate > -5:
                trend_direction = "Stable"
            else:
                trend_direction = "Decreasing"
            
            return {
                'start_value': start_value,
                'end_value': end_value,
                'total_change': total_change,
                'percent_change': percent_change,
                'avg_growth_rate': avg_growth_rate,
                'volatility': volatility,
                'coefficient_of_variation': coefficient_of_variation,
                'trend_direction': trend_direction,
                'max_value': ts_clean.max(),
                'min_value': ts_clean.min(),
                'mean_value': ts_clean.mean()
            }
            
        except Exception as e:
            print(f"Error calculating trend metrics: {e}")
            return {}
    
    def create_forecast_visualization(self, ts_data, column, forecasts=None):
        """Create interactive forecast visualization"""
        try:
            fig = go.Figure()
            
            # Plot historical data
            fig.add_trace(go.Scatter(
                x=ts_data.index,
                y=ts_data[column],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='blue', width=2)
            ))
            
            # Plot forecasts
            if forecasts:
                colors = ['red', 'green', 'orange', 'purple']
                for i, (model_name, forecast_series) in enumerate(forecasts.items()):
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=forecast_series.index,
                        y=forecast_series.values,
                        mode='lines+markers',
                        name=f'{model_name} Forecast',
                        line=dict(color=color, width=2, dash='dash')
                    ))
            
            # Update layout
            fig.update_layout(
                title=f'Time Series Forecast: {column}',
                xaxis_title='Date',
                yaxis_title=column,
                hovermode='x unified',
                height=500
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating forecast visualization: {e}")
            return None
    
    def create_seasonal_decomposition_plot(self, decomposition, title="Seasonal Decomposition"):
        """Create seasonal decomposition visualization"""
        try:
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'],
                vertical_spacing=0.08
            )
            
            # Observed
            fig.add_trace(go.Scatter(
                x=decomposition.observed.index,
                y=decomposition.observed.values,
                mode='lines',
                name='Observed',
                line=dict(color='blue')
            ), row=1, col=1)
            
            # Trend
            fig.add_trace(go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend.values,
                mode='lines',
                name='Trend',
                line=dict(color='red')
            ), row=2, col=1)
            
            # Seasonal
            fig.add_trace(go.Scatter(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal.values,
                mode='lines',
                name='Seasonal',
                line=dict(color='green')
            ), row=3, col=1)
            
            # Residual
            fig.add_trace(go.Scatter(
                x=decomposition.resid.index,
                y=decomposition.resid.values,
                mode='lines',
                name='Residual',
                line=dict(color='orange')
            ), row=4, col=1)
            
            fig.update_layout(
                title=title,
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating seasonal decomposition plot: {e}")
            return None
    
    def generate_forecast_report(self, column, forecasts, metrics):
        """Generate comprehensive forecast report"""
        try:
            report = {
                'column': column,
                'forecast_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'historical_metrics': metrics,
                'forecasts': {},
                'model_comparison': {}
            }
            
            # Add forecast data
            for model_name, forecast_series in forecasts.items():
                report['forecasts'][model_name] = {
                    'values': forecast_series.tolist(),
                    'dates': forecast_series.index.strftime('%Y-%m-%d').tolist(),
                    'mean_forecast': forecast_series.mean(),
                    'total_forecast': forecast_series.sum()
                }
            
            # Model comparison (simplified)
            if len(forecasts) > 1:
                all_forecasts = pd.DataFrame(forecasts)
                report['model_comparison'] = {
                    'forecast_variance': all_forecasts.var(axis=1).mean(),
                    'forecast_agreement': all_forecasts.corr().mean().mean()
                }
            
            return report
            
        except Exception as e:
            print(f"Error generating forecast report: {e}")
            return {}

# Streamlit integration functions
def create_time_series_dashboard(analyzer, data):
    """Create time series analysis dashboard for Streamlit"""
    
    st.markdown("## 📈 Time Series Analysis & Forecasting")
    
    # Sidebar controls
    st.sidebar.markdown("### Time Series Controls")
    
    # Select analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Overall Trends", "Country Analysis", "Attack Type Analysis", "Forecasting"]
    )
    
    # Select metrics
    available_metrics = [
        'Financial Loss (in Million $)',
        'Number of Affected Users',
        'Incident Resolution Time (in Hours)'
    ]
    selected_metric = st.sidebar.selectbox("Select Metric", available_metrics)
    
    # Prepare time series data
    analyzer.load_and_prepare_data(data)
    ts_aggregations = analyzer.create_time_series_aggregations()
    
    if analysis_type == "Overall Trends":
        st.markdown("### 📊 Overall Attack Trends")
        
        if 'overall' in ts_aggregations:
            overall_data = ts_aggregations['overall']
            
            # Plot overall trends
            fig = go.Figure()
            
            if selected_metric in overall_data.columns:
                if isinstance(overall_data.columns, pd.MultiIndex):
                    # Handle MultiIndex columns
                    for col in overall_data.columns:
                        if col[0] == selected_metric:
                            fig.add_trace(go.Scatter(
                                x=overall_data.index,
                                y=overall_data[col],
                                mode='lines+markers',
                                name=f'{col[1]}',
                                line=dict(width=2)
                            ))
                else:
                    fig.add_trace(go.Scatter(
                        x=overall_data.index,
                        y=overall_data[selected_metric],
                        mode='lines+markers',
                        name=selected_metric,
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title=f"Overall Trends: {selected_metric}",
                xaxis_title="Year",
                yaxis_title=selected_metric,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display trend metrics
            if isinstance(overall_data.columns, pd.MultiIndex):
                target_col = (selected_metric, 'sum')
                if target_col in overall_data.columns:
                    metrics = analyzer.calculate_trend_metrics(overall_data, target_col)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Change", f"{metrics.get('total_change', 0):.2f}")
                    with col2:
                        st.metric("Percent Change", f"{metrics.get('percent_change', 0):.1f}%")
                    with col3:
                        st.metric("Avg Growth Rate", f"{metrics.get('avg_growth_rate', 0):.1f}%")
                    with col4:
                        st.metric("Trend Direction", metrics.get('trend_direction', 'Unknown'))
    
    elif analysis_type == "Country Analysis":
        st.markdown("### 🌍 Country-wise Time Series Analysis")
        
        if 'country' in ts_aggregations:
            country_data = ts_aggregations['country']
            
            # Select country
            available_countries = list(country_data.keys())
            selected_country = st.selectbox("Select Country", available_countries)
            
            if selected_country in country_data:
                country_ts = country_data[selected_country]
                
                # Plot country trends
                fig = go.Figure()
                
                if selected_metric in country_ts.columns:
                    fig.add_trace(go.Scatter(
                        x=country_ts.index,
                        y=country_ts[selected_metric],
                        mode='lines+markers',
                        name=f'{selected_country} - {selected_metric}',
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title=f"{selected_country}: {selected_metric} Over Time",
                    xaxis_title="Year",
                    yaxis_title=selected_metric,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly detection
                st.markdown("#### 🚨 Anomaly Detection")
                anomaly_method = st.selectbox(
                    "Anomaly Detection Method",
                    ["iqr", "rolling_std", "isolation_forest"]
                )
                
                anomalies = analyzer.detect_time_series_anomalies(
                    country_ts, selected_metric, method=anomaly_method
                )
                
                if not anomalies.empty:
                    st.write(f"**Detected {len(anomalies)} anomalies:**")
                    for date, value in anomalies.items():
                        st.write(f"• {date}: {value:.2f}")
                else:
                    st.info("No anomalies detected")
    
    elif analysis_type == "Attack Type Analysis":
        st.markdown("### 🎯 Attack Type Time Series Analysis")
        
        if 'attack_type' in ts_aggregations:
            attack_data = ts_aggregations['attack_type']
            
            # Select attack type
            available_attacks = list(attack_data.keys())
            selected_attack = st.selectbox("Select Attack Type", available_attacks)
            
            if selected_attack in attack_data:
                attack_ts = attack_data[selected_attack]
                
                # Plot attack type trends
                fig = go.Figure()
                
                if selected_metric in attack_ts.columns:
                    fig.add_trace(go.Scatter(
                        x=attack_ts.index,
                        y=attack_ts[selected_metric],
                        mode='lines+markers',
                        name=f'{selected_attack} - {selected_metric}',
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title=f"{selected_attack}: {selected_metric} Over Time",
                    xaxis_title="Year",
                    yaxis_title=selected_metric,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Forecasting":
        st.markdown("### 🔮 Attack Forecasting")
        
        # Forecasting controls
        forecast_steps = st.slider("Forecast Steps (Years)", 1, 10, 5)
        forecast_models = st.multiselect(
            "Select Forecasting Models",
            ["ARIMA", "Exponential Smoothing", "Prophet"],
            default=["ARIMA", "Exponential Smoothing"]
        )
        
        if st.button("Generate Forecasts"):
            if 'overall' in ts_aggregations:
                overall_data = ts_aggregations['overall']
                
                # Prepare data for forecasting
                if isinstance(overall_data.columns, pd.MultiIndex):
                    target_col = (selected_metric, 'sum')
                    if target_col in overall_data.columns:
                        ts_data = overall_data[[target_col]]
                        ts_data.columns = [selected_metric]
                    else:
                        st.error("Selected metric not available for forecasting")
                        return
                else:
                    ts_data = overall_data[[selected_metric]]
                
                forecasts = {}
                
                # Generate forecasts with selected models
                with st.spinner("Generating forecasts..."):
                    if "ARIMA" in forecast_models and STATSMODELS_AVAILABLE:
                        _, arima_forecast = analyzer.forecast_with_arima(
                            ts_data, selected_metric, steps=forecast_steps
                        )
                        if arima_forecast is not None:
                            forecasts['ARIMA'] = arima_forecast
                    
                    if "Exponential Smoothing" in forecast_models and STATSMODELS_AVAILABLE:
                        _, exp_forecast = analyzer.forecast_with_exponential_smoothing(
                            ts_data, selected_metric, steps=forecast_steps
                        )
                        if exp_forecast is not None:
                            forecasts['Exponential Smoothing'] = exp_forecast
                    
                    if "Prophet" in forecast_models and PROPHET_AVAILABLE:
                        _, prophet_forecast = analyzer.forecast_with_prophet(
                            ts_data, selected_metric, steps=forecast_steps
                        )
                        if prophet_forecast is not None:
                            forecasts['Prophet'] = prophet_forecast
                
                if forecasts:
                    # Create forecast visualization
                    fig = analyzer.create_forecast_visualization(
                        ts_data, selected_metric, forecasts
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display forecast values
                    st.markdown("#### 📋 Forecast Results")
                    forecast_df = pd.DataFrame(forecasts)
                    st.dataframe(forecast_df.round(2))
                    
                    # Forecast summary
                    st.markdown("#### 📊 Forecast Summary")
                    for model_name, forecast_series in forecasts.items():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{model_name} - Next Year", f"{forecast_series.iloc[0]:.2f}")
                        with col2:
                            st.metric(f"{model_name} - Average", f"{forecast_series.mean():.2f}")
                        with col3:
                            growth_rate = ((forecast_series.iloc[-1] - ts_data[selected_metric].iloc[-1]) / ts_data[selected_metric].iloc[-1]) * 100
                            st.metric(f"{model_name} - Growth Rate", f"{growth_rate:.1f}%")
                
                else:
                    st.warning("No forecasts could be generated. Please check if required libraries are installed.")

if __name__ == "__main__":
    # Example usage
    analyzer = CyberAttackTimeSeriesAnalyzer()
    
    # Create sample time series data
    dates = pd.date_range('2015-01-01', '2024-01-01', freq='YS')
    sample_data = pd.DataFrame({
        'Year': dates.year,
        'Financial Loss (in Million $)': np.random.exponential(50, len(dates)) + np.arange(len(dates)) * 10,
        'Number of Affected Users': np.random.exponential(100000, len(dates)) + np.arange(len(dates)) * 50000,
        'Country': np.random.choice(['USA', 'China', 'Russia', 'UK'], len(dates)),
        'Attack Type': np.random.choice(['DDoS', 'Phishing', 'Ransomware'], len(dates))
    })
    
    # Analyze
    analyzer.load_and_prepare_data(sample_data)
    ts_agg = analyzer.create_time_series_aggregations()
    print("Time series analysis completed")
