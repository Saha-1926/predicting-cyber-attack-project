"""
Auto-Retraining Module for Cybersecurity Attack Prediction
Features: Continuous learning, model drift detection, automated retraining, and performance monitoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import os
import threading
import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelPerformanceMonitor:
    """Monitor model performance and detect degradation"""
    
    def __init__(self, performance_threshold=0.05):
        self.performance_threshold = performance_threshold
        self.performance_history = {}
        self.baseline_metrics = {}
        self.drift_alerts = []
        
    def set_baseline_performance(self, model_name, metrics):
        """Set baseline performance metrics for a model"""
        self.baseline_metrics[model_name] = {
            'metrics': metrics,
            'timestamp': datetime.now(),
            'data_points': 0
        }
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
    
    def log_performance(self, model_name, metrics, data_size):
        """Log current model performance"""
        performance_entry = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'data_size': data_size
        }
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append(performance_entry)
        
        # Check for performance drift
        if model_name in self.baseline_metrics:
            drift_detected = self._detect_performance_drift(model_name, metrics)
            if drift_detected:
                self._create_drift_alert(model_name, metrics)
    
    def _detect_performance_drift(self, model_name, current_metrics):
        """Detect if model performance has drifted significantly"""
        try:
            baseline = self.baseline_metrics[model_name]['metrics']
            
            # Check key metrics for drift
            key_metrics = ['accuracy', 'f1_score', 'r2_score', 'rmse']
            
            for metric in key_metrics:
                if metric in baseline and metric in current_metrics:
                    baseline_value = baseline[metric]
                    current_value = current_metrics[metric]
                    
                    # Calculate relative change
                    if baseline_value != 0:
                        relative_change = abs(current_value - baseline_value) / abs(baseline_value)
                        
                        # For error metrics (lower is better), invert the logic
                        if metric in ['rmse', 'mae']:
                            if current_value > baseline_value * (1 + self.performance_threshold):
                                return True
                        else:
                            # For performance metrics (higher is better)
                            if current_value < baseline_value * (1 - self.performance_threshold):
                                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error detecting performance drift: {e}")
            return False
    
    def _create_drift_alert(self, model_name, current_metrics):
        """Create alert for performance drift"""
        alert = {
            'model_name': model_name,
            'timestamp': datetime.now(),
            'alert_type': 'performance_drift',
            'current_metrics': current_metrics,
            'baseline_metrics': self.baseline_metrics[model_name]['metrics'],
            'message': f"Performance drift detected for {model_name}"
        }
        
        self.drift_alerts.append(alert)
        logging.warning(f"Performance drift alert: {alert['message']}")
    
    def get_drift_alerts(self, hours_back=24):
        """Get recent drift alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_alerts = [
            alert for alert in self.drift_alerts 
            if alert['timestamp'] > cutoff_time
        ]
        return recent_alerts
    
    def generate_performance_report(self, model_name):
        """Generate comprehensive performance report"""
        if model_name not in self.performance_history:
            return {}
        
        history = self.performance_history[model_name]
        
        # Calculate trends
        recent_performance = history[-10:] if len(history) >= 10 else history
        
        report = {
            'model_name': model_name,
            'total_evaluations': len(history),
            'baseline_set': model_name in self.baseline_metrics,
            'recent_trend': self._calculate_performance_trend(recent_performance),
            'stability_score': self._calculate_stability_score(history),
            'last_evaluation': history[-1] if history else None
        }
        
        return report
    
    def _calculate_performance_trend(self, recent_performance):
        """Calculate performance trend (improving/declining/stable)"""
        if len(recent_performance) < 2:
            return "insufficient_data"
        
        # Use the primary metric (accuracy or f1_score for classification, r2 for regression)
        metric_values = []
        for entry in recent_performance:
            metrics = entry['metrics']
            if 'accuracy' in metrics:
                metric_values.append(metrics['accuracy'])
            elif 'f1_score' in metrics:
                metric_values.append(metrics['f1_score'])
            elif 'r2_score' in metrics:
                metric_values.append(metrics['r2_score'])
        
        if len(metric_values) < 2:
            return "insufficient_data"
        
        # Calculate trend
        slope = np.polyfit(range(len(metric_values)), metric_values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _calculate_stability_score(self, history):
        """Calculate model stability score (0-1, higher is more stable)"""
        if len(history) < 3:
            return 1.0
        
        # Get primary metric values
        metric_values = []
        for entry in history:
            metrics = entry['metrics']
            if 'accuracy' in metrics:
                metric_values.append(metrics['accuracy'])
            elif 'f1_score' in metrics:
                metric_values.append(metrics['f1_score'])
            elif 'r2_score' in metrics:
                metric_values.append(metrics['r2_score'])
        
        if len(metric_values) < 3:
            return 1.0
        
        # Calculate coefficient of variation (lower = more stable)
        cv = np.std(metric_values) / np.mean(metric_values) if np.mean(metric_values) != 0 else 0
        
        # Convert to stability score (0-1)
        stability_score = max(0, 1 - cv * 2)  # Scale so that CV of 0.5 gives stability of 0
        
        return min(1.0, stability_score)

class AutoRetrainingOrchestrator:
    """Orchestrate automatic model retraining"""
    
    def __init__(self, data_buffer_size=1000, retrain_interval_hours=24):
        self.data_buffer_size = data_buffer_size
        self.retrain_interval_hours = retrain_interval_hours
        self.new_data_buffer = []
        self.models = {}
        self.model_metadata = {}
        self.performance_monitor = ModelPerformanceMonitor()
        self.retrain_scheduler = None
        self.is_running = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.model_configs = {
            'attack_type_classifier': {
                'type': 'classification',
                'target_column': 'Attack Type',
                'features': ['Country', 'Target Industry', 'Attack Source', 
                           'Security Vulnerability Type', 'Defense Mechanism Used', 
                           'Financial Loss (in Million $)', 'Number of Affected Users'],
                'model_class': RandomForestClassifier,
                'model_params': {'n_estimators': 100, 'random_state': 42}
            },
            'financial_loss_predictor': {
                'type': 'regression',
                'target_column': 'Financial Loss (in Million $)',
                'features': ['Country', 'Attack Type', 'Target Industry', 'Attack Source',
                           'Security Vulnerability Type', 'Defense Mechanism Used',
                           'Number of Affected Users'],
                'model_class': RandomForestRegressor,
                'model_params': {'n_estimators': 100, 'random_state': 42}
            },
            'resolution_time_predictor': {
                'type': 'regression',
                'target_column': 'Incident Resolution Time (in Hours)',
                'features': ['Country', 'Attack Type', 'Target Industry', 'Attack Source',
                           'Security Vulnerability Type', 'Defense Mechanism Used',
                           'Financial Loss (in Million $)', 'Number of Affected Users'],
                'model_class': RandomForestRegressor,
                'model_params': {'n_estimators': 100, 'random_state': 42}
            }
        }
    
    def add_new_data(self, data_batch):
        """Add new data to the buffer for retraining"""
        try:
            if isinstance(data_batch, pd.DataFrame):
                new_records = data_batch.to_dict('records')
            else:
                new_records = data_batch if isinstance(data_batch, list) else [data_batch]
            
            self.new_data_buffer.extend(new_records)
            
            # Maintain buffer size
            if len(self.new_data_buffer) > self.data_buffer_size:
                self.new_data_buffer = self.new_data_buffer[-self.data_buffer_size:]
            
            self.logger.info(f"Added {len(new_records)} new data points. Buffer size: {len(self.new_data_buffer)}")
            
            # Check if immediate retraining is needed
            if self._should_trigger_immediate_retrain():
                self.trigger_retraining()
                
        except Exception as e:
            self.logger.error(f"Error adding new data: {e}")
    
    def _should_trigger_immediate_retrain(self):
        """Determine if immediate retraining should be triggered"""
        # Trigger if buffer is full or if recent drift alerts
        if len(self.new_data_buffer) >= self.data_buffer_size:
            return True
        
        # Check for recent drift alerts
        recent_alerts = self.performance_monitor.get_drift_alerts(hours_back=1)
        if len(recent_alerts) > 0:
            return True
        
        return False
    
    def trigger_retraining(self):
        """Trigger model retraining"""
        if len(self.new_data_buffer) < 50:  # Minimum data requirement
            self.logger.warning("Insufficient data for retraining")
            return
        
        self.logger.info("Starting automatic model retraining...")
        
        try:
            # Convert buffer to DataFrame
            new_df = pd.DataFrame(self.new_data_buffer)
            
            # Retrain each model
            for model_name, config in self.model_configs.items():
                self._retrain_single_model(model_name, config, new_df)
            
            # Clear buffer after successful retraining
            self.new_data_buffer = []
            self.logger.info("Automatic retraining completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during retraining: {e}")
    
    def _retrain_single_model(self, model_name, config, data):
        """Retrain a single model"""
        try:
            # Check if we have the required columns
            required_cols = config['features'] + [config['target_column']]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns for {model_name}: {missing_cols}")
                return
            
            # Prepare data
            X, y, encoders, scaler = self._prepare_training_data(data, config)
            
            if len(X) < 10:  # Minimum samples
                self.logger.warning(f"Insufficient samples for {model_name}: {len(X)}")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if config['type'] == 'classification' else None
            )
            
            # Handle class imbalance for classification
            if config['type'] == 'classification' and len(np.unique(y_train)) > 1:
                try:
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                except ValueError:
                    self.logger.warning(f"SMOTE failed for {model_name}, using original data")
            
            # Train model
            model = config['model_class'](**config['model_params'])
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred, config['type'])
            
            # Store model and metadata
            self.models[model_name] = {
                'model': model,
                'encoders': encoders,
                'scaler': scaler,
                'config': config,
                'last_trained': datetime.now(),
                'training_size': len(X),
                'metrics': metrics
            }
            
            # Log performance
            self.performance_monitor.log_performance(model_name, metrics, len(X))
            
            # Set as baseline if first training
            if model_name not in self.performance_monitor.baseline_metrics:
                self.performance_monitor.set_baseline_performance(model_name, metrics)
            
            self.logger.info(f"Successfully retrained {model_name}. Metrics: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Error retraining {model_name}: {e}")
    
    def _prepare_training_data(self, data, config):
        """Prepare training data with encoding and scaling"""
        # Create a copy
        df = data.copy()
        
        # Handle missing values
        df = df.dropna(subset=[config['target_column']])
        
        # Prepare features
        X_raw = df[config['features']].copy()
        y = df[config['target_column']].copy()
        
        # Encode categorical variables
        encoders = {}
        for col in X_raw.columns:
            if X_raw[col].dtype == 'object':
                encoder = LabelEncoder()
                X_raw[col] = encoder.fit_transform(X_raw[col].astype(str))
                encoders[col] = encoder
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        # Encode target for classification
        if config['type'] == 'classification':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            encoders['target'] = target_encoder
        
        return X_scaled, y, encoders, scaler
    
    def _calculate_metrics(self, y_true, y_pred, task_type):
        """Calculate appropriate metrics based on task type"""
        metrics = {}
        
        try:
            if task_type == 'classification':
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:  # regression
                metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['r2_score'] = r2_score(y_true, y_pred)
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    def predict(self, model_name, input_data):
        """Make prediction using a trained model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model_info = self.models[model_name]
            model = model_info['model']
            encoders = model_info['encoders']
            scaler = model_info['scaler']
            config = model_info['config']
            
            # Prepare input data
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = pd.DataFrame(input_data)
            
            # Select features
            X_raw = input_df[config['features']].copy()
            
            # Apply encoders
            for col in X_raw.columns:
                if col in encoders:
                    # Handle unseen categories
                    try:
                        X_raw[col] = encoders[col].transform(X_raw[col].astype(str))
                    except ValueError:
                        # Use most frequent class for unseen categories
                        most_frequent = encoders[col].classes_[0]
                        X_raw[col] = X_raw[col].apply(
                            lambda x: x if x in encoders[col].classes_ else most_frequent
                        )
                        X_raw[col] = encoders[col].transform(X_raw[col].astype(str))
            
            # Scale features
            X_scaled = scaler.transform(X_raw)
            
            # Make prediction
            prediction = model.predict(X_scaled)
            
            # Decode prediction if classification
            if config['type'] == 'classification' and 'target' in encoders:
                prediction = encoders['target'].inverse_transform(prediction)
            
            return prediction[0] if len(prediction) == 1 else prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction with {model_name}: {e}")
            return None
    
    def start_scheduled_retraining(self):
        """Start scheduled retraining"""
        if self.is_running:
            self.logger.warning("Scheduled retraining already running")
            return
        
        self.is_running = True
        self.retrain_scheduler = threading.Thread(target=self._scheduled_retrain_loop)
        self.retrain_scheduler.daemon = True
        self.retrain_scheduler.start()
        self.logger.info(f"Started scheduled retraining every {self.retrain_interval_hours} hours")
    
    def stop_scheduled_retraining(self):
        """Stop scheduled retraining"""
        self.is_running = False
        if self.retrain_scheduler:
            self.retrain_scheduler.join(timeout=5)
        self.logger.info("Stopped scheduled retraining")
    
    def _scheduled_retrain_loop(self):
        """Scheduled retraining loop"""
        while self.is_running:
            try:
                time.sleep(self.retrain_interval_hours * 3600)  # Convert hours to seconds
                if self.is_running and len(self.new_data_buffer) >= 10:
                    self.trigger_retraining()
            except Exception as e:
                self.logger.error(f"Error in scheduled retraining loop: {e}")
    
    def export_model(self, model_name, filepath):
        """Export trained model to file"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.models[model_name], f)
            
            self.logger.info(f"Model {model_name} exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
    
    def import_model(self, model_name, filepath):
        """Import trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_info = pickle.load(f)
            
            self.models[model_name] = model_info
            self.logger.info(f"Model {model_name} imported from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error importing model: {e}")
    
    def get_system_status(self):
        """Get system status and statistics"""
        status = {
            'is_running': self.is_running,
            'buffer_size': len(self.new_data_buffer),
            'buffer_capacity': self.data_buffer_size,
            'trained_models': list(self.models.keys()),
            'recent_alerts': self.performance_monitor.get_drift_alerts(hours_back=24),
            'last_retrain': None
        }
        
        # Get last retrain time
        if self.models:
            last_times = [info['last_trained'] for info in self.models.values()]
            status['last_retrain'] = max(last_times).isoformat()
        
        return status

# Streamlit integration functions
def create_auto_retrain_dashboard(orchestrator):
    """Create auto-retraining dashboard for Streamlit"""
    
    st.markdown("## 🔄 Auto-Retraining System")
    
    # System status
    status = orchestrator.get_system_status()
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_icon = "🟢" if status['is_running'] else "🔴"
        st.metric("System Status", f"{status_icon} {'Active' if status['is_running'] else 'Inactive'}")
    
    with col2:
        buffer_pct = (status['buffer_size'] / status['buffer_capacity']) * 100
        st.metric("Data Buffer", f"{status['buffer_size']}/{status['buffer_capacity']}", f"{buffer_pct:.1f}%")
    
    with col3:
        st.metric("Trained Models", len(status['trained_models']))
    
    with col4:
        alert_count = len(status['recent_alerts'])
        st.metric("Recent Alerts", alert_count, "🚨" if alert_count > 0 else "✅")
    
    # Control panel
    st.markdown("### Control Panel")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start Auto-Retrain"):
            orchestrator.start_scheduled_retraining()
            st.success("Auto-retraining started")
    
    with col2:
        if st.button("⏹️ Stop Auto-Retrain"):
            orchestrator.stop_scheduled_retraining()
            st.warning("Auto-retraining stopped")
    
    with col3:
        if st.button("🔄 Trigger Retrain Now"):
            if len(orchestrator.new_data_buffer) >= 10:
                orchestrator.trigger_retraining()
                st.success("Retraining triggered")
            else:
                st.error("Insufficient data for retraining")
    
    with col4:
        if st.button("🧹 Clear Buffer"):
            orchestrator.new_data_buffer = []
            st.success("Data buffer cleared")
    
    # Model performance overview
    st.markdown("### 📊 Model Performance Overview")
    
    if status['trained_models']:
        model_tabs = st.tabs(status['trained_models'])
        
        for i, model_name in enumerate(status['trained_models']):
            with model_tabs[i]:
                if model_name in orchestrator.models:
                    model_info = orchestrator.models[model_name]
                    
                    # Model details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {model_info['config']['type'].title()}")
                        st.write(f"**Last Trained:** {model_info['last_trained'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Training Size:** {model_info['training_size']:,} samples")
                    
                    with col2:
                        metrics = model_info['metrics']
                        st.write("**Current Metrics:**")
                        for metric, value in metrics.items():
                            st.write(f"• {metric.replace('_', ' ').title()}: {value:.4f}")
                    
                    # Performance report
                    perf_report = orchestrator.performance_monitor.generate_performance_report(model_name)
                    
                    if perf_report:
                        st.markdown("#### Performance Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Evaluations", perf_report['total_evaluations'])
                        with col2:
                            trend = perf_report['recent_trend']
                            trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(trend, "❓")
                            st.metric("Trend", f"{trend_icon} {trend.title()}")
                        with col3:
                            stability = perf_report['stability_score']
                            st.metric("Stability", f"{stability:.2f}")
                    
                    # Performance history chart
                    if model_name in orchestrator.performance_monitor.performance_history:
                        history = orchestrator.performance_monitor.performance_history[model_name]
                        
                        if len(history) > 1:
                            # Create performance timeline
                            timestamps = [entry['timestamp'] for entry in history]
                            
                            fig = go.Figure()
                            
                            # Plot primary metric
                            primary_metric = None
                            if model_info['config']['type'] == 'classification':
                                primary_metric = 'accuracy' if 'accuracy' in history[0]['metrics'] else 'f1_score'
                            else:
                                primary_metric = 'r2_score'
                            
                            if primary_metric:
                                values = [entry['metrics'].get(primary_metric, 0) for entry in history]
                                
                                fig.add_trace(go.Scatter(
                                    x=timestamps,
                                    y=values,
                                    mode='lines+markers',
                                    name=primary_metric.replace('_', ' ').title(),
                                    line=dict(width=2)
                                ))
                            
                            fig.update_layout(
                                title=f"{model_name} Performance History",
                                xaxis_title="Time",
                                yaxis_title="Score",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No models trained yet")
    
    # Recent alerts
    st.markdown("### 🚨 Recent Performance Alerts")
    
    if status['recent_alerts']:
        for alert in status['recent_alerts'][-5:]:  # Show last 5 alerts
            with st.expander(f"⚠️ {alert['alert_type']} - {alert['model_name']}"):
                st.write(f"**Time:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Message:** {alert['message']}")
                
                if 'current_metrics' in alert and 'baseline_metrics' in alert:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Current Metrics:**")
                        for metric, value in alert['current_metrics'].items():
                            st.write(f"• {metric}: {value:.4f}")
                    
                    with col2:
                        st.write("**Baseline Metrics:**")
                        for metric, value in alert['baseline_metrics'].items():
                            st.write(f"• {metric}: {value:.4f}")
    else:
        st.info("No recent alerts")
    
    # Data upload for testing
    st.markdown("### 📥 Add New Training Data")
    
    uploaded_file = st.file_uploader("Upload CSV with new data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.write("**Preview of uploaded data:**")
            st.dataframe(new_data.head())
            
            if st.button("Add to Training Buffer"):
                orchestrator.add_new_data(new_data)
                st.success(f"Added {len(new_data)} records to training buffer")
                
        except Exception as e:
            st.error(f"Error uploading data: {e}")

if __name__ == "__main__":
    # Example usage
    orchestrator = AutoRetrainingOrchestrator()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Country': ['USA', 'China', 'Russia'] * 20,
        'Attack Type': ['DDoS', 'Phishing', 'Ransomware'] * 20,
        'Target Industry': ['Banking', 'Healthcare', 'Tech'] * 20,
        'Attack Source': ['Hacker Group', 'Nation State', 'Insider'] * 20,
        'Security Vulnerability Type': ['Unpatched Software', 'Weak Passwords', 'Social Engineering'] * 20,
        'Defense Mechanism Used': ['Firewall', 'Antivirus', 'AI-based Detection'] * 20,
        'Financial Loss (in Million $)': np.random.exponential(50, 60),
        'Number of Affected Users': np.random.exponential(100000, 60),
        'Incident Resolution Time (in Hours)': np.random.exponential(24, 60)
    })
    
    # Add sample data and trigger training
    orchestrator.add_new_data(sample_data)
    print("Auto-retraining system initialized with sample data")
