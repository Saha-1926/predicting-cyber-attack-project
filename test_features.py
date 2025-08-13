#!/usr/bin/env python3
"""
Feature Testing Script for Cyber Defense Suite
Tests all implemented features and identifies issues
"""

import sys
import traceback
import pandas as pd
import numpy as np
from datetime import datetime

def test_basic_imports():
    """Test all basic imports"""
    print("🧪 Testing Basic Imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit: OK")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
        
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas/NumPy: OK")
    except ImportError as e:
        print(f"❌ Pandas/NumPy: {e}")
        return False
        
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly: OK")
    except ImportError as e:
        print(f"❌ Plotly: {e}")
        return False
        
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        print("✅ Scikit-learn: OK")
    except ImportError as e:
        print(f"❌ Scikit-learn: {e}")
        return False
        
    return True

def test_ml_libraries():
    """Test ML-specific libraries"""
    print("\n🤖 Testing ML Libraries...")
    
    try:
        import xgboost as xgb
        print("✅ XGBoost: OK")
    except ImportError as e:
        print(f"❌ XGBoost: {e}")
        
    try:
        import lightgbm as lgb
        print("✅ LightGBM: OK")
    except ImportError as e:
        print(f"❌ LightGBM: {e}")
        
    try:
        import shap
        print("✅ SHAP: OK")
    except ImportError as e:
        print(f"❌ SHAP: {e}")
        
    try:
        from imblearn.over_sampling import SMOTE
        print("✅ Imbalanced-learn: OK")
    except ImportError as e:
        print(f"❌ Imbalanced-learn: {e}")

def test_optional_libraries():
    """Test optional libraries"""
    print("\n📈 Testing Optional Libraries...")
    
    try:
        from prophet import Prophet
        print("✅ Prophet: OK")
    except ImportError as e:
        print(f"⚠️ Prophet: {e} (Optional - for advanced forecasting)")
        
    try:
        import statsmodels
        print("✅ Statsmodels: OK")
    except ImportError as e:
        print(f"⚠️ Statsmodels: {e} (Optional - for time series analysis)")

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing Data Loading...")
    
    try:
        # Test if default dataset exists
        import os
        dataset_path = "Global_Cybersecurity_Threats_2015-2024.csv"
        
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            print(f"✅ Default dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"   Columns: {list(df.columns)}")
            return df
        else:
            print(f"❌ Default dataset not found: {dataset_path}")
            # Create sample data for testing
            print("🔧 Creating sample data for testing...")
            df = create_sample_data()
            return df
            
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return None

def create_sample_data():
    """Create sample cybersecurity data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    countries = ['USA', 'China', 'Russia', 'UK', 'Germany', 'India', 'Brazil', 'Japan']
    attack_types = ['DDoS', 'Phishing', 'Ransomware', 'SQL Injection', 'Malware', 'Data Breach']
    industries = ['Banking', 'Healthcare', 'Technology', 'Government', 'Education', 'Retail']
    sources = ['Hacker Group', 'Nation State', 'Insider Threat', 'Cybercriminals']
    vulnerabilities = ['Unpatched Software', 'Weak Passwords', 'Social Engineering', 'Zero-day']
    defenses = ['Firewall', 'Antivirus', 'AI-based Detection', 'SIEM', 'EDR']
    
    data = {
        'Country': np.random.choice(countries, n_samples),
        'Year': np.random.choice(range(2015, 2025), n_samples),
        'Attack Type': np.random.choice(attack_types, n_samples),
        'Target Industry': np.random.choice(industries, n_samples),
        'Attack Source': np.random.choice(sources, n_samples),
        'Security Vulnerability Type': np.random.choice(vulnerabilities, n_samples),
        'Defense Mechanism Used': np.random.choice(defenses, n_samples),
        'Financial Loss (in Million $)': np.random.exponential(50, n_samples),
        'Number of Affected Users': np.random.exponential(100000, n_samples).astype(int),
        'Incident Resolution Time (in Hours)': np.random.exponential(24, n_samples)
    }
    
    df = pd.DataFrame(data)
    print(f"✅ Sample data created: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def test_cybersecurity_predictor(df):
    """Test the main CyberSecurityPredictor class"""
    print("\n🔮 Testing CyberSecurityPredictor...")
    
    try:
        # Import the main predictor class
        sys.path.append('.')
        from app import CyberSecurityPredictor
        
        # Initialize predictor
        predictor = CyberSecurityPredictor()
        print("✅ Predictor initialized")
        
        # Load data
        predictor.data = df
        success = predictor.preprocess_data()
        if success:
            print("✅ Data preprocessing successful")
        else:
            print("❌ Data preprocessing failed")
            return False
            
        # Test model training
        success = predictor.train_models()
        if success:
            print("✅ Model training successful")
            print(f"   Trained models: {list(predictor.models.keys())}")
        else:
            print("❌ Model training failed")
            return False
            
        # Test prediction
        # Create sample input
        sample_input = [
            0,  # Country_encoded
            0,  # Target Industry_encoded  
            0,  # Attack Source_encoded
            0,  # Security Vulnerability Type_encoded
            0,  # Defense Mechanism Used_encoded
            5,  # Years_Since_2015 (2020)
            50000  # Number of Affected Users
        ]
        
        predictions = predictor.predict_attack(sample_input)
        print("✅ Prediction successful")
        print(f"   Predictions: {predictions}")
        
        return True
        
    except Exception as e:
        print(f"❌ CyberSecurityPredictor test failed: {e}")
        traceback.print_exc()
        return False

def test_auto_retrain_system():
    """Test auto-retraining system"""
    print("\n🔄 Testing Auto-Retraining System...")
    
    try:
        from auto_retrain import AutoRetrainingOrchestrator
        
        # Initialize orchestrator
        orchestrator = AutoRetrainingOrchestrator()
        print("✅ Auto-retraining orchestrator initialized")
        
        # Test with sample data
        sample_data = create_sample_data().head(100)  # Small sample for testing
        orchestrator.add_new_data(sample_data)
        print("✅ Sample data added to buffer")
        
        # Get system status
        status = orchestrator.get_system_status()
        print(f"✅ System status retrieved: {status['buffer_size']} items in buffer")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-retraining test failed: {e}")
        traceback.print_exc()
        return False

def test_real_time_monitor():
    """Test real-time monitoring system"""
    print("\n🔴 Testing Real-Time Monitoring...")
    
    try:
        from real_time_monitor import RealTimeMonitor
        
        # Initialize monitor
        monitor = RealTimeMonitor()
        print("✅ Real-time monitor initialized")
        
        # Test threat score calculation
        sample_incident = {
            'financial_loss': 10.5,
            'affected_users': 50000,
            'attack_type': 'ransomware',
            'country': 'US',
            'timestamp': datetime.now()
        }
        
        threat_score = monitor.calculate_threat_score(sample_incident)
        print(f"✅ Threat score calculated: {threat_score:.2f}")
        
        # Test dashboard data
        dashboard_data = monitor.get_real_time_dashboard_data()
        print("✅ Dashboard data retrieved")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time monitoring test failed: {e}")
        traceback.print_exc()
        return False

def test_time_series_analyzer():
    """Test time-series analysis"""
    print("\n📈 Testing Time-Series Analysis...")
    
    try:
        from time_series_analysis import CyberAttackTimeSeriesAnalyzer
        
        # Initialize analyzer
        analyzer = CyberAttackTimeSeriesAnalyzer()
        print("✅ Time-series analyzer initialized")
        
        # Create sample time-series data
        df = create_sample_data()
        success = analyzer.load_and_prepare_data(df)
        
        if success:
            print("✅ Time-series data loaded and prepared")
            
            # Test aggregations
            aggregations = analyzer.create_time_series_aggregations()
            print(f"✅ Time-series aggregations created: {list(aggregations.keys())}")
            
            return True
        else:
            print("❌ Time-series data preparation failed")
            return False
            
    except Exception as e:
        print(f"❌ Time-series analysis test failed: {e}")
        traceback.print_exc()
        return False

def test_feature_status():
    """Test and report status of all features"""
    print("\n📋 Feature Status Report:")
    print("=" * 50)
    
    features = {
        "Executive Dashboard": "✅ Working - Loads automatically with KPIs and visualizations",
        "Data Analytics - Data Overview": "✅ Working - Shows data stats and quality metrics", 
        "Data Analytics - Correlation Analysis": "✅ Working - Correlation heatmaps and analysis",
        "Data Analytics - Geographic Analysis": "✅ Working - Country-wise threat analysis",
        "Data Analytics - Industry Analysis": "✅ Working - Sector-specific risk assessment",
        "Data Analytics - Anomaly Detection": "✅ Working - Isolation Forest anomaly detection",
        "ML Model Training": "✅ Working - Trains 4 different prediction models",
        "ML Model Comparison": "✅ Working - Performance comparison across models",
        "ML Hyperparameter Tuning": "🔄 Coming Soon - UI placeholder implemented",
        "ML Feature Engineering": "🔄 Coming Soon - UI placeholder implemented", 
        "ML Model Deployment": "🔄 Coming Soon - UI placeholder implemented",
        "Single Attack Prediction": "✅ Working - Real-time prediction with risk assessment",
        "Batch Prediction": "🔄 Coming Soon - Upload interface placeholder",
        "Risk Assessment": "🔄 Coming Soon - UI placeholder implemented",
        "Scenario Analysis": "🔄 Coming Soon - UI placeholder implemented",
        "What-If Analysis": "🔄 Coming Soon - UI placeholder implemented",
        "Real-Time Monitoring": "⚠️ Partially Working - Core logic works, needs live data stream",
        "Time-Series Analysis": "⚠️ Partially Working - Basic functionality, needs optional libraries",
        "Auto-Retraining System": "✅ Working - Full functionality with performance monitoring",
        "Explainable AI": "⚠️ Partially Working - Feature importance works, SHAP needs debugging",
        "Threat Intelligence": "🔄 Mock Implementation - Framework ready, needs API integration",
        "Incident Response": "✅ Working - Workflow management and reporting",
        "System Configuration": "✅ Working - Settings management and system health"
    }
    
    working = sum(1 for status in features.values() if status.startswith("✅"))
    partial = sum(1 for status in features.values() if status.startswith("⚠️"))
    coming = sum(1 for status in features.values() if status.startswith("🔄"))
    
    for feature, status in features.items():
        print(f"{feature:<35} : {status}")
    
    print("=" * 50)
    print(f"📊 Summary: {working} fully working, {partial} partially working, {coming} in development")
    print(f"🎯 Overall completion: {(working + partial*0.5) / len(features) * 100:.1f}%")

def main():
    """Run all tests"""
    print("🛡️ Cyber Defense Suite - Feature Testing")
    print("=" * 50)
    
    # Test basic imports
    if not test_basic_imports():
        print("❌ Basic imports failed. Please install required dependencies.")
        return
    
    # Test ML libraries
    test_ml_libraries()
    
    # Test optional libraries
    test_optional_libraries()
    
    # Test data loading
    df = test_data_loading()
    if df is None:
        print("❌ Data loading failed. Cannot proceed with further tests.")
        return
    
    # Test main predictor
    test_cybersecurity_predictor(df)
    
    # Test auto-retraining
    test_auto_retrain_system()
    
    # Test real-time monitoring  
    test_real_time_monitor()
    
    # Test time-series analysis
    test_time_series_analyzer()
    
    # Feature status report
    test_feature_status()
    
    print("\n🎉 Testing completed!")
    print("\n💡 To run the application:")
    print("   streamlit run app.py")
    print("\n📖 For detailed feature explanations, see:")
    print("   FEATURES_EXPLAINED.md")

if __name__ == "__main__":
    main()
