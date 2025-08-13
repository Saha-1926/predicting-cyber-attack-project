#!/usr/bin/env python3
"""
Setup script for the Cybersecurity Attack Prediction System
This script installs required dependencies and launches the Streamlit application.
"""

import subprocess
import sys
import os
import importlib

def check_and_install_packages():
    """Check if required packages are installed and install them if not"""
    
    required_packages = [
        'streamlit==1.28.0',
        'pandas==2.0.3',
        'numpy==1.24.3',
        'scikit-learn==1.3.0',
        'matplotlib==3.7.2',
        'seaborn==0.12.2',
        'plotly==5.15.0',
        'xgboost==1.7.6',
        'lightgbm==4.0.0',
        'imbalanced-learn==0.11.0',
        'joblib==1.3.2'
    ]
    
    print("🔍 Checking required packages...")
    
    missing_packages = []
    
    # Check each package
    for package in required_packages:
        package_name = package.split('==')[0].replace('-', '_')
        try:
            importlib.import_module(package_name)
            print(f"✅ {package_name} is already installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package_name} not found")
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    else:
        print("🎉 All required packages are already installed!")
    
    return True

def check_data_file():
    """Check if the dataset file exists"""
    data_file = "Global_Cybersecurity_Threats_2015-2024.csv"
    
    if os.path.exists(data_file):
        print(f"✅ Dataset file '{data_file}' found")
        return True
    else:
        print(f"⚠️  Dataset file '{data_file}' not found in current directory")
        print("   The application will still work - you can upload your own dataset")
        return False

def launch_streamlit_app():
    """Launch the Streamlit application"""
    try:
        print("\n🚀 Launching Cybersecurity Attack Prediction System...")
        print("   The application will open in your default web browser")
        print("   URL: http://localhost:8501")
        print("\n   To stop the application, press Ctrl+C in this terminal")
        
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("   Try running: streamlit run app.py")

def main():
    """Main setup and launch function"""
    print("="*60)
    print("🛡️  CYBERSECURITY ATTACK PREDICTION SYSTEM")
    print("="*60)
    print("Setting up the environment and launching the application...")
    print()
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check required files
    required_files = ['app.py', 'data_utils.py', 'ml_models.py', 'requirements.txt']
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            missing_files.append(file)
            print(f"❌ {file} not found")
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        print("   Please make sure all files are in the same directory")
        return
    
    print()
    
    # Install packages
    if not check_and_install_packages():
        print("❌ Failed to install required packages")
        return
    
    print()
    
    # Check data file
    check_data_file()
    
    print()
    
    # Launch application
    launch_streamlit_app()

if __name__ == "__main__":
    main()
