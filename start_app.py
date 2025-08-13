#!/usr/bin/env python3
"""
Startup script for Cybersecurity Application
This script ensures the app is accessible via WiFi on any network
"""

import os
import sys
import subprocess
import socket
from config import NetworkConfig, AppConfig

def check_port_availability(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False

def find_available_port():
    """Find an available port from the configured options"""
    ports_to_try = [NetworkConfig.PORT] + NetworkConfig.ALTERNATIVE_PORTS
    
    for port in ports_to_try:
        if check_port_availability(port):
            return port
    
    # If no configured ports are available, try random ports
    for port in range(8506, 8600):
        if check_port_availability(port):
            return port
    
    return None

def display_access_info(port):
    """Display access information for users"""
    local_ip = NetworkConfig.get_local_ip()
    
    print("=" * 60)
    print(f"🛡️  {AppConfig.APP_NAME} v{AppConfig.APP_VERSION}")
    print("=" * 60)
    print(f"✅ Server started successfully on port {port}")
    print(f"🌐 Application is accessible via WiFi!")
    print()
    print("📡 Access URLs:")
    print(f"   • Local access:    http://localhost:{port}")
    print(f"   • WiFi access:     http://{local_ip}:{port}")
    print(f"   • Network access:  http://127.0.0.1:{port}")
    print()
    print("📱 To access from other devices on the same WiFi:")
    print(f"   1. Connect your device to the same WiFi network")
    print(f"   2. Open browser and go to: http://{local_ip}:{port}")
    print()
    print("🔧 Troubleshooting:")
    print("   • Make sure your firewall allows connections on this port")
    print("   • Ensure all devices are on the same WiFi network")
    print("   • Try disabling VPN if connection fails")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

def setup_environment():
    """Setup the environment for the application"""
    # Set environment variables for Streamlit
    os.environ['STREAMLIT_SERVER_PORT'] = str(NetworkConfig.PORT)
    os.environ['STREAMLIT_SERVER_ADDRESS'] = NetworkConfig.HOST
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Disable Streamlit warnings
    os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'scikit-learn', 'xgboost', 'lightgbm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print()
        print("📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print()
        return False
    
    return True

def main():
    """Main function to start the application"""
    print(f"🚀 Starting {AppConfig.APP_NAME}...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Find available port
    available_port = find_available_port()
    if not available_port:
        print("❌ Error: No available ports found!")
        print("   Please close other applications using ports 8501-8599")
        sys.exit(1)
    
    # Update port in environment
    os.environ['STREAMLIT_SERVER_PORT'] = str(available_port)
    
    # Setup environment
    setup_environment()
    
    # Display access information
    display_access_info(available_port)
    
    try:
        # Start Streamlit app
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', str(available_port),
            '--server.address', NetworkConfig.HOST,
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            '--logger.level', 'error'
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
