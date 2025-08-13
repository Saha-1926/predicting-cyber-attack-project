# Configuration file for Cybersecurity Application
import os

# Network Configuration
class NetworkConfig:
    """Network configuration for WiFi accessibility"""
    
    # Server Configuration
    HOST = '0.0.0.0'  # Listen on all network interfaces (allows WiFi access)
    PORT = 8501       # Default Streamlit port
    
    # Alternative ports in case default is busy
    ALTERNATIVE_PORTS = [8502, 8503, 8504, 8505]
    
    # Database Configuration  
    DATABASE_PATH = 'cybersecurity_alerts.db'
    
    # Session Configuration
    SESSION_TIMEOUT = 3600  # 1 hour in seconds
    
    @staticmethod
    def get_local_ip():
        """Get local IP address for WiFi access"""
        import socket
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            return local_ip
        except Exception:
            return "127.0.0.1"
    
    @staticmethod
    def get_access_urls():
        """Get all possible access URLs"""
        local_ip = NetworkConfig.get_local_ip()
        base_ports = [NetworkConfig.PORT] + NetworkConfig.ALTERNATIVE_PORTS
        
        urls = []
        # Local access
        for port in base_ports:
            urls.append(f"http://localhost:{port}")
            urls.append(f"http://127.0.0.1:{port}")
        
        # WiFi/Network access
        for port in base_ports:
            urls.append(f"http://{local_ip}:{port}")
            
        return urls

# Alert Configuration
class AlertConfig:
    """Configuration for alert management"""
    
    # Alert severity levels
    SEVERITY_LEVELS = ['Critical', 'High', 'Medium', 'Low']
    
    # Alert types
    ALERT_TYPES = [
        'Intrusion', 'Phishing', 'Malware', 'DDoS', 
        'Data Breach', 'Ransomware', 'Unauthorized Access',
        'Suspicious Activity', 'Policy Violation', 'Other'
    ]
    
    # Alert statuses
    ALERT_STATUSES = ['Active', 'Investigating', 'Resolved', 'False Positive']
    
    # Default alert retention (days)
    ALERT_RETENTION_DAYS = 90
    
    # Maximum alerts to display at once
    MAX_DISPLAY_ALERTS = 100

# Application Configuration
class AppConfig:
    """General application configuration"""
    
    # Application metadata
    APP_NAME = "Cybersecurity Threat Predictor"
    APP_VERSION = "1.0.0"
    
    # File upload limits
    MAX_FILE_SIZE_MB = 50
    ALLOWED_FILE_TYPES = ['csv', 'json', 'xlsx']
    
    # Performance settings
    CACHE_TTL = 3600  # Cache time-to-live in seconds
    
    # Security settings
    ENABLE_AUTH = False  # Set to True to enable authentication
    SESSION_ENCRYPTION = True
