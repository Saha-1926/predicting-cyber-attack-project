"""
Real-time Cybersecurity Monitoring and Anomaly Detection Module
Features: Live traffic monitoring, anomaly detection, threat scoring, and alerting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue
import logging
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import json
import sqlite3
import hashlib
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class RealTimeMonitor:
    """Real-time cybersecurity monitoring system with anomaly detection"""
    
    def __init__(self, buffer_size=1000, anomaly_threshold=0.1):
        self.buffer_size = buffer_size
        self.anomaly_threshold = anomaly_threshold
        self.data_buffer = deque(maxlen=buffer_size)
        self.anomaly_buffer = deque(maxlen=100)
        self.threat_scores = deque(maxlen=buffer_size)
        
        # Models for anomaly detection
        self.isolation_forest = IsolationForest(contamination=anomaly_threshold, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        
        # Threading components
        self.monitoring_active = False
        self.data_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        
        # Threat intelligence
        self.threat_signatures = self._load_threat_signatures()
        self.geo_risk_scores = self._load_geo_risk_scores()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_threat_signatures(self):
        """Load known threat signatures and IoCs"""
        return {
            'malware_hashes': [
                '5d41402abc4b2a76b9719d911017c592',
                'adc83b19e793491b1c6ea0fd8b46cd9f32e592fc'
            ],
            'suspicious_ips': [
                '192.168.1.100', '10.0.0.50', '172.16.0.25'
            ],
            'attack_patterns': {
                'port_scan': {'ports': [22, 23, 80, 443, 3389], 'threshold': 10},
                'brute_force': {'failed_attempts': 5, 'time_window': 300},
                'ddos': {'request_rate': 1000, 'time_window': 60}
            }
        }
    
    def _load_geo_risk_scores(self):
        """Load geographical risk scoring"""
        return {
            'high_risk': ['CN', 'RU', 'KP', 'IR'],
            'medium_risk': ['PK', 'BD', 'ID', 'NG'],
            'low_risk': ['US', 'CA', 'GB', 'DE', 'AU', 'JP']
        }
    
    def calculate_threat_score(self, incident_data):
        """Calculate comprehensive threat score for an incident"""
        try:
            base_score = 0.0
            weights = {
                'financial_impact': 0.3,
                'affected_users': 0.25,
                'attack_sophistication': 0.2,
                'geographic_risk': 0.15,
                'time_criticality': 0.1
            }
            
            # Financial impact scoring (0-100)
            financial_loss = incident_data.get('financial_loss', 0)
            financial_score = min(financial_loss * 10, 100)  # Scale to 0-100
            
            # Affected users scoring (0-100)
            affected_users = incident_data.get('affected_users', 0)
            user_score = min(np.log10(max(affected_users, 1)) * 20, 100)
            
            # Attack sophistication (0-100)
            attack_type = incident_data.get('attack_type', '').lower()
            sophistication_scores = {
                'apt': 90, 'zero_day': 95, 'ransomware': 85,
                'ddos': 60, 'phishing': 40, 'malware': 70,
                'sql_injection': 65, 'xss': 45
            }
            sophistication_score = sophistication_scores.get(attack_type, 50)
            
            # Geographic risk scoring (0-100)
            country = incident_data.get('country', '').upper()
            if country in self.geo_risk_scores['high_risk']:
                geo_score = 80
            elif country in self.geo_risk_scores['medium_risk']:
                geo_score = 50
            else:
                geo_score = 20
            
            # Time criticality (higher score for recent attacks)
            attack_time = incident_data.get('timestamp', datetime.now())
            if isinstance(attack_time, str):
                attack_time = datetime.fromisoformat(attack_time)
            
            time_diff = (datetime.now() - attack_time).total_seconds() / 3600  # hours
            time_score = max(100 - time_diff, 0)  # Decay over time
            
            # Calculate weighted threat score
            threat_score = (
                financial_score * weights['financial_impact'] +
                user_score * weights['affected_users'] +
                sophistication_score * weights['attack_sophistication'] +
                geo_score * weights['geographic_risk'] +
                time_score * weights['time_criticality']
            )
            
            # Add bonus for known threat signatures
            if self._check_threat_signatures(incident_data):
                threat_score += 15
            
            return min(threat_score, 100)  # Cap at 100
            
        except Exception as e:
            self.logger.error(f"Error calculating threat score: {e}")
            return 50.0  # Default moderate threat score
    
    def _check_threat_signatures(self, incident_data):
        """Check for known threat signatures"""
        try:
            # Check for malware hashes
            file_hash = incident_data.get('file_hash', '')
            if file_hash in self.threat_signatures['malware_hashes']:
                return True
            
            # Check for suspicious IPs
            source_ip = incident_data.get('source_ip', '')
            if source_ip in self.threat_signatures['suspicious_ips']:
                return True
            
            return False
        except:
            return False
    
    def detect_anomalies(self, data_point):
        """Detect anomalies in real-time data"""
        try:
            if len(self.data_buffer) < 50:  # Need minimum data for detection
                return False, 0.0
            
            # Convert buffer to numpy array for processing
            buffer_array = np.array(list(self.data_buffer))
            
            # Prepare current data point
            current_point = np.array(data_point).reshape(1, -1)
            
            # Fit isolation forest on buffer data
            self.isolation_forest.fit(buffer_array)
            
            # Check if current point is anomaly
            anomaly_score = self.isolation_forest.decision_function(current_point)[0]
            is_anomaly = self.isolation_forest.predict(current_point)[0] == -1
            
            # Normalize anomaly score to 0-1 range
            normalized_score = (anomaly_score + 0.5) / 1.0
            normalized_score = max(0, min(1, normalized_score))
            
            return is_anomaly, normalized_score
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return False, 0.0
    
    def process_live_data(self, data_stream):
        """Process live network data stream"""
        try:
            for data_point in data_stream:
                # Extract features for anomaly detection
                features = self._extract_features(data_point)
                
                # Add to buffer
                self.data_buffer.append(features)
                
                # Calculate threat score
                threat_score = self.calculate_threat_score(data_point)
                self.threat_scores.append(threat_score)
                
                # Detect anomalies
                is_anomaly, anomaly_score = self.detect_anomalies(features)
                
                # Generate alert if necessary
                if is_anomaly or threat_score > 75:
                    alert = self._generate_alert(data_point, threat_score, anomaly_score)
                    self.alert_queue.put(alert)
                    self.anomaly_buffer.append({
                        'timestamp': datetime.now(),
                        'data': data_point,
                        'threat_score': threat_score,
                        'anomaly_score': anomaly_score,
                        'is_anomaly': is_anomaly
                    })
                
                yield {
                    'data_point': data_point,
                    'threat_score': threat_score,
                    'is_anomaly': is_anomaly,
                    'anomaly_score': anomaly_score
                }
                
        except Exception as e:
            self.logger.error(f"Error processing live data: {e}")
    
    def _extract_features(self, data_point):
        """Extract numerical features for anomaly detection"""
        try:
            features = [
                data_point.get('financial_loss', 0),
                data_point.get('affected_users', 0),
                len(data_point.get('attack_type', '')),
                hash(data_point.get('country', '')) % 1000,
                hash(data_point.get('source_ip', '')) % 1000,
                (datetime.now() - datetime.fromisoformat(
                    data_point.get('timestamp', datetime.now().isoformat())
                )).total_seconds() / 3600
            ]
            return features
        except:
            return [0] * 6
    
    def _generate_alert(self, data_point, threat_score, anomaly_score):
        """Generate security alert"""
        severity = self._get_alert_severity(threat_score)
        
        return {
            'id': hashlib.md5(str(data_point).encode()).hexdigest()[:8],
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'threat_score': threat_score,
            'anomaly_score': anomaly_score,
            'message': f"{severity} threat detected: {data_point.get('attack_type', 'Unknown')}",
            'details': data_point,
            'recommended_actions': self._get_recommended_actions(severity, data_point)
        }
    
    def _get_alert_severity(self, threat_score):
        """Determine alert severity based on threat score"""
        if threat_score >= 90:
            return "CRITICAL"
        elif threat_score >= 75:
            return "HIGH"
        elif threat_score >= 50:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_recommended_actions(self, severity, data_point):
        """Get recommended actions based on alert severity"""
        actions = {
            "CRITICAL": [
                "Immediate incident response team activation",
                "Network isolation of affected systems",
                "Executive notification required",
                "Forensic analysis initiation"
            ],
            "HIGH": [
                "Security team immediate review",
                "Enhanced monitoring of affected systems",
                "Threat hunting activities",
                "Consider system isolation"
            ],
            "MEDIUM": [
                "Security analyst investigation",
                "Log analysis and correlation",
                "User notification if applicable",
                "Preventive measures review"
            ],
            "LOW": [
                "Routine security review",
                "Documentation and tracking",
                "Pattern analysis for trends"
            ]
        }
        return actions.get(severity, ["Review and investigate"])
    
    def get_real_time_dashboard_data(self):
        """Get data for real-time dashboard"""
        try:
            current_time = datetime.now()
            
            # Recent alerts (last hour)
            recent_alerts = []
            temp_queue = queue.Queue()
            
            while not self.alert_queue.empty():
                alert = self.alert_queue.get()
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if (current_time - alert_time).total_seconds() < 3600:  # Last hour
                    recent_alerts.append(alert)
                temp_queue.put(alert)
            
            # Put alerts back in queue
            while not temp_queue.empty():
                self.alert_queue.put(temp_queue.get())
            
            # Threat score statistics
            threat_stats = {
                'current_avg': np.mean(list(self.threat_scores)) if self.threat_scores else 0,
                'current_max': np.max(list(self.threat_scores)) if self.threat_scores else 0,
                'trend': self._calculate_threat_trend()
            }
            
            # Anomaly statistics
            anomaly_count = len([a for a in self.anomaly_buffer 
                               if (current_time - a['timestamp']).total_seconds() < 3600])
            
            return {
                'recent_alerts': recent_alerts,
                'threat_statistics': threat_stats,
                'anomaly_count_hour': anomaly_count,
                'system_status': 'ACTIVE' if self.monitoring_active else 'INACTIVE',
                'buffer_utilization': len(self.data_buffer) / self.buffer_size * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    def _calculate_threat_trend(self):
        """Calculate threat score trend"""
        if len(self.threat_scores) < 10:
            return "STABLE"
        
        recent_scores = list(self.threat_scores)[-10:]
        older_scores = list(self.threat_scores)[-20:-10] if len(self.threat_scores) >= 20 else recent_scores
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        if recent_avg > older_avg * 1.2:
            return "INCREASING"
        elif recent_avg < older_avg * 0.8:
            return "DECREASING"
        else:
            return "STABLE"
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring_active = True
        self.logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        self.logger.info("Real-time monitoring stopped")
    
    def export_alerts(self, format='json'):
        """Export alerts to file"""
        try:
            alerts = []
            temp_queue = queue.Queue()
            
            while not self.alert_queue.empty():
                alert = self.alert_queue.get()
                alerts.append(alert)
                temp_queue.put(alert)
            
            # Put alerts back
            while not temp_queue.empty():
                self.alert_queue.put(temp_queue.get())
            
            if format == 'json':
                return json.dumps(alerts, indent=2)
            elif format == 'csv':
                df = pd.DataFrame(alerts)
                return df.to_csv(index=False)
            
        except Exception as e:
            self.logger.error(f"Error exporting alerts: {e}")
            return ""

class ThreatIntelligenceIntegrator:
    """Integrate external threat intelligence feeds"""
    
    def __init__(self):
        self.feeds = {}
        self.cache_duration = 3600  # 1 hour cache
        self.last_update = {}
    
    def add_threat_feed(self, feed_name, feed_url, api_key=None):
        """Add threat intelligence feed"""
        self.feeds[feed_name] = {
            'url': feed_url,
            'api_key': api_key,
            'active': True
        }
    
    def update_threat_intelligence(self):
        """Update threat intelligence from all feeds"""
        updated_feeds = []
        
        for feed_name, feed_config in self.feeds.items():
            if feed_config['active']:
                try:
                    # Simulate threat intelligence update
                    # In real implementation, this would fetch from actual APIs
                    threat_data = self._simulate_threat_feed_data(feed_name)
                    self.last_update[feed_name] = datetime.now()
                    updated_feeds.append(feed_name)
                except Exception as e:
                    logging.error(f"Failed to update {feed_name}: {e}")
        
        return updated_feeds
    
    def _simulate_threat_feed_data(self, feed_name):
        """Simulate threat intelligence data (replace with real API calls)"""
        return {
            'iocs': [
                {'type': 'ip', 'value': '192.168.1.100', 'confidence': 85},
                {'type': 'hash', 'value': 'abc123def456', 'confidence': 92}
            ],
            'campaigns': [
                {'name': 'APT29', 'active': True, 'techniques': ['T1566', 'T1055']}
            ],
            'vulnerabilities': [
                {'cve': 'CVE-2024-1234', 'severity': 'HIGH', 'exploited': True}
            ]
        }
    
    def get_threat_context(self, indicator):
        """Get threat context for a given indicator"""
        # This would query the threat intelligence database
        return {
            'reputation': 'malicious',
            'confidence': 85,
            'first_seen': '2024-01-15',
            'associated_campaigns': ['APT29'],
            'malware_families': ['Cobalt Strike']
        }
    
    def create_dashboard(self):
        """Create threat intelligence dashboard for Streamlit"""
        st.markdown("## 🔍 Threat Intelligence Dashboard")
        
        # Control panel
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Update Threat Feeds"):
                updated = self.update_threat_intelligence()
                if updated:
                    st.success(f"Updated feeds: {', '.join(updated)}")
                else:
                    st.warning("No feeds were updated")
        
        with col2:
            if st.button("📊 Refresh Dashboard"):
                st.rerun()
        
        # Feed status overview
        st.markdown("### 📡 Threat Feed Status")
        
        if self.feeds:
            feed_data = []
            for feed_name, config in self.feeds.items():
                last_update = self.last_update.get(feed_name, "Never")
                if isinstance(last_update, datetime):
                    last_update = last_update.strftime("%Y-%m-%d %H:%M:%S")
                
                feed_data.append({
                    "Feed Name": feed_name,
                    "Status": "🟢 Active" if config['active'] else "🔴 Inactive",
                    "Last Update": last_update,
                    "URL": config['url'][:50] + "..." if len(config['url']) > 50 else config['url']
                })
            
            st.dataframe(pd.DataFrame(feed_data), use_container_width=True)
        else:
            st.info("No threat feeds configured. Add feeds below.")
        
        # Current threat intelligence summary
        st.markdown("### 🎯 Current Threat Intelligence")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Simulate current threat intelligence data
        threat_data = self._get_current_threat_summary()
        
        with col1:
            st.metric("Active IOCs", threat_data['active_iocs'])
        
        with col2:
            st.metric("Active Campaigns", threat_data['active_campaigns'])
        
        with col3:
            st.metric("Critical CVEs", threat_data['critical_cves'])
        
        with col4:
            st.metric("Feed Updates (24h)", threat_data['feed_updates'])
        
        # Recent indicators of compromise
        st.markdown("### 🚩 Recent Indicators of Compromise (IOCs)")
        
        ioc_data = self._get_recent_iocs()
        if ioc_data:
            st.dataframe(pd.DataFrame(ioc_data), use_container_width=True)
        else:
            st.info("No recent IOCs available")
        
        # Active threat campaigns
        st.markdown("### 🎭 Active Threat Campaigns")
        
        campaigns = self._get_active_campaigns()
        if campaigns:
            for campaign in campaigns:
                with st.expander(f"🎯 {campaign['name']} - {campaign['severity']}"):
                    st.write(f"**Description:** {campaign['description']}")
                    st.write(f"**Active Since:** {campaign['active_since']}")
                    st.write(f"**MITRE ATT&CK Techniques:** {', '.join(campaign['techniques'])}")
                    st.write(f"**Targeted Sectors:** {', '.join(campaign['sectors'])}")
                    
                    if campaign['iocs']:
                        st.write("**Associated IOCs:")
                        for ioc in campaign['iocs']:
                            st.write(f"• {ioc['type']}: `{ioc['value']}` (Confidence: {ioc['confidence']}%)")
        else:
            st.info("No active threat campaigns detected")
        
        # Vulnerability intelligence
        st.markdown("### 🔓 Critical Vulnerabilities")
        
        vuln_data = self._get_critical_vulnerabilities()
        if vuln_data:
            st.dataframe(pd.DataFrame(vuln_data), use_container_width=True)
        else:
            st.info("No critical vulnerabilities in current feed")
        
        # IOC lookup tool
        st.markdown("### 🔍 IOC Lookup Tool")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            ioc_input = st.text_input(
                "Enter IOC to lookup (IP, hash, domain):",
                placeholder="e.g., 192.168.1.100 or abc123def456"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("🔍 Lookup"):
                if ioc_input:
                    context = self.get_threat_context(ioc_input)
                    
                    st.markdown("#### Lookup Results:")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Reputation:** {context['reputation']}")
                        st.write(f"**Confidence:** {context['confidence']}%")
                        st.write(f"**First Seen:** {context['first_seen']}")
                    
                    with col2:
                        st.write(f"**Associated Campaigns:** {', '.join(context['associated_campaigns'])}")
                        st.write(f"**Malware Families:** {', '.join(context['malware_families'])}")
                    
                    # Threat assessment
                    if context['reputation'] == 'malicious':
                        st.error("⚠️ This indicator is associated with malicious activity!")
                    elif context['reputation'] == 'suspicious':
                        st.warning("⚠️ This indicator has suspicious activity.")
                    else:
                        st.success("✅ No malicious activity associated with this indicator.")
                else:
                    st.warning("Please enter an IOC to lookup")
        
        # Feed management
        st.markdown("### ⚙️ Feed Management")
        
        with st.expander("Add New Threat Feed"):
            with st.form("add_feed_form"):
                feed_name = st.text_input("Feed Name", placeholder="e.g., VirusTotal, AlienVault")
                feed_url = st.text_input("Feed URL", placeholder="https://api.example.com/threats")
                api_key = st.text_input("API Key (optional)", type="password")
                
                if st.form_submit_button("Add Feed"):
                    if feed_name and feed_url:
                        self.add_threat_feed(feed_name, feed_url, api_key if api_key else None)
                        st.success(f"Added threat feed: {feed_name}")
                        st.rerun()
                    else:
                        st.error("Please provide both feed name and URL")
    
    def _get_current_threat_summary(self):
        """Get current threat intelligence summary"""
        return {
            'active_iocs': 1247,
            'active_campaigns': 15,
            'critical_cves': 23,
            'feed_updates': 8
        }
    
    def _get_recent_iocs(self):
        """Get recent indicators of compromise"""
        return [
            {
                "Type": "IP Address",
                "Value": "192.168.1.100",
                "Confidence": "85%",
                "Source": "ThreatFeed1",
                "First Seen": "2024-01-15 10:30:00",
                "Threat Type": "C2 Server"
            },
            {
                "Type": "File Hash",
                "Value": "abc123def456789",
                "Confidence": "92%",
                "Source": "VirusTotal",
                "First Seen": "2024-01-15 09:15:00",
                "Threat Type": "Malware"
            },
            {
                "Type": "Domain",
                "Value": "malicious-domain.com",
                "Confidence": "78%",
                "Source": "ThreatFeed2",
                "First Seen": "2024-01-15 08:45:00",
                "Threat Type": "Phishing"
            },
            {
                "Type": "URL",
                "Value": "http://suspicious-site.net/payload",
                "Confidence": "89%",
                "Source": "URLVoid",
                "First Seen": "2024-01-15 07:20:00",
                "Threat Type": "Malware Dropper"
            }
        ]
    
    def _get_active_campaigns(self):
        """Get active threat campaigns"""
        return [
            {
                "name": "APT29 (Cozy Bear)",
                "severity": "HIGH",
                "description": "Advanced persistent threat group conducting espionage operations against government and private sector targets.",
                "active_since": "2024-01-10",
                "techniques": ["T1566", "T1055", "T1083", "T1005"],
                "sectors": ["Government", "Healthcare", "Technology"],
                "iocs": [
                    {"type": "IP", "value": "203.0.113.15", "confidence": 90},
                    {"type": "Hash", "value": "d41d8cd98f00b204e9800998ecf8427e", "confidence": 85}
                ]
            },
            {
                "name": "Lazarus Group",
                "severity": "CRITICAL",
                "description": "North Korean-linked group targeting financial institutions and cryptocurrency exchanges.",
                "active_since": "2024-01-08",
                "techniques": ["T1566", "T1078", "T1105", "T1043"],
                "sectors": ["Financial", "Cryptocurrency", "Entertainment"],
                "iocs": [
                    {"type": "Domain", "value": "fake-bank-portal.com", "confidence": 95},
                    {"type": "IP", "value": "198.51.100.42", "confidence": 88}
                ]
            },
            {
                "name": "Fancy Bear (APT28)",
                "severity": "HIGH",
                "description": "Russian military intelligence-linked group targeting NATO countries and Ukraine.",
                "active_since": "2024-01-05",
                "techniques": ["T1566", "T1059", "T1087", "T1003"],
                "sectors": ["Government", "Military", "Defense Contractors"],
                "iocs": [
                    {"type": "Email", "value": "spear-phish@fake-gov.org", "confidence": 82},
                    {"type": "Hash", "value": "5d41402abc4b2a76b9719d911017c592", "confidence": 91}
                ]
            }
        ]
    
    def _get_critical_vulnerabilities(self):
        """Get critical vulnerabilities from threat feeds"""
        return [
            {
                "CVE ID": "CVE-2024-1234",
                "Severity": "CRITICAL",
                "CVSS Score": "9.8",
                "Product": "Apache Struts",
                "Status": "Actively Exploited",
                "Published": "2024-01-14",
                "Description": "Remote code execution vulnerability in Apache Struts framework"
            },
            {
                "CVE ID": "CVE-2024-5678",
                "Severity": "HIGH",
                "CVSS Score": "8.1",
                "Product": "Microsoft Exchange",
                "Status": "Proof of Concept",
                "Published": "2024-01-13",
                "Description": "Privilege escalation vulnerability in Exchange Server"
            },
            {
                "CVE ID": "CVE-2024-9012",
                "Severity": "HIGH",
                "CVSS Score": "7.5",
                "Product": "Cisco IOS",
                "Status": "Not Exploited",
                "Published": "2024-01-12",
                "Description": "Information disclosure vulnerability in Cisco IOS software"
            }
        ]

# Streamlit integration functions
def create_realtime_dashboard():
    """Create real-time monitoring dashboard for Streamlit"""
    
    if 'monitor' not in st.session_state:
        st.session_state.monitor = RealTimeMonitor()
    
    monitor = st.session_state.monitor
    
    st.markdown("## 🔴 Real-Time Threat Monitoring")
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Monitoring"):
            monitor.start_monitoring()
            st.success("Monitoring started")
    
    with col2:
        if st.button("⏹️ Stop Monitoring"):
            monitor.stop_monitoring()
            st.warning("Monitoring stopped")
    
    with col3:
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()
    
    # Dashboard metrics
    dashboard_data = monitor.get_real_time_dashboard_data()
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if dashboard_data.get('system_status') == 'ACTIVE' else "🔴"
        st.metric("System Status", f"{status_color} {dashboard_data.get('system_status', 'UNKNOWN')}")
    
    with col2:
        threat_avg = dashboard_data.get('threat_statistics', {}).get('current_avg', 0)
        trend = dashboard_data.get('threat_statistics', {}).get('trend', 'STABLE')
        trend_icon = {"INCREASING": "📈", "DECREASING": "📉", "STABLE": "➡️"}.get(trend, "➡️")
        st.metric("Avg Threat Score", f"{threat_avg:.1f}", f"{trend_icon} {trend}")
    
    with col3:
        st.metric("Anomalies (1h)", dashboard_data.get('anomaly_count_hour', 0))
    
    with col4:
        buffer_util = dashboard_data.get('buffer_utilization', 0)
        st.metric("Buffer Usage", f"{buffer_util:.1f}%")
    
    # Recent alerts
    st.markdown("### 🚨 Recent Alerts")
    recent_alerts = dashboard_data.get('recent_alerts', [])
    
    if recent_alerts:
        for alert in recent_alerts[-5:]:  # Show last 5 alerts
            severity_colors = {
                'CRITICAL': '🔴', 'HIGH': '🟠', 
                'MEDIUM': '🟡', 'LOW': '🔵'
            }
            severity_icon = severity_colors.get(alert['severity'], '⚪')
            
            with st.expander(f"{severity_icon} {alert['severity']} - {alert['message']}"):
                st.write(f"**Threat Score:** {alert['threat_score']:.1f}")
                st.write(f"**Anomaly Score:** {alert['anomaly_score']:.3f}")
                st.write(f"**Time:** {alert['timestamp']}")
                st.write("**Recommended Actions:**")
                for action in alert['recommended_actions']:
                    st.write(f"• {action}")
    else:
        st.info("No recent alerts")
    
    # Threat score timeline
    st.markdown("### 📊 Threat Score Timeline")
    if monitor.threat_scores:
        scores = list(monitor.threat_scores)
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(len(scores)-1, -1, -1)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps, y=scores,
            mode='lines+markers',
            name='Threat Score',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Real-Time Threat Score Monitoring",
            xaxis_title="Time",
            yaxis_title="Threat Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No threat score data available")

if __name__ == "__main__":
    # Example usage
    monitor = RealTimeMonitor()
    
    # Simulate some data points
    sample_data = [
        {
            'timestamp': datetime.now().isoformat(),
            'attack_type': 'ddos',
            'country': 'CN',
            'financial_loss': 50000,
            'affected_users': 10000,
            'source_ip': '192.168.1.100'
        }
    ]
    
    # Process data
    for result in monitor.process_live_data(sample_data):
        print(f"Processed: {result}")
