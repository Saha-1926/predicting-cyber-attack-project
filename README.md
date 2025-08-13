# 🛡️ Cybersecurity Attack Prediction System

A comprehensive machine learning-powered system for predicting and analyzing cybersecurity attacks using historical data. Built with **Streamlit**, **Python**, and advanced ML algorithms.

## 🌟 Features

### Core Functionality
- **Multi-Model Prediction**: Attack type classification, financial loss estimation, resolution time prediction, and severity assessment
- **Real-time Analysis**: Interactive dashboard with live data processing
- **Advanced Analytics**: Time series analysis, correlation studies, and trend identification
- **Model Explainability**: SHAP values and feature importance analysis
- **Anomaly Detection**: Identify unusual attack patterns and outliers
- **Data Upload**: Support for custom datasets via CSV upload

### Machine Learning Models
- **Random Forest** (Classification & Regression)
- **XGBoost** (High-performance gradient boosting)
- **LightGBM** (Fast gradient boosting)
- **Support Vector Machines** (SVM/SVR)
- **Neural Networks** (MLPClassifier/MLPRegressor)
- **Logistic/Linear Regression**
- **Isolation Forest** (Anomaly detection)

### Key Prediction Capabilities
1. **Attack Type Classification**: Predict DDoS, Phishing, Ransomware, SQL Injection, etc.
2. **Financial Impact Assessment**: Estimate monetary losses in millions
3. **Resolution Time Prediction**: Forecast incident response duration
4. **Severity Level Classification**: Low, Medium, High, Critical threat levels
5. **Risk Scoring**: Comprehensive threat assessment

## 📋 Prerequisites

- Python 3.7 or higher
- Windows/macOS/Linux
- 4GB+ RAM recommended
- Internet connection for package installation

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Navigate to the project directory
cd "C:\Users\KingSaha\Downloads\Cyber\Cyber"

# Run the automated setup script
python setup_and_run.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py
```

The application will automatically open in your web browser at `http://localhost:8501`

## 📁 Project Structure

```
Cybersecurity-Attack-Predictor/
├── app.py                              # Main Streamlit application
├── data_utils.py                       # Data processing utilities
├── ml_models.py                        # Machine learning pipeline
├── setup_and_run.py                    # Automated setup script
├── requirements.txt                    # Python dependencies
├── README.md                           # Documentation
└── Global_Cybersecurity_Threats_2015-2024.csv  # Sample dataset
```

## 🖥️ Application Interface

### 1. 📊 Dashboard
- **System Overview**: Key metrics and statistics
- **Attack Distribution**: Pie charts and trend analysis
- **Geographic Analysis**: Country-wise attack patterns
- **Time Series Trends**: Yearly attack evolution

### 2. 🤖 Model Training
- **Multi-Model Training**: Train all ML models simultaneously
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Model Comparison**: Side-by-side performance analysis
- **Training Progress**: Real-time training status

### 3. 🔮 Attack Prediction
- **Interactive Form**: Input attack parameters
- **Real-time Predictions**: Instant results with confidence scores
- **Risk Assessment**: Automated threat level evaluation
- **Multi-target Outputs**: Type, financial loss, resolution time, severity

### 4. 📈 Advanced Analytics
- **Time Series Analysis**: Temporal attack patterns
- **Correlation Matrix**: Feature relationship analysis
- **Industry Risk Analysis**: Sector-specific insights
- **Attack Source Analysis**: Origin-based statistics

### 5. 📋 Data Management
- **CSV Upload**: Custom dataset support
- **Data Preview**: Interactive data exploration
- **Statistics Summary**: Comprehensive data overview
- **Missing Value Analysis**: Data quality assessment

### 6. 🧠 Model Insights
- **Feature Importance**: ML model interpretability
- **Performance Comparison**: Cross-model evaluation
- **SHAP Analysis**: Explainable AI insights
- **Model Metrics**: Detailed performance statistics

## 📊 Dataset Features

The system works with cybersecurity incident data containing:

| Feature | Description | Type |
|---------|-------------|------|
| Country | Geographic location of attack | Categorical |
| Year | Year of incident (2015-2024) | Numerical |
| Attack Type | Type of cybersecurity threat | Categorical |
| Target Industry | Affected business sector | Categorical |
| Financial Loss | Monetary impact in millions | Numerical |
| Affected Users | Number of impacted users | Numerical |
| Attack Source | Origin of the attack | Categorical |
| Vulnerability Type | Exploited security weakness | Categorical |
| Defense Mechanism | Security measure used | Categorical |
| Resolution Time | Incident response duration | Numerical |

## 🤖 Machine Learning Pipeline

### Data Preprocessing
1. **Missing Value Handling**: Median/mode imputation
2. **Feature Engineering**: Severity scoring, risk levels
3. **Categorical Encoding**: Label encoding for ML compatibility
4. **Feature Scaling**: StandardScaler for numerical features
5. **Class Balancing**: SMOTE for imbalanced datasets

### Model Training
1. **Data Splitting**: 80/20 train-test split
2. **Cross-Validation**: 5-fold CV for robust evaluation
3. **Hyperparameter Tuning**: GridSearchCV optimization
4. **Ensemble Methods**: Multiple model combination
5. **Performance Evaluation**: Comprehensive metrics

### Prediction Tasks
1. **Classification**: Attack type, severity level
2. **Regression**: Financial loss, resolution time
3. **Anomaly Detection**: Unusual pattern identification
4. **Time Series**: Temporal trend forecasting

## 📈 Model Performance

### Classification Metrics
- **Accuracy**: Overall correctness rate
- **Precision**: True positive rate
- **Recall**: Sensitivity measure
- **F1-Score**: Harmonic mean of precision/recall
- **ROC-AUC**: Area under ROC curve

### Regression Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MSE**: Mean Square Error

## 🔧 Customization

### Adding New Models
```python
# In ml_models.py
def train_custom_model(self, X_train, X_test, y_train, y_test):
    model = YourCustomModel()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Add evaluation metrics
```

### Custom Features
```python
# In data_utils.py
def custom_feature_engineering(self, df):
    df['new_feature'] = df['existing_feature'].apply(custom_function)
    return df
```

### UI Modifications
```python
# In app.py
def custom_page():
    st.markdown("## Your Custom Page")
    # Add your custom functionality
```

## 📚 Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **plotly**: Interactive visualizations

### ML Libraries
- **xgboost**: Gradient boosting framework
- **lightgbm**: Fast gradient boosting
- **imbalanced-learn**: Handling imbalanced datasets
- **shap**: Model explainability

### Visualization
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive charts

## 🚀 Deployment Options

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Cloud Deployment
1. **Streamlit Cloud**: Direct GitHub integration
2. **Heroku**: Platform-as-a-Service deployment
3. **AWS/GCP/Azure**: Cloud platform hosting
4. **Docker**: Containerized deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🔒 Security Considerations

- **Data Privacy**: No sensitive data storage
- **Input Validation**: Sanitized user inputs
- **Model Security**: Secure model serialization
- **Access Control**: Authentication mechanisms (if needed)

## 🐛 Troubleshooting

### Common Issues

1. **Package Installation Errors**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

2. **Streamlit Not Starting**
   ```bash
   streamlit --version
   streamlit config show
   ```

3. **Memory Issues**
   - Reduce dataset size
   - Use data sampling
   - Optimize model parameters

4. **Model Training Failures**
   - Check data quality
   - Verify feature types
   - Adjust model parameters

## 📖 Usage Examples

### Predicting Attack Characteristics
```python
# Input parameters
input_data = {
    'Country': 'USA',
    'Industry': 'Banking',
    'Attack_Source': 'Hacker Group',
    'Vulnerability': 'Unpatched Software',
    'Defense': 'AI-based Detection',
    'Year': 2024,
    'Affected_Users': 100000
}

# Get predictions
predictions = predictor.predict_attack(input_data)
print(f"Attack Type: {predictions['attack_type']}")
print(f"Financial Loss: ${predictions['financial_loss']:.2f}M")
```

### Custom Data Analysis
```python
# Load custom dataset
df = pd.read_csv('your_cybersecurity_data.csv')

# Process and analyze
processor = DataProcessor()
df_processed = processor.engineer_features(df)
visualizer = DataVisualizer()
fig = visualizer.plot_attack_trends(df_processed)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **scikit-learn**: Machine learning framework
- **Streamlit**: Web application framework
- **XGBoost/LightGBM**: Gradient boosting implementations
- **Plotly**: Interactive visualization library
- **SHAP**: Model explainability tools

## 📞 Support

For questions, issues, or contributions:

1. **GitHub Issues**: Report bugs and request features
2. **Documentation**: Check this README for detailed information
3. **Community**: Join cybersecurity ML discussions

---

**Built with ❤️ for cybersecurity professionals and data scientists**

*Empowering proactive cyber defense through machine learning and predictive analytics*
"# java-ml-project" 
