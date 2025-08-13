import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           mean_squared_error, mean_absolute_error, classification_report, 
                           confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

class CyberSecurityMLPipeline:
    def __init__(self):
        self.models = {}
        self.model_performances = {}
        self.feature_importance = {}
        self.explainers = {}
        
    def train_classification_models(self, X_train, X_test, y_train, y_test, task_name):
        """Train multiple classification models"""
        models_to_train = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'neural_network': MLPClassifier(random_state=42, max_iter=500)
        }
        
        results = {}
        
        for model_name, model in models_to_train.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # ROC AUC for binary classification
                try:
                    if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                except:
                    roc_auc = 0.0
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Store model
                self.models[f'{task_name}_{model_name}'] = model
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[f'{task_name}_{model_name}'] = model.feature_importances_
                
            except Exception as e:
                print(f"Error training {model_name} for {task_name}: {str(e)}")
                continue
        
        self.model_performances[f'{task_name}_classification'] = results
        return results
    
    def train_regression_models(self, X_train, X_test, y_train, y_test, task_name):
        """Train multiple regression models"""
        models_to_train = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42),
            'lightgbm': lgb.LGBMRegressor(random_state=42, verbose=-1),
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(random_state=42),
            'lasso_regression': Lasso(random_state=42),
            'svr': SVR(),
            'neural_network': MLPRegressor(random_state=42, max_iter=500)
        }
        
        results = {}
        
        for model_name, model in models_to_train.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                
                # R-squared
                from sklearn.metrics import r2_score
                r2 = r2_score(y_test, y_pred)
                
                results[model_name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2,
                    'predictions': y_pred
                }
                
                # Store model
                self.models[f'{task_name}_{model_name}'] = model
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[f'{task_name}_{model_name}'] = model.feature_importances_
                
            except Exception as e:
                print(f"Error training {model_name} for {task_name}: {str(e)}")
                continue
        
        self.model_performances[f'{task_name}_regression'] = results
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='random_forest', task='classification'):
        """Perform hyperparameter tuning"""
        if task == 'classification':
            if model_type == 'random_forest':
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_type == 'xgboost':
                model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
        else:  # regression
            if model_type == 'random_forest':
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_type == 'xgboost':
                model = xgb.XGBRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=5, 
            scoring='accuracy' if task == 'classification' else 'neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def handle_class_imbalance(self, X, y, method='smote'):
        """Handle class imbalance using various techniques"""
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        else:
            raise ValueError("Method must be 'smote', 'adasyn', or 'undersample'")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def anomaly_detection(self, X, contamination=0.1):
        """Detect anomalies using Isolation Forest"""
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)
        
        # -1 for anomalies, 1 for normal instances
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        
        self.models['anomaly_detector'] = iso_forest
        
        return anomaly_labels, anomaly_indices
    
    def explain_predictions_shap(self, model_name, X_sample, feature_names):
        """Generate SHAP explanations for model predictions"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Create SHAP explainer
        if 'xgboost' in model_name or 'lightgbm' in model_name:
            explainer = shap.TreeExplainer(model)
        elif 'random_forest' in model_name:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)
        
        # Get SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        self.explainers[model_name] = explainer
        
        return shap_values, explainer
    
    def time_series_features(self, df, date_column='Year'):
        """Create time series features for temporal analysis"""
        df_ts = df.copy()
        
        # Time-based features
        df_ts['Year_Sin'] = np.sin(2 * np.pi * df_ts[date_column] / 12)
        df_ts['Year_Cos'] = np.cos(2 * np.pi * df_ts[date_column] / 12)
        
        # Lag features for attack counts
        attack_counts = df_ts.groupby(date_column).size().reset_index(name='Attack_Count')
        attack_counts['Attack_Count_Lag1'] = attack_counts['Attack_Count'].shift(1)
        attack_counts['Attack_Count_Lag2'] = attack_counts['Attack_Count'].shift(2)
        
        # Rolling averages
        attack_counts['Attack_Count_MA3'] = attack_counts['Attack_Count'].rolling(window=3).mean()
        attack_counts['Attack_Count_MA5'] = attack_counts['Attack_Count'].rolling(window=5).mean()
        
        # Merge back with original data
        df_ts = df_ts.merge(attack_counts, on=date_column, how='left')
        
        return df_ts
    
    def cross_validation_evaluation(self, model, X, y, cv=5):
        """Perform cross-validation evaluation"""
        if hasattr(model, 'predict_proba'):  # Classification
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:  # Regression
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        cv_results = {}
        
        for score in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=score)
            cv_results[score] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
        
        return cv_results
    
    def ensemble_predictions(self, models_list, X_test, method='voting'):
        """Combine predictions from multiple models"""
        if method == 'voting':
            # Simple majority voting for classification
            predictions = []
            for model in models_list:
                pred = model.predict(X_test)
                predictions.append(pred)
            
            # Convert to array and get mode
            predictions = np.array(predictions)
            ensemble_pred = []
            
            for i in range(predictions.shape[1]):
                votes = predictions[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                majority_vote = unique[np.argmax(counts)]
                ensemble_pred.append(majority_vote)
            
            return np.array(ensemble_pred)
        
        elif method == 'averaging':
            # Average predictions for regression
            predictions = []
            for model in models_list:
                pred = model.predict(X_test)
                predictions.append(pred)
            
            return np.mean(predictions, axis=0)
    
    def feature_selection(self, X, y, method='importance', k=10):
        """Perform feature selection"""
        if method == 'importance':
            # Use Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            feature_importance = rf.feature_importances_
            indices = np.argsort(feature_importance)[::-1][:k]
            
            return indices, feature_importance[indices]
        
        elif method == 'correlation':
            # Use correlation with target
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            
            indices = np.argsort(correlations)[::-1][:k]
            return indices, np.array(correlations)[indices]
    
    def save_models(self, filepath_prefix):
        """Save trained models to disk"""
        for model_name, model in self.models.items():
            filename = f"{filepath_prefix}_{model_name}.joblib"
            joblib.dump(model, filename)
        
        # Save performance metrics
        performance_filename = f"{filepath_prefix}_performance.joblib"
        joblib.dump(self.model_performances, performance_filename)
    
    def load_models(self, filepath_prefix):
        """Load models from disk"""
        import os
        
        # Find all model files
        model_files = [f for f in os.listdir('.') if f.startswith(filepath_prefix) and f.endswith('.joblib')]
        
        for file in model_files:
            if 'performance' not in file:
                model_name = file.replace(f"{filepath_prefix}_", "").replace(".joblib", "")
                self.models[model_name] = joblib.load(file)
        
        # Load performance metrics
        performance_file = f"{filepath_prefix}_performance.joblib"
        if os.path.exists(performance_file):
            self.model_performances = joblib.load(performance_file)
    
    def generate_model_report(self):
        """Generate comprehensive model performance report"""
        report = {}
        
        for task, performances in self.model_performances.items():
            report[task] = {}
            
            if 'classification' in task:
                # Classification metrics
                for model_name, metrics in performances.items():
                    report[task][model_name] = {
                        'Accuracy': f"{metrics['accuracy']:.4f}",
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1-Score': f"{metrics['f1_score']:.4f}",
                        'ROC-AUC': f"{metrics['roc_auc']:.4f}"
                    }
            else:
                # Regression metrics
                for model_name, metrics in performances.items():
                    report[task][model_name] = {
                        'RMSE': f"{metrics['rmse']:.4f}",
                        'MAE': f"{metrics['mae']:.4f}",
                        'R²': f"{metrics['r2_score']:.4f}"
                    }
        
        return report

class TimeSeriesPredictor:
    def __init__(self):
        self.models = {}
        
    def prepare_time_series_data(self, df, target_col, time_col='Year', window_size=3):
        """Prepare data for time series prediction"""
        # Sort by time
        df_sorted = df.sort_values(time_col)
        
        # Aggregate by time period
        time_series = df_sorted.groupby(time_col)[target_col].agg(['count', 'mean', 'sum']).reset_index()
        
        # Create sliding windows
        X, y = [], []
        
        for i in range(window_size, len(time_series)):
            X.append(time_series.iloc[i-window_size:i][['count', 'mean', 'sum']].values.flatten())
            y.append(time_series.iloc[i]['count'])  # Predict attack count
        
        return np.array(X), np.array(y), time_series
    
    def train_lstm_model(self, X, y):
        """Train LSTM model for time series prediction"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            # Reshape data for LSTM
            X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(0.001), loss='mse')
            
            # Train model
            history = model.fit(X_reshaped, y, epochs=50, batch_size=32, verbose=0)
            
            self.models['lstm'] = model
            return model, history
            
        except ImportError:
            print("TensorFlow not available. Using traditional ML for time series.")
            return None, None
    
    def predict_future_attacks(self, model, last_window, n_steps=5):
        """Predict future attack patterns"""
        predictions = []
        current_window = last_window.copy()
        
        for _ in range(n_steps):
            # Reshape for prediction
            if len(current_window.shape) == 1:
                pred_input = current_window.reshape(1, -1)
            else:
                pred_input = current_window.reshape(1, current_window.shape[0], 1)
            
            # Make prediction
            pred = model.predict(pred_input, verbose=0)[0]
            predictions.append(pred)
            
            # Update window (simple approach - can be improved)
            if len(current_window.shape) == 1:
                current_window = np.roll(current_window, -1)
                current_window[-1] = pred
            
        return np.array(predictions)
