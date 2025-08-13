import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

class DataProcessor:
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.feature_columns = []
        
    def load_and_clean_data(self, file_path):
        """Load and perform initial cleaning of cybersecurity data"""
        try:
            df = pd.read_csv(file_path)
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            numeric_columns = ['Financial Loss (in Million $)', 'Number of Affected Users', 
                             'Incident Resolution Time (in Hours)']
            
            for col in numeric_columns:
                if df[col].isnull().sum() > 0:
                    # Fill with median for numeric columns
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical missing values with mode
            categorical_columns = ['Attack Type', 'Target Industry', 'Attack Source', 
                                 'Security Vulnerability Type', 'Defense Mechanism Used']
            
            for col in categorical_columns:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def engineer_features(self, df):
        """Create additional features for better prediction"""
        df_enhanced = df.copy()
        
        # Time-based features
        df_enhanced['Years_Since_2015'] = df_enhanced['Year'] - 2015
        df_enhanced['Is_Recent'] = (df_enhanced['Year'] >= 2020).astype(int)
        
        # Severity scoring
        df_enhanced['Financial_Impact_Score'] = pd.qcut(
            df_enhanced['Financial Loss (in Million $)'], 
            q=5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        df_enhanced['User_Impact_Score'] = pd.qcut(
            df_enhanced['Number of Affected Users'], 
            q=5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        df_enhanced['Resolution_Efficiency'] = pd.qcut(
            df_enhanced['Incident Resolution Time (in Hours)'], 
            q=5, labels=[5, 4, 3, 2, 1]  # Lower time = higher efficiency
        ).astype(int)
        
        # Combined severity score
        df_enhanced['Overall_Severity'] = (
            df_enhanced['Financial_Impact_Score'] * 0.4 +
            df_enhanced['User_Impact_Score'] * 0.3 +
            (6 - df_enhanced['Resolution_Efficiency']) * 0.3
        )
        
        # Attack frequency by country/industry
        country_counts = df_enhanced.groupby('Country').size()
        industry_counts = df_enhanced.groupby('Target Industry').size()
        
        df_enhanced['Country_Risk_Level'] = df_enhanced['Country'].map(country_counts)
        df_enhanced['Industry_Risk_Level'] = df_enhanced['Target Industry'].map(industry_counts)
        
        # Binary features for common attack characteristics
        df_enhanced['Is_High_Value_Target'] = df_enhanced['Target Industry'].isin(
            ['Banking', 'Government', 'Healthcare']
        ).astype(int)
        
        df_enhanced['Is_Advanced_Attack'] = df_enhanced['Attack Type'].isin(
            ['Ransomware', 'Man-in-the-Middle', 'SQL Injection']
        ).astype(int)
        
        df_enhanced['Has_AI_Defense'] = (
            df_enhanced['Defense Mechanism Used'] == 'AI-based Detection'
        ).astype(int)
        
        return df_enhanced
    
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        categorical_columns = ['Country', 'Attack Type', 'Target Industry', 
                             'Attack Source', 'Security Vulnerability Type', 
                             'Defense Mechanism Used']
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
            self.encoders[col] = le
        
        return df_encoded
    
    def scale_features(self, df, method='standard'):
        """Scale numerical features"""
        df_scaled = df.copy()
        
        numerical_columns = [
            'Financial Loss (in Million $)', 'Number of Affected Users',
            'Incident Resolution Time (in Hours)', 'Years_Since_2015',
            'Financial_Impact_Score', 'User_Impact_Score', 'Resolution_Efficiency',
            'Overall_Severity', 'Country_Risk_Level', 'Industry_Risk_Level'
        ]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
        self.scalers['numerical'] = scaler
        
        return df_scaled
    
    def get_feature_columns(self):
        """Return list of feature columns for model training"""
        feature_cols = [
            'Country_encoded', 'Target Industry_encoded', 'Attack Source_encoded',
            'Security Vulnerability Type_encoded', 'Defense Mechanism Used_encoded',
            'Years_Since_2015', 'Number of Affected Users', 'Financial_Impact_Score',
            'User_Impact_Score', 'Resolution_Efficiency', 'Country_Risk_Level',
            'Industry_Risk_Level', 'Is_High_Value_Target', 'Is_Advanced_Attack',
            'Has_AI_Defense', 'Is_Recent'
        ]
        
        self.feature_columns = feature_cols
        return feature_cols
    
    def create_target_variables(self, df):
        """Create different target variables for various prediction tasks"""
        targets = {}
        
        # Attack type classification
        targets['attack_type'] = df['Attack Type_encoded']
        
        # Financial loss regression (original scale)
        targets['financial_loss'] = df['Financial Loss (in Million $)']
        
        # Resolution time regression
        targets['resolution_time'] = df['Incident Resolution Time (in Hours)']
        
        # Severity classification (4 levels)
        targets['severity_level'] = pd.cut(
            df['Overall_Severity'], 
            bins=4, 
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Binary high-impact classification
        targets['high_impact'] = (
            df['Financial Loss (in Million $)'] > df['Financial Loss (in Million $)'].quantile(0.75)
        ).astype(int)
        
        return targets

class DataVisualizer:
    def __init__(self):
        pass
    
    def plot_attack_trends(self, df):
        """Create time series plots for attack trends"""
        yearly_trends = df.groupby(['Year', 'Attack Type']).size().reset_index(name='Count')
        
        fig = px.line(yearly_trends, x='Year', y='Count', color='Attack Type',
                     title='Cybersecurity Attack Trends Over Time',
                     labels={'Count': 'Number of Attacks'})
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Attacks",
            legend_title="Attack Type",
            hovermode='x unified'
        )
        
        return fig
    
    def plot_geographic_distribution(self, df):
        """Create geographic distribution plots"""
        country_stats = df.groupby('Country').agg({
            'Financial Loss (in Million $)': 'sum',
            'Number of Affected Users': 'sum',
            'Attack Type': 'count'
        }).reset_index()
        
        country_stats.columns = ['Country', 'Total_Loss', 'Total_Users', 'Attack_Count']
        
        fig = px.bar(country_stats.sort_values('Total_Loss', ascending=False).head(15),
                    x='Country', y='Total_Loss',
                    title='Total Financial Loss by Country',
                    labels={'Total_Loss': 'Financial Loss (Million $)'})
        
        fig.update_xaxis(tickangle=45)
        return fig
    
    def plot_industry_analysis(self, df):
        """Create industry-wise analysis plots"""
        industry_stats = df.groupby('Target Industry').agg({
            'Financial Loss (in Million $)': ['mean', 'sum', 'count'],
            'Number of Affected Users': 'mean',
            'Incident Resolution Time (in Hours)': 'mean'
        }).round(2)
        
        industry_stats.columns = ['Avg_Loss', 'Total_Loss', 'Attack_Count', 
                                'Avg_Users', 'Avg_Resolution']
        industry_stats = industry_stats.reset_index()
        
        # Create subplot with multiple metrics
        fig = px.scatter(industry_stats, 
                        x='Avg_Loss', y='Avg_Resolution',
                        size='Attack_Count', color='Target Industry',
                        title='Industry Risk Analysis: Average Loss vs Resolution Time',
                        labels={
                            'Avg_Loss': 'Average Financial Loss (Million $)',
                            'Avg_Resolution': 'Average Resolution Time (Hours)'
                        })
        
        return fig
    
    def plot_correlation_matrix(self, df):
        """Create correlation matrix heatmap"""
        numerical_cols = [
            'Financial Loss (in Million $)', 'Number of Affected Users',
            'Incident Resolution Time (in Hours)', 'Year'
        ]
        
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Correlation Matrix of Key Metrics",
                       color_continuous_scale="RdBu")
        
        return fig
    
    def plot_attack_source_analysis(self, df):
        """Analyze attack sources and their characteristics"""
        source_stats = df.groupby('Attack Source').agg({
            'Financial Loss (in Million $)': 'mean',
            'Number of Affected Users': 'mean',
            'Incident Resolution Time (in Hours)': 'mean',
            'Attack Type': 'count'
        }).round(2).reset_index()
        
        source_stats.columns = ['Attack_Source', 'Avg_Loss', 'Avg_Users', 
                               'Avg_Resolution', 'Attack_Count']
        
        fig = px.bar(source_stats, x='Attack_Source', y='Avg_Loss',
                    title='Average Financial Loss by Attack Source',
                    labels={'Avg_Loss': 'Average Financial Loss (Million $)'})
        
        return fig

def generate_synthetic_data(base_df, n_samples=100):
    """Generate synthetic cybersecurity data for testing"""
    np.random.seed(42)
    
    synthetic_data = []
    
    for _ in range(n_samples):
        # Random selections from existing data distributions
        country = np.random.choice(base_df['Country'].unique())
        attack_type = np.random.choice(base_df['Attack Type'].unique())
        industry = np.random.choice(base_df['Target Industry'].unique())
        source = np.random.choice(base_df['Attack Source'].unique())
        vulnerability = np.random.choice(base_df['Security Vulnerability Type'].unique())
        defense = np.random.choice(base_df['Defense Mechanism Used'].unique())
        
        # Generate realistic numerical values based on distributions
        year = np.random.choice(range(2020, 2025))
        
        # Financial loss with some correlation to attack type
        base_loss = base_df[base_df['Attack Type'] == attack_type]['Financial Loss (in Million $)'].mean()
        financial_loss = max(1, np.random.normal(base_loss, base_loss * 0.3))
        
        # Affected users
        base_users = base_df[base_df['Attack Type'] == attack_type]['Number of Affected Users'].mean()
        affected_users = max(1000, int(np.random.normal(base_users, base_users * 0.4)))
        
        # Resolution time
        base_resolution = base_df[base_df['Defense Mechanism Used'] == defense]['Incident Resolution Time (in Hours)'].mean()
        resolution_time = max(1, int(np.random.normal(base_resolution, base_resolution * 0.3)))
        
        synthetic_data.append({
            'Country': country,
            'Year': year,
            'Attack Type': attack_type,
            'Target Industry': industry,
            'Financial Loss (in Million $)': round(financial_loss, 2),
            'Number of Affected Users': affected_users,
            'Attack Source': source,
            'Security Vulnerability Type': vulnerability,
            'Defense Mechanism Used': defense,
            'Incident Resolution Time (in Hours)': resolution_time
        })
    
    return pd.DataFrame(synthetic_data)
