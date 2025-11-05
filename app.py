import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class CyberAttackPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = ['Country', 'Year', 'Target Industry', 'Financial Loss (in Million $)', 
                              'Number of Affected Users', 'Attack Source', 'Security Vulnerability Type', 
                              'Defense Mechanism Used', 'Incident Resolution Time (in Hours)']
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset"""
        df = pd.read_csv(file_path)
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df['Attack Type']
        
        # Encode categorical variables
        categorical_columns = ['Country', 'Target Industry', 'Attack Source', 
                             'Security Vulnerability Type', 'Defense Mechanism Used']
        
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col])
        
        # Encode target variable
        self.label_encoders['Attack Type'] = LabelEncoder()
        y_encoded = self.label_encoders['Attack Type'].fit_transform(y)
        
        return X, y_encoded, df
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def predict_attack(self, input_features):
        """Predict attack type based on input features"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_features])
        
        # Encode categorical features
        categorical_columns = ['Country', 'Target Industry', 'Attack Source', 
                             'Security Vulnerability Type', 'Defense Mechanism Used']
        
        for col in categorical_columns:
            if col in input_df.columns:
                # Handle unseen labels
                try:
                    input_df[col] = self.label_encoders[col].transform([input_features[col]])[0]
                except ValueError:
                    # If label not seen during training, use the most common one
                    input_df[col] = 0
        
        # Make prediction
        prediction_encoded = self.model.predict(input_df)[0]
        prediction = self.label_encoders['Attack Type'].inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(input_df)[0]
        attack_types = self.label_encoders['Attack Type'].classes_
        
        return prediction, dict(zip(attack_types, probabilities))

def main():
    st.set_page_config(
        page_title="Cyber Attack Type Predictor",
        page_icon="🛡️",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
    }
    .probability-bar {
        background-color: #1f77b4;
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🛡️ Cyber Attack Type Predictor</h1>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = CyberAttackPredictor()
    
    # Load and train model
    try:
        with st.spinner('Loading data and training model...'):
            X, y_encoded, df = predictor.load_and_preprocess_data('Global_Cybersecurity_Threats_2015-2024 (3).csv')
            accuracy = predictor.train_model(X, y_encoded)
        
        st.success(f'Model trained successfully! Accuracy: {accuracy:.2%}')
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Get unique values for dropdowns
    countries = sorted(df['Country'].unique())
    industries = sorted(df['Target Industry'].unique())
    attack_sources = sorted(df['Attack Source'].unique())
    vulnerabilities = sorted(df['Security Vulnerability Type'].unique())
    defenses = sorted(df['Defense Mechanism Used'].unique())
    attack_types = sorted(df['Attack Type'].unique())
    
    # Create input form
    st.markdown("### Enter Attack Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox("Country", countries, index=countries.index('USA') if 'USA' in countries else 0)
        year = st.slider("Year", 2015, 2024, 2022)
        target_industry = st.selectbox("Target Industry", industries)
        financial_loss = st.number_input("Financial Loss (in Million $)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    
    with col2:
        affected_users = st.number_input("Number of Affected Users", min_value=0, max_value=1000000, value=500000, step=1000)
        attack_source = st.selectbox("Attack Source", attack_sources)
        vulnerability_type = st.selectbox("Security Vulnerability Type", vulnerabilities)
        defense_mechanism = st.selectbox("Defense Mechanism Used", defenses)
        resolution_time = st.slider("Incident Resolution Time (in Hours)", 1, 72, 24)
    
    # Prediction button
    if st.button("🔮 Predict Attack Type", use_container_width=True):
        # Prepare input features
        input_features = {
            'Country': country,
            'Year': year,
            'Target Industry': target_industry,
            'Financial Loss (in Million $)': financial_loss,
            'Number of Affected Users': affected_users,
            'Attack Source': attack_source,
            'Security Vulnerability Type': vulnerability_type,
            'Defense Mechanism Used': defense_mechanism,
            'Incident Resolution Time (in Hours)': resolution_time
        }
        
        try:
            # Make prediction
            prediction, probabilities = predictor.predict_attack(input_features)
            
            # Display results
            st.markdown("---")
            st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Predicted Attack Type: **{prediction}**")
            
            # Display probabilities
            st.markdown("#### Prediction Probabilities:")
            for attack_type, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                percentage = prob * 100
                st.write(f"{attack_type}: {percentage:.1f}%")
                st.progress(float(prob))
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show feature importance (optional)
            with st.expander("📊 Show Feature Importance"):
                feature_importance = pd.DataFrame({
                    'feature': predictor.feature_columns,
                    'importance': predictor.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                st.bar_chart(feature_importance.set_index('feature')['importance'])
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    # Dataset overview
    with st.expander("📈 Dataset Overview"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
            st.metric("Unique Attack Types", len(attack_types))
        
        with col2:
            st.metric("Countries", len(countries))
            st.metric("Industries", len(industries))
        
        with col3:
            avg_loss = df['Financial Loss (in Million $)'].mean()
            st.metric("Average Financial Loss", f"${avg_loss:.2f}M")
    
    # Attack type distribution
    with st.expander("📊 Attack Type Distribution"):
        attack_counts = df['Attack Type'].value_counts()
        st.bar_chart(attack_counts)

if __name__ == "__main__":
    main()