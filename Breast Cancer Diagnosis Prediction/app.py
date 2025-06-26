import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Breast Cancer Diagnosis Prediction", layout="wide")

# Define CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #FF5675;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #FF5675;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.info-text {
    font-size: 1rem;
    color: #555;
}
.success-box {
    padding: 1rem;
    background-color: #D5F5E3;
    border-radius: 5px;
    margin-bottom: 1rem;
}
.warning-box {
    padding: 1rem;
    background-color: #FDEDEC;
    border-radius: 5px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">Breast Cancer Diagnosis Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="info-text">This application helps in predicting whether a breast cancer tumor is malignant or benign based on various features extracted from digitized images of fine needle aspirates (FNA) of breast masses.</p>', unsafe_allow_html=True)

# Class for cancer model
class CancerModel:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        # Drop unnecessary columns
        if 'Unnamed: 32' in self.data.columns:
            self.data = self.data.drop('Unnamed: 32', axis=1)
        if 'id' in self.data.columns:
            self.data = self.data.drop('id', axis=1)
        
        # Check for missing values
        self.data = self.data.dropna()
        
        # Encode target variable
        self.data['diagnosis'] = self.label_encoder.fit_transform(self.data['diagnosis'])
        
        # Separate features and target
        self.y = self.data['diagnosis']
        self.X = self.data.drop('diagnosis', axis=1)
        
        return self.data
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, kernel='rbf', C=1.0, gamma='scale'):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.model.fit(self.X_train, self.y_train)
        return self.model
    
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        return accuracy, report, cm
    
    def predict(self, features):
        # Scale the input features
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)
        probability = self.model.predict_proba(features_scaled)[0]
        return prediction[0], probability

# Initialize session state
if 'cancer_model' not in st.session_state:
    st.session_state.cancer_model = CancerModel()
    st.session_state.data_loaded = False
    st.session_state.model_trained = False

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Prediction"])

# Load data
if not st.session_state.data_loaded:
    try:
        st.session_state.cancer_model.load_data("cancer.csv")
        st.session_state.data_loaded = True
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Data Exploration Page
if page == "Data Exploration":
    st.markdown('<h2 class="sub-header">Data Exploration</h2>', unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
        data = st.session_state.cancer_model.data
        
        # Display basic information
        st.markdown('<h3 class="sub-header">Dataset Overview</h3>', unsafe_allow_html=True)
        st.write(f"Number of samples: {data.shape[0]}")
        st.write(f"Number of features: {data.shape[1]-1}")
        
        # Display class distribution
        st.markdown('<h3 class="sub-header">Class Distribution</h3>', unsafe_allow_html=True)
        diagnosis_counts = data['diagnosis'].value_counts()
        fig = px.pie(values=diagnosis_counts.values, 
                    names=['Benign', 'Malignant'] if diagnosis_counts.index[0] == 0 else ['Malignant', 'Benign'],
                    title='Distribution of Benign vs Malignant Cases',
                    color_discrete_sequence=['#3498db', '#e74c3c'])
        st.plotly_chart(fig)
        
        # Feature correlation
        st.markdown('<h3 class="sub-header">Feature Correlation</h3>', unsafe_allow_html=True)
        corr = data.corr()
        fig = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)
        
        # Feature distribution by diagnosis
        st.markdown('<h3 class="sub-header">Feature Distribution by Diagnosis</h3>', unsafe_allow_html=True)
        feature_col = st.selectbox("Select Feature", data.drop('diagnosis', axis=1).columns)
        
        fig = px.histogram(data, x=feature_col, color='diagnosis', 
                          color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                          labels={'diagnosis': 'Diagnosis', '0': 'Benign', '1': 'Malignant'},
                          marginal="box", opacity=0.7,
                          barmode='overlay')
        fig.update_layout(title=f"Distribution of {feature_col} by Diagnosis")
        st.plotly_chart(fig)
        
        # Show raw data
        if st.checkbox("Show Raw Data"):
            st.write(data)
    else:
        st.warning("Data not loaded. Please check the file path.")

# Model Training Page
elif page == "Model Training":
    st.markdown('<h2 class="sub-header">Model Training</h2>', unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
        # Model parameters
        st.markdown('<h3 class="sub-header">Model Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            kernel = st.selectbox("Kernel", options=["rbf", "linear", "poly", "sigmoid"], index=0)
            C = st.slider("C (Regularization parameter)", 0.01, 10.0, 1.0, 0.01)
        with col2:
            gamma = st.selectbox("Gamma", options=["scale", "auto"], index=0)
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Preprocess data
                X_train, X_test, y_train, y_test = st.session_state.cancer_model.preprocess_data(test_size=test_size)
                
                # Train model
                st.session_state.cancer_model.train_model(kernel=kernel, C=C, gamma=gamma)
                
                # Evaluate model
                accuracy, report, cm = st.session_state.cancer_model.evaluate_model()
                st.session_state.model_trained = True
                st.session_state.accuracy = accuracy
                st.session_state.report = report
                st.session_state.cm = cm
            
            st.success(f"Model trained successfully! Accuracy: {accuracy:.4f}")
        
        # Display evaluation results if model is trained
        if st.session_state.model_trained:
            st.markdown('<h3 class="sub-header">Model Evaluation</h3>', unsafe_allow_html=True)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{st.session_state.accuracy:.4f}")
            with col2:
                st.metric("Precision (Malignant)", f"{st.session_state.report['1']['precision']:.4f}")
            with col3:
                st.metric("Recall (Malignant)", f"{st.session_state.report['1']['recall']:.4f}")
            
            # Display confusion matrix
            st.markdown('<h4 class="sub-header">Confusion Matrix</h4>', unsafe_allow_html=True)
            cm = st.session_state.cm
            fig = px.imshow(cm, text_auto=True, 
                          labels=dict(x="Predicted Label", y="True Label"),
                          x=['Benign', 'Malignant'],
                          y=['Benign', 'Malignant'],
                          color_continuous_scale='Blues')
            st.plotly_chart(fig)
            
            # Display classification report
            st.markdown('<h4 class="sub-header">Classification Report</h4>', unsafe_allow_html=True)
            report_df = pd.DataFrame(st.session_state.report).transpose()
            st.write(report_df)
    else:
        st.warning("Data not loaded. Please check the file path.")

# Prediction Page
elif page == "Prediction":
    st.markdown('<h2 class="sub-header">Cancer Diagnosis Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("Model not trained yet. Please go to the Model Training page first.")
    else:
        st.markdown('<p class="info-text">Enter the feature values to get a prediction.</p>', unsafe_allow_html=True)
        
        # Create feature input form with tabs for different feature groups
        tab1, tab2, tab3 = st.tabs(["Mean Values", "Standard Error Values", "Worst Values"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=50.0, value=14.0)
                texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=50.0, value=19.0)
                perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=92.0)
                area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=650.0)
                smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, value=0.1)
            with col2:
                compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=1.0, value=0.1)
                concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=1.0, value=0.1)
                concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=1.0, value=0.05)
                symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=1.0, value=0.2)
                fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=1.0, value=0.06)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                radius_se = st.number_input("Radius SE", min_value=0.0, max_value=5.0, value=0.4)
                texture_se = st.number_input("Texture SE", min_value=0.0, max_value=5.0, value=1.2)
                perimeter_se = st.number_input("Perimeter SE", min_value=0.0, max_value=20.0, value=2.5)
                area_se = st.number_input("Area SE", min_value=0.0, max_value=200.0, value=40.0)
                smoothness_se = st.number_input("Smoothness SE", min_value=0.0, max_value=0.1, value=0.006)
            with col2:
                compactness_se = st.number_input("Compactness SE", min_value=0.0, max_value=0.2, value=0.02)
                concavity_se = st.number_input("Concavity SE", min_value=0.0, max_value=0.2, value=0.02)
                concave_points_se = st.number_input("Concave Points SE", min_value=0.0, max_value=0.1, value=0.01)
                symmetry_se = st.number_input("Symmetry SE", min_value=0.0, max_value=0.1, value=0.02)
                fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, max_value=0.1, value=0.003)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                radius_worst = st.number_input("Radius Worst", min_value=0.0, max_value=50.0, value=16.0)
                texture_worst = st.number_input("Texture Worst", min_value=0.0, max_value=50.0, value=25.0)
                perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, max_value=300.0, value=100.0)
                area_worst = st.number_input("Area Worst", min_value=0.0, max_value=4000.0, value=800.0)
                smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, max_value=1.0, value=0.13)
            with col2:
                compactness_worst = st.number_input("Compactness Worst", min_value=0.0, max_value=2.0, value=0.25)
                concavity_worst = st.number_input("Concavity Worst", min_value=0.0, max_value=2.0, value=0.3)
                concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, max_value=1.0, value=0.15)
                symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, max_value=1.0, value=0.3)
                fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, max_value=1.0, value=0.08)
        
        # Create feature vector
        features = [
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
            compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
            radius_se, texture_se, perimeter_se, area_se, smoothness_se,
            compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
            radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
            compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
        ]
        
        # Make prediction
        if st.button("Predict"):
            prediction, probability = st.session_state.cancer_model.predict(features)
            
            # Display prediction
            if prediction == 1:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: #e74c3c;'>Prediction: Malignant</h3>", unsafe_allow_html=True)
                st.markdown(f"<p>Probability of being malignant: {probability[1]:.4f} ({probability[1]*100:.2f}%)</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: #2ecc71;'>Prediction: Benign</h3>", unsafe_allow_html=True)
                st.markdown(f"<p>Probability of being benign: {probability[0]:.4f} ({probability[0]*100:.2f}%)</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display probability gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability[1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability of Malignancy (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c"},
                    'steps': [
                        {'range': [0, 20], 'color': "#2ecc71"},
                        {'range': [20, 40], 'color': "#27ae60"},
                        {'range': [40, 60], 'color': "#f39c12"},
                        {'range': [60, 80], 'color': "#d35400"},
                        {'range': [80, 100], 'color': "#c0392b"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center'>Breast Cancer Diagnosis Prediction App | Created with Streamlit</p>", unsafe_allow_html=True)