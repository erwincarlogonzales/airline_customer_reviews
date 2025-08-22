import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import config
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import create_preprocessing_pipeline
from src.categorical_encoder import categorical_encoder

# Page config with custom styling
st.set_page_config(
    page_title="✈️ Flight Satisfaction Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main background and fonts */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        color: white;
        text-align: center;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Input section styling */
    .input-section {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Prediction result styling */
    .prediction-box {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .satisfied {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .not-satisfied {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Animation for buttons */
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }
    
    /* Progress indicators */
    .progress-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Define st_shap to embed SHAP plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

@st.cache_resource
def load_models():
    lgb_model = joblib.load(config.MODEL_LGBM_PATH)
    dec_tree = joblib.load(config.MODEL_DEC_TREE_PATH)
    return lgb_model, dec_tree

@st.cache_data
def load_and_fit_preprocessor(file_path):
    flight_df = pd.read_csv(file_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(flight_df)
    return preprocessor, flight_df

def aggregate_shap_to_original_features(shap_values, original_feature_names, preprocessor=None):
    """
    Aggregate SHAP values from preprocessed features back to original features
    """
    # Flatten SHAP values if needed
    shap_flat = shap_values.flatten() if len(shap_values.shape) > 1 else shap_values
    
    # Try to get processed feature names
    try:
        if hasattr(preprocessor, 'get_feature_names_out'):
            processed_names = list(preprocessor.get_feature_names_out())
        elif hasattr(preprocessor, 'feature_names_'):
            processed_names = list(preprocessor.feature_names_)
        else:
            # Fallback: create generic names
            processed_names = [f"feature_{i}" for i in range(len(shap_flat))]
    except:
        processed_names = [f"feature_{i}" for i in range(len(shap_flat))]
    
    # Create aggregated SHAP values
    aggregated_shap = {}
    
    # Known categorical features that get one-hot encoded
    categorical_features = ['Business Travel', 'Loyal Customer', 'Class']
    numerical_features = ['Online boarding', 'Inflight wifi service', 'Inflight entertainment', 
                         'Checkin service', 'Seat comfort', 'Age', 'Flight Distance']
    
    # Initialize all original features with 0
    for feature in original_feature_names:
        aggregated_shap[feature] = 0.0
    
    # Map processed features back to original features
    for i, processed_name in enumerate(processed_names):
        if i >= len(shap_flat):
            break
            
        shap_value = shap_flat[i]
        
        # Find which original feature this processed feature belongs to
        matched = False
        
        # Check for exact matches (numerical features)
        for original_feature in original_feature_names:
            if original_feature.lower().replace(' ', '_') in processed_name.lower() or \
               processed_name.lower().replace('_', ' ') == original_feature.lower():
                aggregated_shap[original_feature] += shap_value
                matched = True
                break
        
        # Check for categorical feature patterns
        if not matched:
            for cat_feature in categorical_features:
                if cat_feature.lower().replace(' ', '_') in processed_name.lower():
                    aggregated_shap[cat_feature] += shap_value
                    matched = True
                    break
        
        # If still not matched, try partial matching
        if not matched:
            for original_feature in original_feature_names:
                # Remove common words and check for partial matches
                original_clean = original_feature.lower().replace(' ', '').replace('_', '')
                processed_clean = processed_name.lower().replace(' ', '').replace('_', '')
                
                if any(word in processed_clean for word in original_clean.split()) or \
                   any(word in original_clean for word in processed_clean.split()):
                    aggregated_shap[original_feature] += shap_value
                    matched = True
                    break
        
        # If still no match, add to the most similar feature or create "Other"
        if not matched:
            if "Other" not in aggregated_shap:
                aggregated_shap["Other"] = 0.0
            aggregated_shap["Other"] += shap_value
    
    # Convert to lists for DataFrame creation
    features = list(aggregated_shap.keys())
    values = list(aggregated_shap.values())
    
    # Remove features with zero contribution (likely unused one-hot categories)
    non_zero_features = []
    non_zero_values = []
    for f, v in zip(features, values):
        if abs(v) > 1e-6:  # Keep only non-zero contributions
            non_zero_features.append(f)
            non_zero_values.append(v)
    
    return non_zero_features, non_zero_values

def create_service_radar_chart(input_df):
    """Create a radar chart showing service quality metrics"""
    services = ['Online boarding', 'Inflight wifi service', 'Inflight entertainment', 'Checkin service', 'Seat comfort']
    values = [input_df.loc[0, service] for service in services]
    
    # Create figure and polar subplot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Compute angle for each axis
    angles = np.linspace(0, 2*np.pi, len(services), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    values += values[:1]  # Complete the circle
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, color='#667eea')
    ax.fill(angles, values, alpha=0.25, color='#667eea')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([s.replace(' ', '\n') for s in services])
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(True)
    
    plt.title('Service Quality Ratings', size=16, fontweight='bold', pad=20)
    return fig
    """Create a radar chart showing service quality metrics"""
    services = ['Online boarding', 'Inflight wifi service', 'Inflight entertainment', 'Checkin service', 'Seat comfort']
    values = [input_df.loc[0, service] for service in services]
    
    # Create figure and polar subplot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Compute angle for each axis
    angles = np.linspace(0, 2*np.pi, len(services), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    values += values[:1]  # Complete the circle
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, color='#667eea')
    ax.fill(angles, values, alpha=0.25, color='#667eea')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([s.replace(' ', '\n') for s in services])
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(True)
    
    plt.title('Service Quality Ratings', size=16, fontweight='bold', pad=20)
    return fig

def create_feature_importance_chart(shap_values, feature_names, preprocessor=None):
    """Create a horizontal bar chart for feature importance"""
    # Use the aggregation function to map back to original features
    agg_features, agg_values = aggregate_shap_to_original_features(shap_values, feature_names, preprocessor)
    
    # Create DataFrame and sort by absolute importance
    importance_df = pd.DataFrame({
        'Feature': agg_features,
        'Importance': [abs(v) for v in agg_values]
    }).sort_values('Importance', ascending=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], 
                   color='#667eea', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Feature Importance (Absolute SHAP Value)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Impact on Prediction', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def user_input_features():
    """Collect user inputs with modern styling"""
    
    # Sidebar header with logo space
    st.sidebar.markdown("### ✈️ Flight Experience Input")
    st.sidebar.markdown("---")
    
    # Service Quality Metrics
    st.sidebar.markdown("#### 🛎️ **Service Quality**")
    online_boarding = st.sidebar.slider('Online Boarding', 1, 5, 3, help="Rate your online boarding experience")
    wifi_service = st.sidebar.slider('Inflight WIFI', 1, 5, 3, help="Rate the WiFi quality")
    entertainment = st.sidebar.slider('Inflight Entertainment', 1, 5, 3, help="Rate the entertainment options")
    checkin = st.sidebar.slider('Check-In Service', 1, 5, 3, help="Rate the check-in process")
    seat_comfort = st.sidebar.slider('Seat Comfort', 1, 5, 3, help="Rate your seat comfort")
    
    st.sidebar.markdown("#### 👤 **Personal Information**")
    age = st.sidebar.number_input('Age', 0, 100, 25, help="Your age")
    flight_distance = st.sidebar.number_input('Flight Distance (miles)', 0, 10000, 1000, help="Distance of your flight")
    
    st.sidebar.markdown("#### 🎫 **Travel Details**")
    business_travel = st.sidebar.selectbox('Business Travel', ['Yes', 'No'], help="Is this a business trip?")
    loyal_customer = st.sidebar.selectbox('Loyal Customer', ['Yes', 'No'], help="Are you a loyal customer?")
    travel_class = st.sidebar.selectbox('Class', ['Eco', 'Eco Plus', 'Business'], help="Your travel class")
    
    inputs = {
        'Online boarding': online_boarding,
        'Inflight wifi service': wifi_service,
        'Inflight entertainment': entertainment,
        'Checkin service': checkin,
        'Seat comfort': seat_comfort,
        'Age': age,
        'Flight Distance': flight_distance,
        'Business Travel': business_travel,
        'Loyal Customer': loyal_customer,
        'Class': travel_class
    }
    
    return pd.DataFrame(inputs, index=[0])

def make_predictions(model, preprocessor, input_df, explainer, lime_explainer):
    try:
        input_df_transformed = preprocessor.transform(categorical_encoder(input_df))
        prediction = model.predict(input_df_transformed)
        
        shap_values = explainer.shap_values(input_df_transformed)
        
        # Handle different SHAP value formats for different models
        if isinstance(shap_values, list):
            # For models that return a list (like some LightGBM configurations)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Ensure shap_values is 2D for consistent indexing
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
            
        lime_explanation = lime_explainer.explain_instance(
            data_row = input_df_transformed[0],
            predict_fn = model.predict_proba,
            num_features = config.LIME_NUM_FEATURES
        )
        
        return 'Satisfied' if prediction[0] else 'Not Satisfied', shap_values, explainer.expected_value, input_df_transformed, lime_explanation
    
    except Exception as error:
        st.error(f'❌ Error occurred: {str(error)}')
        return None, None, None, None, None

# =========================
# MAIN APP LAYOUT
# =========================

# Hero Header
st.markdown("""
<div class="main-header">
    <h1>🛫 Cebu Pacific Flight Satisfaction Analytics</h1>
    <p style="font-size: 1.2em; margin-top: 1rem; opacity: 0.9;">
        Predict and analyze your flight satisfaction using AI-powered insights
    </p>
</div>
""", unsafe_allow_html=True)

# Load models and data
lgb_model, dec_tree = load_models()
preprocessor, flight_df = load_and_fit_preprocessor(config.DATA_PATH)

# Model selection with better UI
col_model, col_info = st.columns([1, 2])

with col_model:
    st.markdown("### 🛠️ Select AI Model")
    model_choice = st.selectbox('', ['LightGBM', 'Decision Tree'], 
                               help="Choose between LightGBM (advanced) or Decision Tree (interpretable)")
    selected_model = lgb_model if model_choice == 'LightGBM' else dec_tree

with col_info:
    st.markdown("### 📊 Model Information")
    if model_choice == 'LightGBM':
        st.info("🚀 **LightGBM**: Advanced gradient boosting model with high accuracy and fast predictions")
    else:
        st.info("🌳 **Decision Tree**: Highly interpretable model that shows clear decision paths")

# Initialize explainers
explainer = shap.TreeExplainer(selected_model)
lime_explainer = LimeTabularExplainer(
    training_data = preprocessor.transform(categorical_encoder(flight_df)),
    feature_names = flight_df.columns,
    mode = 'classification'
)

# Collect user inputs
input_df = user_input_features()

# Main content area
st.markdown("### 📋 Your Flight Experience Summary")

# Display inputs in a nice layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>🛎️ Service Ratings</h4>
        <p><strong>Online Boarding:</strong> {}/5</p>
        <p><strong>WiFi Service:</strong> {}/5</p>
        <p><strong>Entertainment:</strong> {}/5</p>
        <p><strong>Check-in:</strong> {}/5</p>
        <p><strong>Seat Comfort:</strong> {}/5</p>
    </div>
    """.format(
        input_df.loc[0, 'Online boarding'],
        input_df.loc[0, 'Inflight wifi service'],
        input_df.loc[0, 'Inflight entertainment'],
        input_df.loc[0, 'Checkin service'],
        input_df.loc[0, 'Seat comfort']
    ), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>👤 Personal Details</h4>
        <p><strong>Age:</strong> {} years</p>
        <p><strong>Flight Distance:</strong> {} miles</p>
        <p><strong>Business Travel:</strong> {}</p>
        <p><strong>Loyal Customer:</strong> {}</p>
        <p><strong>Travel Class:</strong> {}</p>
    </div>
    """.format(
        input_df.loc[0, 'Age'],
        input_df.loc[0, 'Flight Distance'],
        input_df.loc[0, 'Business Travel'],
        input_df.loc[0, 'Loyal Customer'],
        input_df.loc[0, 'Class']
    ), unsafe_allow_html=True)

with col3:
    st.markdown("#### 📊 Service Quality Overview")
    
    # Calculate average service rating
    avg_service = np.mean([
        input_df.loc[0, 'Online boarding'],
        input_df.loc[0, 'Inflight wifi service'],
        input_df.loc[0, 'Inflight entertainment'],
        input_df.loc[0, 'Checkin service'],
        input_df.loc[0, 'Seat comfort']
    ])
    
    # Display as a metric with progress bar
    st.metric("Average Service Rating", f"{avg_service:.1f}/5")
    st.progress(avg_service / 5)
    
    # Color-coded feedback
    if avg_service >= 4:
        st.success("🌟 Excellent service experience!")
    elif avg_service >= 3:
        st.info("👍 Good service experience")
    else:
        st.warning("⚠️ Room for improvement")

# Prediction section
st.markdown("---")
col_predict, col_space = st.columns([2, 1])

with col_predict:
    if st.button('🔮 Analyze My Flight Experience', help="Click to get AI-powered satisfaction prediction"):
        with st.spinner('🤖 AI is analyzing your experience...'):
            result, shap_values, expected_value, input_df_transformed, lime_explanation = make_predictions(
                selected_model, preprocessor, input_df, explainer, lime_explainer
            )
            
            if result:
                # Prediction result with dynamic styling
                prediction_class = "satisfied" if result == "Satisfied" else "not-satisfied"
                icon = "🤩" if result == "Satisfied" else "😔"
                
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    <h2>{icon} Prediction: You are likely <strong>{result}</strong> {icon}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Analysis section
                st.markdown("## 🔍 Detailed Analysis")
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs(["📊 SHAP Analysis", "🔍 LIME Explanation", "📈 Feature Impact", "🎯 Service Radar"])
                
                with tab1:
                    st.markdown("### 🎯 What Drove This Prediction?")
                    
                    if model_choice == 'LightGBM':
                        st.info("💡 This visualization shows which features pushed the prediction towards satisfied (blue) or not satisfied (red).")
                        
                        # Handle expected value properly for LightGBM
                        if isinstance(expected_value, (list, np.ndarray)):
                            base_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
                        else:
                            base_value = expected_value
                        
                        # For LightGBM, use the standard SHAP force plot
                        shap_values_for_plot = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                        force_plot = shap.force_plot(base_value, shap_values_for_plot, features=input_df.iloc[0,:])
                        st_shap(force_plot, 300)
                    
                    else:  # Decision Tree
                        st.info("💡 For Decision Trees, we show feature contributions directly. Red bars decrease satisfaction, blue bars increase it.")
                        
                        # Use the aggregation function to map SHAP values back to original features
                        original_features = list(input_df.columns)
                        agg_features, agg_values = aggregate_shap_to_original_features(shap_values, original_features, preprocessor)
                        
                        contrib_df = pd.DataFrame({
                            'Feature': agg_features,
                            'Contribution': agg_values
                        }).sort_values('Contribution', key=abs, ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        colors = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in contrib_df['Contribution']]
                        bars = ax.barh(contrib_df['Feature'], contrib_df['Contribution'], color=colors, alpha=0.8)
                        
                        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
                        ax.set_title('Feature Contributions to Prediction', fontsize=14, fontweight='bold')
                        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
                        ax.grid(axis='x', alpha=0.3)
                        
                        # Add value labels
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(width + (0.001 if width >= 0 else -0.001), bar.get_y() + bar.get_height()/2, 
                                   f'{width:.3f}', ha='left' if width >= 0 else 'right', 
                                   va='center', fontweight='bold', fontsize=10)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add explanation
                        st.markdown("""
                        **How to read this chart:**
                        - 🔴 **Red bars (negative values)**: Features that decrease satisfaction likelihood
                        - 🔵 **Blue bars (positive values)**: Features that increase satisfaction likelihood  
                        - **Longer bars**: More important features for this prediction
                        
                        *Note: Values from related preprocessed features (like one-hot encoded categories) have been aggregated back to original feature names.*
                        """)
                
                with tab2:
                    st.markdown("### 🔍 Alternative Explanation (LIME)")
                    st.info("💡 This shows the top features that support or contradict the prediction with different methodology.")
                    
                    lime_html = lime_explanation.as_html()
                    st.components.v1.html(lime_html, height=500)
                
                with tab3:
                    st.markdown("### 📈 Feature Importance Breakdown")
                    
                    # Create a feature importance visualization
                    feature_names = input_df.columns
                    importance_fig = create_feature_importance_chart(shap_values, feature_names, preprocessor)
                    st.pyplot(importance_fig)
                    
                    # Show top 3 most important features using aggregated values
                    original_features = list(input_df.columns)
                    agg_features, agg_values = aggregate_shap_to_original_features(shap_values, original_features, preprocessor)
                    
                    importance_df = pd.DataFrame({
                        'Feature': agg_features,
                        'Importance': [abs(v) for v in agg_values]
                    }).sort_values('Importance', ascending=False)
                    
                    st.markdown("#### 🏆 Top 3 Most Important Features:")
                    for i, (_, row) in enumerate(importance_df.head(3).iterrows()):
                        st.markdown(f"**{i+1}. {row['Feature']}** - Impact: {row['Importance']:.3f}")
                
                with tab4:
                    st.markdown("### 🎯 Service Quality Radar")
                    st.info("💡 Visual representation of your service ratings across all categories.")
                    
                    radar_fig = create_service_radar_chart(input_df)
                    st.pyplot(radar_fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🛫 Built with ❤️ using Streamlit • Powered by AI & Machine Learning</p>
    <p style="font-size: 0.9em;">Experience the future of customer satisfaction analytics</p>
</div>
""", unsafe_allow_html=True)