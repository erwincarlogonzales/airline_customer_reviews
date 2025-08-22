import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import config

from src.preprocessing import create_preprocessing_pipeline
from src.categorical_encoder import categorical_encoder

# Define st_shap to embed SHAP plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)
    
@st.cache_resource
# Function to load models
def load_models():
    lgb_model = joblib.load(config.MODEL_LGBM_PATH)
    dec_tree = joblib.load(config.MODEL_DEC_TREE_PATH)
    
    return lgb_model, dec_tree

@st.cache_data
# Function to load and fit the preprocessor
def load_and_fit_preprocessor(file_path):
    flight_df = pd.read_csv(file_path)
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(flight_df)
    
    return preprocessor, flight_df

# Function to collect user inputs
def user_input_features():
    inputs = {
        # feature: st.sidebar.slider(app label, min, max, start)
        'Online boarding': st.sidebar.slider('Online Boarding 🎫', 1, 5, 3),
        'Inflight wifi service': st.sidebar.slider('Inflight WIFI service 🛜', 1, 5, 3),
        'Inflight entertainment': st.sidebar.slider('Inflight Entertainment 🎮', 1, 5, 3),
        'Checkin service': st.sidebar.slider('Check-In service ✅', 1, 5, 3),
        'Seat comfort': st.sidebar.slider('Seat Comfort 💺', 1, 5, 3),
        'Age': st.sidebar.number_input('Age 🎂', 0, 100, 18),
        'Flight Distance': st.sidebar.number_input('Flight Distance 🛫', 0, 10000, 100),
        'Business Travel': st.sidebar.selectbox('Business Travel 💼', ['Yes', 'No']),
        'Loyal Customer': st.sidebar.selectbox('Loyal Customer 🤞', ['Yes', 'No']),
        'Class': st.sidebar.selectbox('Class 👜', ['Eco', 'Eco Plus', 'Business'])
    }
    
    return pd.DataFrame(inputs, index=[0])

# Make predictions and explain the results
def make_predictions(model, preprocessor, input_df, explainer, lime_explainer):
    try:
        # Transform user inputs using the preprocessor
        input_df_transformed = preprocessor.transform(categorical_encoder(input_df))
        prediction = model.predict(input_df_transformed)
        
        # Get SHAP values
        shap_values = explainer.shap_values(input_df_transformed)
        
        # If shap_values is a list (for classification), use the appropriate class
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Assuming binary class, use the second class 1 (Not Satisfied) 0 for (Satisfied)
            
        # LIME explanation
        lime_explanation = lime_explainer.explain_instance(
            data_row = input_df_transformed[0],
            predict_fn = model.predict_proba,
            num_features = config.LIME_NUM_FEATURES
        )
        
        return 'Satisfied 🤩' if prediction[0] else 'Not Satisfied 😔', shap_values, explainer.expected_value, input_df_transformed, lime_explanation
    
    except Exception as error:
        st.error(f'Error occurred during prediction: {str(error)}. Please contact support.')
        
        return None, None, None, None, None
    
# Display logo and hero image
st.sidebar.image('images/logo.png', width = 100)
st.image('images/hero.jpg', use_container_width = True)

st.title('Cebu Pacific Customer Flight Satisfaction ✈️')
st.header('Please Tell Us About Your Experience! 🤩')

# Load models and preprocessor
lgb_model, dec_tree = load_models()
preprocessor, flight_df = load_and_fit_preprocessor(config.DATA_PATH)

# Model selection
model_choice = st.sidebar.selectbox('Choose Model 🛠️', ['LightGBM', 'Decision Tree'])
selected_model = lgb_model if model_choice == 'LightGBM' else dec_tree

# Initialize SHAP and LIME Explainers
explainer = shap.TreeExplainer(selected_model)
lime_explainer = LimeTabularExplainer(
    training_data = preprocessor.transform(categorical_encoder(flight_df)),
    feature_names = flight_df.columns,
    mode = 'classification'
)

# Add app description
st.write("""
    This app predicts flight satisfaction based on user inputs.
    You can select between two machine learning models: LightGBM and Decision Tree ✨
""")

# Collect user inputs
st.sidebar.header('User Input')
input_df = user_input_features()

# Display user inputs
st.write('### User Inputs')
for key, value in input_df.loc[0].items():
    st.write(f'**{key}**: {value}')
    
# Predict button
if st.button('Analyze My Experience ✨'):
    result, shap_values, expected_value, input_df_transformed, lime_explanation = make_predictions(
        selected_model, preprocessor, input_df, explainer, lime_explainer
    )
    if result:
        st.write(f"## Prediction: You are likely **{result}**")
        
        st.write("---")
        
        col1, col2 = st.columns(2)

        with col1:
            # --- SHAP PLOT LOGIC ---
            st.write('### What Drove This Prediction? (SHAP Analysis)')
            st.info("This plot shows which features pushed the prediction higher (red) or lower (blue).", icon="💡")
            
            # Handle the case where expected_value can be a list (from LightGBM) or a single float (from sklearn DecisionTree)
            if isinstance(expected_value, (list, np.ndarray)):
                # If it's a list, we're interested in the base value for the "Not Satisfied" class (index 1)
                base_value = expected_value[1]
            else:
                # If it's a single number, just use it
                base_value = expected_value
            
            # Plot the first prediction's explanation
            force_plot = shap.force_plot(base_value, shap_values[0,:], features=input_df.iloc[0,:])
            st_shap(force_plot, 200)
        
        with col2:
            st.write('### Top Factors (LIME Explanation)')
            st.info("This chart shows the top features supporting or contradicting the prediction.", icon="💡")
            lime_html = lime_explanation.as_html()
            st.components.v1.html(lime_html, height=400)