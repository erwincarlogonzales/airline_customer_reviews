import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
import lime

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from lime.lime_tabular import LimeTabularExplainer

import os
import config

# Preprocessing pipeline
def create_preprocessing_pipeline():
    
    # Select numeric and categorical columns
    num_cols = make_column_selector(dtype_include='number')
    cat_cols = make_column_selector(dtype_include='object')
    
    # Instantiate the transformers
    scaler = StandardScaler()
    encoder = OneHotEncoder()
    knn_imputer = KNNImputer(n_neighbors=2, weights='uniform')
    
    # Create pipeline
    num_pipe = Pipeline([
        ('scaler', scaler),
        ('imputer', knn_imputer)
    ])
     
    cat_pipe = Pipeline([
        ('encoder', encoder)
    ])
    
    preprocessor = ColumnTransformer([
        ('numeric', num_pipe, num_cols),
        ('categorical', cat_pipe, cat_cols),
    ], remainder='drop')
    
    return preprocessor

# Encode categorical variables
def categorical_encoder(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = pd.Categorical(df[column]).codes
    
    return df

# Define st_shap to embed SHAP plots in Streamlit
def st_shap(plot):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html)
    
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
st.image('images/hero.jpg', use_column_width = True)

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


# Collect user inputs
st.sidebar.header('User Input')
input_df = user_input_features()

# Display user inputs
st.write('### User Inputs')
for key, value in input_df.loc[0].items():
    st.write(f'**{key}**: {value}')
    
# Predict button
if st.button('Make Prediction ✨'):
    result, shap_values, expected_value, input_df_transformed, lime_explanation = make_predictions(
        selected_model, preprocessor, input_df, explainer, lime_explainer
    )
    if result:
        st.write(f'### Prediction: {result}')
        
        # Display SHAP force plot
        # st.write('### SHAP Force Plot:')
        # st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], input_df_transformed[0]))
        
        # Display LIME explanation
        st.write('### LIME Explanation:')
        lime_html = lime_explanation.as_html()
        st.components.v1.html(lime_html)