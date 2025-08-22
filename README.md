Airline Customer Satisfaction Predictor ✈️A web application built with Streamlit that predicts airline customer satisfaction using machine learning. The app provides real-time predictions and uses SHAP and LIME to explain the factors driving each prediction, offering insights into the customer experience.✨ Live Demo: https://elly-ml-app.streamlit.app/<!-- It's a good idea to add a screenshot of your app here! -->🚀 FeaturesInteractive Prediction: Users can input various flight-related details (e.g., seat comfort, inflight wifi, age) through an intuitive sidebar.Dual Model Selection: Choose between two pre-trained machine learning models for prediction:LightGBM: A fast and efficient gradient-boosting framework.Decision Tree: A classic, interpretable model.Explainable AI (XAI): Understand why a prediction was made with integrated SHAP and LIME plots.SHAP (SHapley Additive exPlanations): Visualizes which features pushed the prediction towards "Satisfied" or "Not Satisfied".LIME (Local Interpretable Model-agnostic Explanations): Shows the top features that support or contradict the prediction for a specific case.Dynamic UI: The interface is built with Streamlit for a seamless and responsive user experience.⚙️ How It WorksThe application follows a standard machine learning pipeline:Data Loading & Preprocessing: The airline passenger dataset is loaded. A preprocessing pipeline handles scaling for numerical features and one-hot encoding for categorical features.Model Training: The LightGBM and Decision Tree models are pre-trained on the processed dataset.User Input: The Streamlit interface collects input from the user.Prediction: The selected model processes the user's input to predict whether the customer is satisfied.Explanation: SHAP and LIME explainers are used on the model's output to generate visualizations that break down the prediction.🛠️ Technologies UsedBackend & ML:PythonScikit-learn: For preprocessing pipelines and ML models.LightGBM: For the gradient boosting model.Pandas: For data manipulation.NumPy: For numerical operations.Web Framework:Streamlit: For building and deploying the interactive web app.Explainable AI:SHAP: For model explanation.LIME: For local model interpretability.Dependency Management:Poetry: For managing project dependencies and packaging.📂 Project Structureerwincarlogonzales-airline_customer_reviews/
│
├── .streamlit/              # Streamlit configuration (if any)
├── data/                    # CSV data used by the app
│   └── app_flight_df.csv
├── images/                  # Images and logos for the app
│   ├── hero.jpg
│   └── logo.png
├── models/                  # Saved machine learning models
│   ├── dec_tree.joblib
│   └── lgb_model.joblib
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── categorical_encoder.py
│   ├── preprocessing.py
│   └── ...
├── app.py                   # Main Streamlit application file
├── config.py                # Configuration file for paths
├── LICENSE                  # Project license
├── pyproject.toml           # Poetry dependency definitions
└── README.md                # This file
📦 Setup and InstallationTo run this project locally, you'll need to have Python and Poetry installed.Clone the repository:git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies using Poetry:This command will create a virtual environment and install all the packages listed in pyproject.toml.poetry install
▶️ How to Run the AppOnce the dependencies are installed, you can run the Streamlit application from your terminal.Activate the Poetry shell:poetry shell
Run the Streamlit app:streamlit run app.py
Your browser should automatically open to http://localhost:8501 where you can interact with the app.📄 LicenseThis project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.Feel free to reach out if you have any questions or suggestions!