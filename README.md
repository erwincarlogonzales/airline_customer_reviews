# Airline Customer Satisfaction Predictor ✈️

<div align="center">

**A web application built with Streamlit that predicts airline customer satisfaction and explains the *why* behind each prediction using XAI.**

</div>

**[👉 Check out the Tableau dashboard here](https://public.tableau.com/app/profile/erwincarlogonzales/viz/Book5_17312434149340/Reviews)**

**[👉 Check out the Streamlit app here](https://erwincarlogonzales-airline-customer-reviews-app-l8h2pd.streamlit.app/)**

---

<div align="center">
  
![App Screenshot](images/hero.jpg)

</div>

---

## 🚀 Core Features

-   **🔮 Real-Time Predictions:** Instantly predict customer satisfaction by inputting flight details like seat comfort, inflight wifi, and passenger age via an intuitive sidebar.

-   **🤖 Dual Model Selection:** Switch between two powerful, pre-trained machine learning models:
    -   **LightGBM:** A fast, high-performance gradient-boosting framework.
    -   **Decision Tree:** A classic, highly interpretable model.

-   **🔍 Explainable AI (XAI):** Go beyond the prediction and understand *why* a decision was made with integrated SHAP and LIME plots.
    -   **SHAP:** See which features pushed the prediction towards "Satisfied" or "Dissatisfied."
    -   **LIME:** Pinpoint the top features that support or contradict the prediction for a specific customer.

-   **🎨 Dynamic & Responsive UI:** Built with Streamlit for a seamless user experience on any device.

## 🛠️ Tech Stack & Pipeline

The application follows a standard machine learning workflow from data preprocessing to model explanation.

**Pipeline:**
`User Input` ➔ `Data Preprocessing` ➔ `Model Prediction` ➔ `XAI Explanation (SHAP/LIME)`

| Category          | Technology                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------- |
| **Web Framework** | `Streamlit`                                                                                                   |
| **Backend & ML** | `Python`, `Scikit-learn`, `LightGBM`                                                                            |
| **Data Handling** | `Pandas`, `NumPy`                                                                                             |
| **Explainable AI**| `SHAP`, `LIME`                                                                                                |
| **Dependencies** | `Poetry`                                                                                                      |

## 📂 Project Structure

```bash
erwincarlogonzales-airline_customer_reviews/
│
├── .streamlit/             # Streamlit configuration
├── data/                   # Dataset CSV
│   └── app_flight_df.csv
├── images/                 # App images and logos
│   ├── hero.jpg
│   └── logo.png
├── models/                 # Saved machine learning models
│   ├── dec_tree.joblib
│   └── lgb_model.joblib
├── src/                    # Source code modules
│   ├── init.py
│   ├── categorical_encoder.py
│   └── preprocessing.py
├── app.py                  # Main Streamlit application file
├── config.py               # Configuration file for paths
├── LICENSE                 # Project license
├── pyproject.toml          # Poetry dependency definitions
└── README.md               # You are here
```

## 🏁 Getting Started

To run this project locally, you'll need to have **Python** and **Poetry** installed.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```
### 2. Install Dependencies
This command creates a virtual environment and installs all required packages from pyproject.toml.

```bash
poetry install
```

### 3. Run the App
First, activate the virtual environment managed by Poetry, then launch the Streamlit app.

```bash
poetry shell
streamlit run app.py
```

Your browser should automatically open to http://localhost:8501.

### 📄 License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.