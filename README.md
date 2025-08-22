# ✈️ Airline Customer Satisfaction Predictor
**Real-Time ML Predictions with Explainable AI for Data-Driven Customer Experience Optimization**

<div align="center">

[![Machine Learning](https://img.shields.io/badge/ML-LightGBM_+_Decision_Tree-blue?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![Explainable AI](https://img.shields.io/badge/XAI-SHAP_+_LIME-purple?style=for-the-badge)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-Data_Science-green?style=for-the-badge&logo=python)](https://python.org)

*Interactive web application that predicts airline customer satisfaction with transparent, explainable AI insights for every decision*

**🚀 [LIVE DEMO](https://erwincarlogonzales-airline-customer-reviews-app-l8h2pd.streamlit.app/)**

**📊 [TABLEAU DASHBOARD](https://public.tableau.com/app/profile/erwincarlogonzales/viz/Book5_17312434149340/Reviews)**

</div>

---

## 🎯 **Project Impact**

This application bridges the gap between complex machine learning predictions and actionable business insights by providing **real-time customer satisfaction predictions** with **transparent explanations** of every decision made by the AI system.

### **The Business Challenge**
> *"How can airlines proactively identify and address factors that drive customer dissatisfaction before it impacts their business?"*

**The Solution**: An interactive ML-powered system that not only predicts satisfaction but explains the *why* behind each prediction, enabling data-driven customer experience improvements.

---

## 🔥 **Core Capabilities**

### **🎯 Dual-Model Architecture**
**Two complementary ML approaches for comprehensive analysis:**

| Model | Strength | Use Case | Performance Highlight |
|-------|----------|----------|----------------------|
| **LightGBM** | Perfect satisfied customer recall | Customer retention focus | 100% recall, 81% accuracy |
| **Decision Tree** | Superior overall performance | Production deployment | 86% accuracy, 97% precision |

### **🔍 Advanced Explainable AI Integration**
**Go beyond black-box predictions with comprehensive explanation frameworks:**

- **SHAP (SHapley Additive exPlanations)**: Quantifies each feature's contribution to the prediction
  - Feature importance rankings across entire dataset
  - Individual prediction breakdowns
  - Positive/negative impact visualization

- **LIME (Local Interpretable Model-agnostic Explanations)**: Instance-specific explanations
  - Local feature importance for individual customers
  - Confidence intervals for predictions
  - Counterfactual analysis capabilities

### **⚡ Real-Time Interactive Predictions**
**Instant satisfaction predictions through intuitive interface:**
- **Dynamic Input Controls**: Sidebar sliders and selectors for all customer touchpoints
- **Live Prediction Updates**: Real-time model inference as parameters change
- **Visual Feedback**: Color-coded satisfaction indicators and confidence scores
- **Export Capabilities**: Save predictions and explanations for reporting

### **📊 Comprehensive Feature Analysis**
**Deep insights into customer satisfaction drivers:**

#### **Top Satisfaction Drivers (LightGBM Analysis)**
1. **Age** (500+ importance) - Demographic preferences significantly impact satisfaction
2. **Inflight WiFi Service** (500+ importance) - Critical digital amenity expectation
3. **Flight Distance** (450+ importance) - Journey length affects overall experience
4. **Inflight Entertainment** (275+ importance) - Key differentiator for passenger experience
5. **Check-in Service** (250+ importance) - First impression sets satisfaction trajectory

#### **Correlation Insights**
- **Online Boarding** shows strongest positive correlation (0.51) with satisfaction
- **Business Travel** passengers exhibit higher satisfaction rates (0.45 correlation)
- **Class Upgrade** directly impacts satisfaction scores (0.49 correlation)
- **Service Quality Metrics** cluster together, indicating interconnected experience factors

---

## 🛠️ **Technical Excellence**

### **ML Pipeline Architecture**
```
Raw Data → Feature Engineering → Model Training → SHAP/LIME Integration → Web Interface
    ↓              ↓                   ↓                ↓                    ↓
Categorical    Preprocessing     LightGBM/Tree      Explanation         Streamlit
Encoding       Pipeline          Training           Generation          Deployment
```

### **Technology Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Framework** | Streamlit | Interactive UI and real-time updates |
| **ML Models** | LightGBM, Scikit-learn | High-performance prediction engines |
| **Explainable AI** | SHAP, LIME | Transparent decision explanations |
| **Data Processing** | Pandas, NumPy | Efficient data manipulation |
| **Visualization** | Matplotlib, Plotly | Interactive explanation plots |
| **Deployment** | Streamlit Cloud | Scalable web hosting |

### **Model Performance Metrics**

#### **Comparative Model Analysis**

| Metric | LightGBM | Decision Tree | Winner | Key Insight |
|--------|----------|---------------|---------|-------------|
| **Overall Accuracy** | 81% | 86% | 🏆 **Decision Tree** | +5% accuracy advantage |
| **Precision (Dissatisfied)** | 86% | 97% | 🏆 **Decision Tree** | Superior false positive control |
| **Precision (Satisfied)** | 73% | 79% | 🏆 **Decision Tree** | Better satisfied customer identification |
| **Recall (Dissatisfied)** | 63% | 73% | 🏆 **Decision Tree** | Catches more dissatisfied customers |
| **Recall (Satisfied)** | 100% | 98% | 🏆 **LightGBM** | Nearly perfect satisfied detection |
| **F1-Score (Dissatisfied)** | 77% | 84% | 🏆 **Decision Tree** | Better balanced performance |
| **F1-Score (Satisfied)** | 84% | 87% | 🏆 **Decision Tree** | Consistent superiority |

#### **Strategic Model Selection Guide**

**✅ Use Decision Tree When:**
- **Accuracy is paramount** (86% vs 81%)
- **Stakeholder communication** requires transparent decision paths
- **False positive costs** are high (97% precision on dissatisfied customers)
- **Balanced performance** across both classes is needed

**✅ Use LightGBM When:**
- **Perfect recall** on satisfied customers is critical (100% vs 98%)
- **Complex feature interactions** need to be captured
- **Production speed** and efficiency are priorities
- **Feature importance ranking** granularity is required

**Key Business Insight**: Decision Tree's superior overall performance (86% accuracy) makes it ideal for operational deployment, while LightGBM's perfect satisfied customer recall makes it valuable for customer retention initiatives.

#### **Confusion Matrix Insights**

**Decision Tree Performance:**
- **True Negatives**: 6,628 (correctly identified dissatisfied customers)
- **True Positives**: 8,857 (correctly identified satisfied customers)  
- **False Positives**: 194 (dissatisfied customers predicted as satisfied)
- **False Negatives**: 2,423 (satisfied customers predicted as dissatisfied)

**LightGBM Performance:**
- **True Negatives**: 5,715 (correctly identified dissatisfied customers)
- **True Positives**: 9,031 (correctly identified satisfied customers)
- **False Positives**: 20 (dissatisfied customers predicted as satisfied)  
- **False Negatives**: 3,336 (satisfied customers predicted as dissatisfied)

#### **Feature Importance Comparison**
**LightGBM vs Decision Tree priority differences reveal model-specific insights:**
- **LightGBM**: Emphasizes Age, WiFi, Flight Distance
- **Decision Tree**: Prioritizes Online Boarding, WiFi, Business Travel
- **Convergence**: Both models identify WiFi service as critical satisfaction factor

---

## 📁 **Project Architecture**

```bash
airline-customer-satisfaction-predictor/
│
├── 🎨 .streamlit/                 # Streamlit configuration
│   └── config.toml               # App theming and settings
│
├── 📊 data/                      # Dataset and preprocessing
│   └── app_flight_df.csv         # Clean, processed airline data
│
├── 🖼️ images/                    # Visual assets
│   ├── hero.jpg                  # Application hero image
│   └── logo.png                  # Brand/project logo
│
├── 🧠 models/                    # Trained ML models
│   ├── lgb_model.joblib          # LightGBM trained model
│   └── dec_tree.joblib           # Decision Tree trained model
│
├── 🔧 src/                       # Core processing modules
│   ├── __init__.py               # Package initialization
│   ├── categorical_encoder.py    # Custom encoding utilities
│   └── preprocessing.py          # Data preprocessing pipeline
│
├── 🚀 app.py                     # Main Streamlit application
├── ⚙️ config.py                  # Configuration and paths
├── 📋 pyproject.toml             # Poetry dependency management
├── 📄 LICENSE                    # GNU GPL v3.0 license
└── 📖 README.md                  # This comprehensive guide
```

---

## 🚀 **Quick Start Guide**

### **Prerequisites**
- **Python 3.8+** with Poetry dependency management
- **Git** for repository cloning

### **1. Repository Setup**
```bash
# Clone the repository
git clone https://github.com/erwincarlogonzales/airline-customer-satisfaction-predictor.git
cd airline-customer-satisfaction-predictor
```

### **2. Environment Configuration**
```bash
# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

### **3. Launch Application**
```bash
# Start Streamlit development server
streamlit run app.py

# Application will open at: http://localhost:8501
```

### **4. Usage Workflow**
1. **Input Customer Data**: Use sidebar controls to set passenger characteristics
2. **Select Model**: Choose between LightGBM or Decision Tree
3. **View Prediction**: See real-time satisfaction prediction with confidence
4. **Analyze Explanations**: Examine SHAP and LIME plots for decision reasoning
5. **Iterate**: Adjust parameters to understand sensitivity and thresholds

---

## 🎯 **Business Applications**

### **✅ Customer Experience Optimization**
- **Proactive Intervention**: Identify at-risk customers before dissatisfaction occurs
- **Service Prioritization**: Focus improvement efforts on high-impact features
- **Personalized Service**: Tailor experiences based on individual satisfaction drivers
- **Benchmarking**: Compare satisfaction predictors across customer segments

### **📈 Strategic Decision Support**
- **Investment Prioritization**: Quantify ROI of service improvement initiatives
- **Route Planning**: Understand satisfaction patterns across different flight distances
- **Service Design**: Data-driven approach to amenity and service offerings
- **Competitive Analysis**: Benchmark satisfaction drivers against industry standards

### **🔍 Operational Insights**
- **Real-time Monitoring**: Continuous satisfaction assessment during flight operations
- **Staff Training**: Focus training on services with highest satisfaction impact
- **Resource Allocation**: Optimize staffing and amenities based on prediction insights
- **Quality Assurance**: Systematic tracking of satisfaction-critical touchpoints

---

## 🎓 **Technical Achievements**

### **Machine Learning Excellence**
- **Dual Model Implementation**: Balanced accuracy and interpretability approach
- **Feature Engineering**: Comprehensive categorical encoding and preprocessing
- **Model Validation**: Robust training/testing with proper evaluation metrics
- **Hyperparameter Optimization**: Tuned models for optimal performance

### **Explainable AI Integration**
- **SHAP Framework**: Global and local explanation generation
- **LIME Implementation**: Instance-specific interpretability
- **Visualization Excellence**: Clear, actionable explanation plots
- **Interactive Explanations**: Real-time XAI updates with parameter changes

### **Software Engineering**
- **Modular Architecture**: Clean separation of concerns across components
- **Configuration Management**: Centralized settings and path management
- **Dependency Management**: Poetry-based reproducible environments
- **Production Deployment**: Streamlit Cloud hosting with CI/CD integration

---

## 🔮 **Future Enhancements**

### **Model Improvements**
- **Ensemble Methods**: Combine LightGBM and Decision Tree for hybrid predictions
- **Deep Learning**: Neural network architectures for complex pattern recognition
- **Online Learning**: Adaptive models that improve with real-time feedback
- **Multi-objective Optimization**: Balance accuracy, fairness, and interpretability

### **Feature Expansion**
- **Text Analytics**: Incorporate customer review sentiment analysis
- **Temporal Patterns**: Time-series analysis of satisfaction trends
- **External Data**: Weather, delays, and operational data integration
- **Demographic Insights**: Advanced customer segmentation and personalization

### **Platform Development**
- **API Development**: RESTful endpoints for system integration
- **Mobile Application**: Native mobile interface for field usage
- **Dashboard Analytics**: Executive-level reporting and KPI tracking
- **A/B Testing Framework**: Experimental platform for intervention testing

---

## 📊 **Data Science Impact**

This project demonstrates comprehensive data science capabilities across the full ML lifecycle:

**🔬 Research & Analysis:**
- Exploratory data analysis revealing satisfaction correlation patterns
- Statistical validation of feature importance across multiple model types
- Comprehensive evaluation methodology with multiple performance metrics

**🛠️ Engineering Excellence:**
- Production-ready code architecture with proper modularity
- Robust preprocessing pipeline handling categorical and numerical features
- Efficient model serialization and deployment strategies

**📈 Business Intelligence:**
- Translation of technical model outputs into actionable business insights
- Interactive visualization enabling stakeholder engagement and understanding
- Clear communication of AI decision-making processes to non-technical audiences

**🎯 Practical Applications:**
- Real-world deployment demonstrating scalable ML solution architecture
- User-friendly interface enabling widespread adoption across business functions
- Comprehensive documentation supporting knowledge transfer and maintenance

---

## 🤝 **Contributing & Collaboration**

This project represents a foundation for advanced customer satisfaction analytics and is designed for extension and collaboration.

**Areas for Contribution:**
- Model performance optimization and new algorithm integration
- Additional explainability frameworks and visualization improvements
- Enhanced preprocessing pipelines for diverse data sources
- Integration with airline operational systems and real-time data feeds

**Research Applications:**
- Customer experience optimization studies
- AI explainability and interpretability research
- Human-AI interaction design and usability analysis
- Business intelligence and decision support system development

---

## 📋 **License & Usage**

This project is licensed under the **GNU General Public License v3.0**, ensuring open-source availability while protecting contributor rights. See the [LICENSE](LICENSE) file for complete terms.

**Commercial Usage**: Contact for licensing arrangements for commercial applications or proprietary implementations.

---

<div align="center">

**🎯 Data-Driven Decisions • 🔍 Transparent AI • ✈️ Customer-Centric Innovation**

*Transforming airline customer experience through explainable machine learning*

**[🚀 Try Live Demo](https://erwincarlogonzales-airline-customer-reviews-app-l8h2pd.streamlit.app/) • [📊 View Dashboard](https://public.tableau.com/app/profile/erwincarlogonzales/viz/Book5_17312434149340/Reviews)**

</div>