#  🏥 Employee Attrition in Healthcare

This project addresses employee attrition in healthcare using Explainable AI (XAI) to identify influential factors and provide actionable insights to improve retention. It combines machine learning and deep learning models with interpretable AI methods to predict attrition and understand key factors.

##  🗂️ Table of Contents

- Background
- Objectives
- Dataset
- Features
- Tech Stack
- Algorithms
- Explainable AI Techniques
- Hugging Face Deployment

## 📝 Background

Employee attrition in healthcare can lead to staffing shortages, increased operational costs, and impacts on patient care. This project uses predictive models and explainable AI to identify the underlying causes of attrition, offering insights to help healthcare managers take proactive measures to improve employee retention.

## 🎯 Objectives

- Identifying key features which influence attrition in healthcares.
- Developing a model that predicts attrition in healthcare.
- Interpreting the best performing model using Explainable AI.
- Pinpointing actionable changes to reduce employee attrition from the interpretability.

## 🧩 Methodology


The project follows a structured approach to predict and analyze employee attrition using various models and XAI techniques. Below is an overview of the flow diagram of the methodology:

 <p align="center">
    <img src="Images\Attrition_Methodology-new.png">
</p>

## 🗄️ Dataset

The dataset used for this project contains records of healthcare employees, including personal attributes, job roles, working conditions, and performance metrics. Key columns include:

- **Age**
- **Department**
- **Job Role**
- **Education**
- **Job Satisfaction**
- **Years at Company**
- **Work-Life Balance**
- **OverTime**
- **Attrition (Target)**

The dataset used for this project is publicly available and can be accessed [here](https://www.kaggle.com/datasets/jpmiller/employee-attrition-for-healthcare).


## ✨ Features

- **Attrition Prediction**: Predicts employee attrition based on key factors.
- **Explainability**: Uses SHAP and LIME for model interpretation.
- **Data Visualization**: Visual insights into patterns affecting attrition.
- **Actionable Recommendations**: Suggestions for addressing identified risk factors.

## 🛠 Tech Stack

- **Programming Language**: Python
- **Machine Learning**: scikit-learn, PyTorch
- **Explainable AI**: LIME, SHAP-Counterfactual
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📊 Algorithms

This project employs a range of models, including shallow learning, ensemble methods, and deep neural networks.

- **Shallow Learning Models**:
  - Logistic Regression (LR)
  - K-Nearest Neighbors (KNN)
  - Decision Tree (DT)
  - Support Vector Machine (SVM)

- **Ensemble Learning**:
  - **Bagging**: Random Forest (RF)
  - **Boosting**: Gradient Boosting (GB), AdaBoost, XGBoost, LightGBM, CatBoost
  - **Stacking**:
    - Base models: RF, XGBoost, SVM
    - Meta model: Logistic Regression (LR)

- **Deep Neural Networks**:
  - Multilayer Perceptron (MLP)
  - Convolutional Neural Network (CNN)
  - TabNet

## 💡 Explainable AI Techniques

To provide transparency into the model’s predictions, we employ the following Explainable AI (XAI) techniques:

- **LIME (Local Interpretable Model-agnostic Explanations)**
- **SHAP-Counterfactual Technique** 

These methods make it possible to:

1. **Identify Key Features**: Determine which features are most influential in predicting employee attrition.
2. **Generate Visual Explanations**: Use LIME to interpret local instances.
3. **Provide Actionable Insights**: Use SHAP-Counterfactual technique to suggest features and their direction of changes, where improvements may reduce employee turnover.

