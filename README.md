#  🏥 Employee Attrition in Healthcare

This project addresses employee attrition in healthcare using Explainable AI (XAI) to identify influential factors and provide actionable insights to improve retention. It combines machine learning and deep learning models with interpretable AI methods to predict attrition and understand key factors.

##  🗂️ Table of Contents

- [Background](#-background)
- [Objectives](#-objectives)
- [Methodology](#-methodology)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Algorithms](#-algorithms)
- [XAI Techniques](#-explainable-ai-techniques)

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
    <img src="Images\Attrition_Methodology_v2.png">
</p>

## ✨ Features

- **Attrition Prediction**: Predicts employee attrition based on key factors.
- **Explainability**: Uses SHAP and LIME for model interpretation.
- **Data Visualization**: Visual insights into patterns affecting attrition.
- **Actionable Recommendations**: Suggestions for addressing identified risk factors.

## 🛠 Tech Stack

- **Programming Language**: Python
- **Machine Learning**: scikit-learn, TensorFlow, PyTorch
- **Explainable AI**: SHAP, LIME
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

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

- **SHAP (SHapley Additive exPlanations)**
- **LIME (Local Interpretable Model-agnostic Explanations)**

These methods make it possible to:

1. **Identify Key Features**: Determine which features are most influential in predicting employee attrition.
2. **Generate Visual Explanations**: Use SHAP summary and dependence plots, as well as LIME explanations, to interpret individual and global model behaviors.
3. **Provide Actionable Insights**: By identifying top risk factors, management can focus on areas where improvements may reduce empl
