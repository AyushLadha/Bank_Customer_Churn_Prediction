# 🏦 Bank Customer Churn Prediction

A deep learning classification project that predicts whether a bank customer is likely to churn, built with an Artificial Neural Network (ANN) and deployed as an interactive web app using Streamlit.

---

## 📌 Project Overview

Customer churn is one of the most costly problems in banking. Retaining an existing customer is significantly cheaper than acquiring a new one — yet most banks struggle to identify at-risk customers before they leave.

This project builds a binary classification model using an ANN trained on real customer data. The model predicts churn likelihood based on demographics, account activity, and product usage — enabling proactive retention decisions.

**Key outcomes:**
- End-to-end ML pipeline: data preprocessing → model training → evaluation → deployment
- ANN classifier built with Keras/TensorFlow
- Serialized preprocessing pipeline for consistent inference on new data
- Streamlit web app for live predictions without any code

---

## 🗂️ Repository Structure

```
Bank_Customer_Churn_Prediction/
│
├── Churn_Modelling.csv          # Raw dataset
├── churn_prediction.ipynb       # EDA, feature engineering, model training & evaluation
├── model.h5                     # Trained Keras ANN model weights
├── data_preprocessor.pickle     # Serialized preprocessing pipeline (scaler + encoders)
├── app.py                       # Streamlit app for interactive predictions
└── README.md
```

---

## 📊 Dataset

The dataset contains **10,000 bank customer records** with the following features:

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Country of residence (France, Germany, Spain) |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Tenure` | Years with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Credit card ownership (1 = Yes, 0 = No) |
| `IsActiveMember` | Active membership status (1 = Yes, 0 = No) |
| `EstimatedSalary` | Estimated annual salary |
| `Exited` | **Target variable** — churned (1) or retained (0) |

---

## 🧠 Model Architecture

- **Type:** Artificial Neural Network (ANN)
- **Framework:** TensorFlow / Keras
- **Input layer:** 11 features (after encoding and scaling)
- **Hidden layers:** Fully connected dense layers with ReLU activation
- **Output layer:** Single neuron with Sigmoid activation (binary classification)
- **Loss function:** Binary Crossentropy
- **Optimizer:** Adam

**Preprocessing pipeline:**
- Label encoding for binary categorical features
- One-hot encoding for `Geography`
- Standard scaling for numerical features
- Pipeline serialized with `pickle` for consistent train/inference behavior

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow pandas numpy scikit-learn streamlit
```

### Run the Streamlit App
```bash
streamlit run app.py
```
The app will launch in your browser. Enter customer details using the sidebar controls and get an instant churn probability prediction.

### Run the Notebook
Open `churn_prediction.ipynb` in Jupyter or Google Colab to explore the full EDA, training process, and evaluation metrics.

---

## 📈 Results

The model was evaluated on a held-out test set:

| Metric | Score |
|---|---|
| Accuracy | ~86% |
| Precision | ~75% |
| Recall | ~46% |
| AUC-ROC | ~85% |

> **Note:** Class imbalance (~20% churn rate) was addressed during training. AUC-ROC is the primary evaluation metric for this use case.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-red?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)

---

## 📁 Use Cases

This model can be adapted for similar binary churn/attrition problems in:
- Telecom customer retention
- Subscription service cancellation prediction
- Insurance policy lapse prediction

---

## 🙏 Acknowledgments

Inspired by the ML tutorials from [Krish Naik](https://www.youtube.com/@krishnaik06). His hands-on approach to building and deploying neural network models was a key learning resource for this project.

---
