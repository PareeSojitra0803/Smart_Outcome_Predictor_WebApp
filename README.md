# 🎓 Smart Outcome Predictor

> **An End-to-End Machine Learning Web Application for Predicting Student Learning Outcomes**

Predict **Course Completion Status** and **Final Student Score** using Machine Learning models through an interactive Streamlit web application.

---

## 🌐 Live Demo

🚀 **Streamlit Web App**

> [Smart Outcocme Predictor WebApp](https://smart-outcome-predictor-webapp.streamlit.app/)

---

## 📌 Project Overview

Smart Outcome Predictor is an AI-powered educational analytics application that helps estimate student performance based on learning behavior and engagement.

The application supports:

- 🎯 Individual Student Prediction
- 📂 Batch Prediction using CSV Upload
- 📊 Model Insights Dashboard
- 🖥️ Interactive Streamlit Interface

---

## 🚀 Features

### 🧠 Individual Prediction
- Predict Course Completion Status
- Predict Final Student Score
- Interactive Input Form

### 📂 Batch Prediction
- Upload CSV Dataset
- Preview Uploaded Data
- Generate Predictions
- Download Prediction Results

### 📊 Model Insights
- ML Pipeline Overview
- Feature Engineering
- Data Preprocessing
- Model Information

---

## 🤖 Machine Learning Models

| Task | Model |
|------|-------|
| Classification | XGBoost Classifier |
| Regression | XGBoost Regressor |

---

## ⚙️ Data Preprocessing

- Missing Value Imputation
- Label Encoding
- Feature Scaling
- Feature Engineering

---

## 🧩 Feature Engineering

- Week of Year Extraction
- Engagement Score
- Quiz × Attendance Interaction
- Videos per Session
- Time per Session

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Joblib

---

## 📁 Project Structure

```text
Smart_Outcome_Predictor_WebApp/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── xgb_classifier.pkl
│   ├── xgb_regressor.pkl
│   ├── scaler.pkl
│   └── imputer.pkl
```

---

## ▶️ Run Locally

```bash
git clone <repository-url>

cd Smart_Outcome_Predictor_WebApp

pip install -r requirements.txt

streamlit run app.py
```

---

## 👩‍💻 Developer

**Paree Sojitra**

AI / ML & Data Science Student

---

⭐ If you found this project useful, consider giving it a star!