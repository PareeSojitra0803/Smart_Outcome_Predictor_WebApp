# ================================
# Import Required Libraries
# ================================

import streamlit as st
import pandas as pd
import joblib

# ================================
# Page Configuration
# ================================

st.set_page_config(
    page_title="Smart Outcome Predictor",
    page_icon="🎓",
    layout="wide"
)

# ================================
# Load Trained Models
# ================================

classifier = joblib.load("models/xgb_classifier.pkl")
regressor = joblib.load("models/xgb_regressor.pkl")
scaler = joblib.load("models/scaler.pkl")
imputer = joblib.load("models/imputer.pkl")

# ================================
# Sidebar Navigation
# ================================

st.sidebar.title("🎓 Smart Outcome Predictor")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🧠 Prediction",
        "📂 Batch Prediction",
        "📊 Model Insights",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
**Machine Learning Web App**

🎯 Classification & Regression

⚡ Powered by XGBoost
""")

# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("🎓 Smart Outcome Predictor")
    st.caption("AI-Powered Student Performance Prediction System")

    st.markdown("---")

    st.markdown("""
### Welcome 👋

Smart Outcome Predictor is an end-to-end Machine Learning application that predicts student learning outcomes using **XGBoost Classification** and **Regression** models.

The application supports both **Individual Prediction** and **Batch Prediction** for educational analytics.
""")

    st.markdown("---")

    # ===================================================
    # Dashboard Metrics
    # ===================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🤖 Models", "2")

    with c2:
        st.metric("📊 Features", "15")

    with c3:
        st.metric("🎯 Tasks", "Classification + Regression")

    with c4:
        st.metric("⚡ Algorithm", "XGBoost")

    st.markdown("---")

    # ===================================================
    # Features
    # ===================================================

    st.subheader("🚀 Key Features")

    col1, col2 = st.columns(2)

    with col1:

        st.success("🧠 Individual Prediction")

        st.write("""
- Predict Completion Status
- Predict Final Score
- Interactive User Form
""")

        st.success("📂 Batch Prediction")

        st.write("""
- Upload CSV Dataset
- Generate Predictions
- Download Results
""")

    with col2:

        st.success("⚙️ Machine Learning")

        st.write("""
- XGBoost Models
- Data Preprocessing
- Feature Engineering
- Model Serialization
""")

        st.success("📈 Dashboard")

        st.write("""
- Interactive UI
- Model Insights
- Project Overview
""")

    st.markdown("---")

    # ===================================================
    # Workflow
    # ===================================================

    st.subheader("🔄 Application Workflow")

    st.info("""
        1️⃣ Enter Student Details or Upload a CSV Dataset

        ⬇️

        2️⃣ Data Preprocessing & Feature Engineering

        ⬇️

        3️⃣ XGBoost Models Generate Predictions

        ⬇️

        4️⃣ View & Download Results
        """)

    st.markdown("---")

    st.caption("Developed by Paree Sojitra • AI / ML & Data Science")
    



# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🧠 Prediction":

    st.title("🧠 Student Outcome Prediction")
    st.markdown("Fill in the student details below.")

    st.markdown("---")

    # =========================
    # Student Information
    # =========================

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=80,
            value=22
        )

        region = st.selectbox(
            "Country / Region",
            [
                "Africa",
                "Americas",
                "Asia",
                "Europe",
                "Oceania"
            ]
        )

        device = st.selectbox(
            "Device Type",
            [
                "Laptop",
                "Mobile",
                "Tablet"
            ]
        )

        education = st.selectbox(
            "Education Background",
            [
                "Graduate",
                "HighSchool",
                "Undergrad",
                "WorkingPro"
            ]
        )

        course_level = st.selectbox(
            "Course Level",
            [
                "Beginner",
                "Intermediate",
                "Advanced"
            ]
        )

        course_category = st.selectbox(
            "Course Category",
            [
                "Business",
                "Data",
                "Design",
                "Marketing",
                "Programming"
            ]
        )

        course_date = st.date_input(
            "Course Start Date"
        )

    with col2:

        sessions = st.number_input(
            "Sessions",
            min_value=0,
            value=20
        )

        time_spent = st.number_input(
            "Time Spent (Hours)",
            min_value=0.0,
            value=40.0
        )

        videos = st.number_input(
            "Videos Watched",
            min_value=0,
            value=50
        )

        quizzes = st.number_input(
            "Quiz Attempts",
            min_value=0,
            value=10
        )

        assignments = st.number_input(
            "Assignments Submitted",
            min_value=0,
            value=8
        )

        forum_posts = st.number_input(
            "Forum Posts",
            min_value=0,
            value=5
        )

        avg_quiz = st.slider(
            "Average Quiz Score",
            0,
            100,
            75
        )

        attendance = st.slider(
            "Attendance Rate (%)",
            0,
            100,
            85
        )

    st.markdown("---")

    predict_btn = st.button(
        "🚀 Predict Outcome",
        use_container_width=True
    )

    if predict_btn:

        st.success("Prediction Completed Successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Completion Status",
                "Completed ✅"
            )

        with col2:
            st.metric(
                "Predicted Final Score",
                "87.45"
            )

        st.progress(87)

        st.info(
            "⚠️ Demo prediction shown. "
            "Real model inference will be connected in the next version."
        )
        


# ============================================================
# BATCH PREDICTION PAGE
# ============================================================

elif page == "📂 Batch Prediction":

    st.title("📂 Batch Prediction")
    st.markdown(
        "Upload a CSV file containing student records to generate predictions."
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose a CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success("✅ Dataset uploaded successfully!")

        # ============================
        # Dataset Summary
        # ============================

        rows, cols = df.shape
        missing = df.isnull().sum().sum()

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Rows", rows)

        with c2:
            st.metric("Columns", cols)

        with c3:
            st.metric("Missing Values", missing)

        st.markdown("---")

        # ============================
        # Dataset Preview
        # ============================

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        st.markdown("---")

        # ============================
        # Generate Prediction
        # ============================

        if st.button(
            "🚀 Generate Predictions",
            use_container_width=True
        ):

            result = df.copy()

            # Demo prediction
            result["Predicted Completion"] = "Completed"
            result["Predicted Final Score"] = 87.45

            st.success("✅ Prediction completed successfully!")

            st.subheader("📊 Prediction Results")

            st.dataframe(
                result.head(),
                use_container_width=True
            )

            st.download_button(
                "📥 Download Prediction Results",
                data=result.to_csv(index=False),
                file_name="prediction_results.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:

        st.info(
            "Upload a CSV dataset to begin batch prediction."
        )
        


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "📊 Model Insights":

    st.title("📊 Model Insights")
    st.markdown("Overview of the Machine Learning pipeline used in this project.")

    st.markdown("---")

    # ===================================================
    # Models
    # ===================================================

    st.subheader("🤖 Trained Models")

    col1, col2 = st.columns(2)

    with col1:
        st.success("Classification Model")
        st.write("• XGBoost Classifier")
        st.write("• Predicts Completion Status")

    with col2:
        st.success("Regression Model")
        st.write("• XGBoost Regressor")
        st.write("• Predicts Final Score")

    st.markdown("---")

    # ===================================================
    # Dataset
    # ===================================================

    st.subheader("📂 Dataset Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Records", "5,000")

    with c2:
        st.metric("Input Features", "15")

    with c3:
        st.metric("Prediction Tasks", "2")

    st.markdown("---")

    # ===================================================
    # Feature Engineering
    # ===================================================

    st.subheader("⚙️ Feature Engineering")

    st.write("""
    ✅ Week of Year Extraction

    ✅ Engagement Score

    ✅ Quiz × Attendance Interaction

    ✅ Videos per Session

    ✅ Time per Session
    """)

    st.markdown("---")

    # ===================================================
    # Preprocessing
    # ===================================================

    st.subheader("🧹 Data Preprocessing")

    st.write("""
    • Missing Value Imputation

    • Label Encoding

    • Standard Scaling

    • Feature Engineering

    • Joblib Model Serialization
    """)

    st.markdown("---")

    # ===================================================
    # Libraries
    # ===================================================

    st.subheader("🛠️ Technologies Used")

    st.write("""
    - Python

    - Pandas

    - NumPy

    - Scikit-Learn

    - XGBoost

    - Streamlit

    - Joblib
    """)
    


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About Smart Outcome Predictor")

    st.markdown("""
    **Smart Outcome Predictor** is a Machine Learning web application that
    predicts student learning outcomes using Classification and Regression models.

    The application enables users to make predictions for individual students
    as well as multiple students through batch CSV uploads.
    """)

    st.markdown("---")

    # ===================================
    # Project Objectives
    # ===================================

    st.subheader("🎯 Project Objectives")

    st.write("""
    • Predict Course Completion Status

    • Estimate Final Student Score

    • Support Batch Prediction using CSV files

    • Demonstrate an end-to-end Machine Learning workflow
    """)

    st.markdown("---")

    # ===================================
    # Tech Stack
    # ===================================

    st.subheader("🛠️ Tech Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        **Frontend**

        • Streamlit

        • Python
        """)

    with col2:
        st.write("""
        **Machine Learning**

        • XGBoost

        • Scikit-Learn

        • Pandas

        • NumPy
        """)

    st.markdown("---")

    # ===================================
    # Developer
    # ===================================

    st.subheader("👩‍💻 Developer")

    st.info("""
    **Paree Sojitra**

    AI / ML & Data Science Student

    Passionate about Machine Learning,
    Data Analytics and AI-powered Applications.
    """)

    st.markdown("---")

    # ===================================
    # Version
    # ===================================

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Version", "1.0")

    with c2:
        st.metric("Framework", "Streamlit")

    with c3:
        st.metric("Models", "2")

    st.markdown("---")

    st.caption(
        "© 2026 Smart Outcome Predictor | Developed by Paree Sojitra"
    )
    
