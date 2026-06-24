import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")

# Load dataset
data = pd.read_csv("dataset.csv")

# Load accuracy
with open("accuracy.txt", "r") as f:
    accuracy = f.read()

# Title
st.title("🧠 AI Mental Health Prediction System")

# Accuracy
st.info(f"📈 Model Accuracy: {accuracy}%")

# About
st.markdown("""
### About Project
This AI-powered system predicts stress risk using Machine Learning.

It analyzes:
- Age
- Sleep Hours
- Work Hours

and predicts whether a person has a Low Stress Risk or High Stress Risk.
""")

# Inputs
age = st.number_input("Enter Age", min_value=1, max_value=100)

sleep = st.number_input("Sleep Hours", min_value=0, max_value=24)

work = st.number_input("Work Hours", min_value=0, max_value=24)

# Button
predict = st.button("Predict Stress Level")

if predict:

    result = model.predict([[age, sleep, work]])

    probability = model.predict_proba([[age, sleep, work]])
    confidence = max(probability[0]) * 100

    st.info(f"Confidence: {confidence:.2f}%")

    if result[0] == 1:

        st.error("⚠️ High Stress Risk")

        st.subheader("Suggestions:")

        st.write("• Sleep at least 7-8 hours")
        st.write("• Take regular breaks")
        st.write("• Practice meditation")
        st.write("• Reduce excessive workload")

    else:

        st.success("✅ Low Stress Risk")

        st.subheader("Suggestions:")

        st.write("• Maintain your healthy lifestyle")
        st.write("• Continue good sleep habits")
        st.write("• Keep work-life balance")

# Graph
st.subheader("📊 Dataset Stress Level Distribution")

stress_count = data["Stress_Level"].value_counts()

stress_count.index = ["Low Stress", "High Stress"]

st.bar_chart(stress_count)

# Footer
st.markdown("---")
st.caption("Developed using Python, Scikit-Learn and Streamlit")