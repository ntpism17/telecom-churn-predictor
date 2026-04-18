import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('best_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📊 Telecom Customer Churn Predictor")
st.markdown("Business Data Analytics in Practice - Enter customer details below to predict churn probability.")

st.sidebar.header("Customer Information")
age = st.sidebar.slider("Age", 18, 100, 30)
monthly_bill = st.sidebar.number_input("Monthly Bill ($)", min_value=0.0, value=50.0)
usage_gb = st.sidebar.number_input("Total Data Usage (GB)", min_value=0.0, value=20.0)
service_calls = st.sidebar.slider("Customer Service Calls", 0, 10, 1)

if st.sidebar.button("Predict Churn"):
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Monthly_Bill': [monthly_bill],
        'Total_Usage_GB': [usage_gb],
        'Customer_Service_Calls': [service_calls]
    })

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk: This customer is likely to CHURN.")
        st.write("Recommendation: Offer a personalized discount or reach out to resolve any ongoing issues.")
    else:
        st.success("✅ Safe: This customer is likely to STAY.")
        st.write("Recommendation: Continue standard marketing engagement.")
