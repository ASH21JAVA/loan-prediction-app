import streamlit as st
import pickle
import numpy as np

# Load model + scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Loan Approval Prediction")

# Inputs
credit = st.number_input("Credit Score")
income = st.number_input("Applicant Income")
loan = st.number_input("Loan Amount")
dti = st.number_input("DTI Ratio")
age = st.number_input("Age")

if st.button("Predict"):
    data = np.array([[credit, income, loan, dti, age]])
    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Approved ✅")
    else:
        st.error("Not Approved ❌")