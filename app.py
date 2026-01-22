import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

# Title
st.title("🚢 Titanic Survival Prediction System")
st.write("Enter passenger details to predict whether the passenger survived.")

# Load trained model and scaler
with open("model/titanic_survival_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

st.subheader("Passenger Information")

# User inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Encode categorical inputs
sex = 0 if sex == "Male" else 1
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

# Prepare input for model
input_data = np.array([[pclass, sex, age, fare, embarked]])
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("🎉 Prediction: Passenger Survived")
    else:
        st.error("❌ Prediction: Passenger Did Not Survive")
