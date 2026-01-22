# app.py
#
# PART B – Streamlit Web GUI for Titanic Survival Prediction

import joblib
import pandas as pd
import streamlit as st

# ------------------------------
# Load saved model
# ------------------------------
@st.cache_resource
def load_model():
    data = joblib.load("model/titanic_survival_model.pkl")
    model = data["model"]
    features = data["features"]
    return model, features

model, feature_cols = load_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🚢 Titanic Survival Prediction System")
st.write(
    "This app predicts whether a passenger would **Survive** or **Not Survive** "
    "the Titanic disaster based on selected features."
)

st.markdown("---")
st.subheader("Enter Passenger Details")

with st.form("titanic_form"):
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=0)
    sex = st.selectbox("Sex", ["male", "female"], index=0)
    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=29.0,
        step=1.0
    )
    sibsp = st.number_input(
        "Number of Siblings/Spouses Aboard (SibSp)",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )
    embarked = st.selectbox(
        "Port of Embarkation (Embarked)",
        ["S", "C", "Q"],
        index=0
    )

    submit_button = st.form_submit_button("Predict Survival")

if submit_button:
    # Arrange inputs into DataFrame using same columns as during training
    input_dict = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Embarked": embarked,
    }

    input_df = pd.DataFrame([input_dict], columns=feature_cols)

    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    label = "Survived" if prediction == 1 else "Did Not Survive"

    st.markdown("### Prediction Result")
    if prediction == 1:
        st.success(f"The model predicts: **{label}** ✅")
    else:
        st.error(f"The model predicts: **{label}** ❌")

    st.markdown("### Prediction Probabilities")
    st.write(f"Probability of **Did Not Survive (0)**: `{probabilities[0]:.4f}`")
    st.write(f"Probability of **Survived (1)**: `{probabilities[1]:.4f}`")
