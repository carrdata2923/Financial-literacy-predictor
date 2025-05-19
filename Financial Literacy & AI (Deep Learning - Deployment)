import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import joblib

# Load the trained model
try:
    model = keras.models.load_model('student_financial_model.keras')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Load the preprocessor
try:
    preprocessor = joblib.load('preprocessor.joblib')
except Exception as e:
    st.error(f"Error loading the preprocessor: {e}")
    st.stop()

# Load the binary label encoder
try:
    label_encoder_binary = joblib.load('label_encoder_binary.joblib')
except Exception as e:
    st.error(f"Error loading the label encoder: {e}")
    st.stop()

def predict_financial_level(input_data):
    """Makes a prediction using the loaded model."""
    try:
        processed_data = preprocessor.transform(input_data)
        prediction_probability = model.predict(processed_data)
        prediction = (prediction_probability > 0.5).astype(int).flatten()
        predicted_class = label_encoder_binary.inverse_transform(prediction)[0]
        return predicted_class, prediction_probability[0][0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

def main():
    st.title("Student Financial Level Prediction")
    st.write("Enter student features to predict their potential financial level (Passing/Not Passing).")

    # **IMPORTANT:** Adapt these input widgets to match the names and data types
    # of the columns in your original 'student_financial_data.csv' file.
    # Replace the examples below with your actual features.

    # Example Numerical Feature 1
    numerical_feature_1 = st.number_input("Numerical Feature 1", value=0.0)

    # Example Categorical Feature 1 (assuming a limited number of options)
    categorical_feature_1_options = ['Option A', 'Option B', 'Option C'] # Replace with your actual categories
    categorical_feature_1 = st.selectbox("Categorical Feature 1", options=categorical_feature_1_options)

    # Example Numerical Feature 2
    numerical_feature_2 = st.number_input("Numerical Feature 2", value=0.0)

    # Example Categorical Feature 2 (assuming a limited number of options)
    categorical_feature_2_options = ['Category X', 'Category Y', 'Category Z'] # Replace with your actual categories
    categorical_feature_2 = st.selectbox("Categorical Feature 2", options=categorical_feature_2_options)

    # Add input widgets for ALL your ORIGINAL features here...
    # Ensure the labels and input types are correct for each feature.

    # Create a Pandas DataFrame from the user inputs with the ORIGINAL column names
    input_df = pd.DataFrame({
        'numerical_feature_1': [numerical_feature_1], # Replace with your actual column name
        'categorical_feature_1': [categorical_feature_1], # Replace with your actual column name
        'numerical_feature_2': [numerical_feature_2], # Replace with your actual column name
        'categorical_feature_2': [categorical_feature_2], # Replace with your actual column name
        # Add ALL your ORIGINAL feature columns here with the correct names
    })

    if st.button("Predict"):
        predicted_level, probability = predict_financial_level(input_df)
        if predicted_level:
            st.subheader("Prediction:")
            st.write(f"The predicted financial level is: **{predicted_level}**")
            st.write(f"Probability of being 'Passing': {probability:.4f}")
        else:
            st.warning("Prediction failed. Please check the input values.")

if __name__ == '__main__':
    main()
