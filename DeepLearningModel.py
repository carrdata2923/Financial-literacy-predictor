# app.py
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import joblib

# Load the trained model
try:
    model = keras.models.load_model('student_financial_model.keras')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {type(e)}, {e}")
    st.stop()

# Load the preprocessor
try:
    preprocessor = joblib.load('preprocessor.joblib')
    st.success("Preprocessor loaded successfully!")
except Exception as e:
    st.error(f"Error loading the preprocessor: {type(e)}, {e}")
    st.stop()

# Load the binary label encoder
try:
    label_encoder_binary = joblib.load('label_encoder_binary.joblib')
    st.success("Label encoder loaded successfully!")
except Exception as e:
    st.error(f"Error loading the label encoder: {type(e)}, {e}")
    st.stop()

def predict_financial_level(input_data):
    """Makes a prediction using the loaded model and includes detailed debugging."""
    st.subheader("Debugging Input Data:")
    st.write(input_data)

    try:
        processed_data = preprocessor.transform(input_data)
        st.subheader("Debugging Processed Data:")
        st.write(processed_data)

        prediction_probability = model.predict(processed_data)
        st.subheader("Debugging Prediction Probability:")
        st.write(prediction_probability)
        st.write(f"Type of prediction_probability: {type(prediction_probability)}")
        if isinstance(prediction_probability, np.ndarray):
            st.write(f"Shape of prediction_probability: {prediction_probability.shape}")
            if prediction_probability.ndim > 1 and prediction_probability.shape[0] > 0:
                st.write(f"Value of prediction_probability[0][0]: {prediction_probability[0][0]}")
            elif prediction_probability.ndim == 1 and prediction_probability.size > 0:
                st.write(f"Value of prediction_probability[0]: {prediction_probability[0]}")

        prediction = (prediction_probability > 0.5).astype(int).flatten()
        st.subheader("Debugging Raw Prediction (0 or 1):")
        st.write(prediction)

        st.subheader("Debugging Label Encoder Classes:")
        st.write(label_encoder_binary.classes_)

        predicted_class = label_encoder_binary.inverse_transform(prediction)
        st.subheader("Debugging Predicted Class (Before Indexing):")
        st.write(predicted_class)
        st.write(f"Type of predicted_class: {type(predicted_class)}")
        if isinstance(predicted_class, np.ndarray):
            st.write(f"Shape of predicted_class: {predicted_class.shape}")
            if predicted_class.size > 0:
                st.write(f"Value of predicted_class[0]: {predicted_class[0]}")

        st.subheader("Debugging prediction_probability just before return:")
        st.write(prediction_probability)
        st.write(f"Type of prediction_probability: {type(prediction_probability)}")
        if isinstance(prediction_probability, np.ndarray):
            st.write(f"Shape of prediction_probability: {prediction_probability.shape}")

        final_predicted_class = predicted_class[0]
        return final_predicted_class, prediction_probability[0][0]
    except Exception as e:
        st.error(f"Error during final prediction processing: {type(e)}, {e}")
        return None, None

def main():
    st.title("Student Financial Level Prediction (Debugging)")
    st.write("Enter student features to predict their potential financial level (Passing/Not Passing).")

    edad_options = ['Menos de 24 años', 'De 25 a 44 años', 'De 45 a 64 años', 'Más de 65 años']
    edad = st.selectbox("Edad", options=edad_options)

    genero_options = ['Hombre', 'Mujer', 'Otro']
    genero = st.selectbox("Género", options=genero_options)

    estado_options = ['Solter@', 'En pareja', 'Otro', 'Divorciad@', 'Casad@']
    estado = st.selectbox("Estado", options=estado_options)

    acceso_options = ['Título de Técnico de Grado Superior',
                        'Prueba de Acceso a Ciclos Formativos de Grado Superior',
                        'Título Universitario o Equivalente', 'EBAU o EvAU', 'Bachillerato',
                        'Título de Técnico de Grado Medio',
                        'Prueba de acceso a la Universidad para mayores de 25 años']
    acceso = st.selectbox("Acceso", options=acceso_options)

    actualidad_options = ['Sí', 'No']
    actualidad = st.selectbox("Actualidad", options=actualidad_options)

    canal_options = ['Redes Sociales', 'Medios digitales (Internet)', 'Radio o televisión',
                        'Medios escritos (Periódico)']
    canal = st.selectbox("Canal", options=canal_options)

    medio_options = ['El Pais', 'El Mundo', 'Ok Diario', 'La Vanguardia', '20 Minutos',
                        'El Economista', 'Otros', 'ABC', 'La Voz de Galicia', 'Expansión', 'La Razón',
                        'El Confidencial', 'El Español', '5 días', 'El Correo']
    medio = st.selectbox("Medio", options=medio_options)

    equipo_options = ['Real Madrid', 'FC Barcelona', 'Otro', 'Atlético de Madrid',
                        'No soy seguidor de este deporte']
    equipo = st.selectbox("Equipo", options=equipo_options)

    resultado_equipo_options = ['Si', 'Empate', 'No', 'No soy seguidor de ningún equipo']
    resultado_equipo = st.selectbox("Resultado_Equipo", options=resultado_equipo_options)

    clima_options = ['Soleado y radiante', 'Nublado', 'Llueve', 'Nieva']
    clima = st.selectbox("Clima", options=clima_options)

    animo_options = [1, 2, 3, 4] # Assuming Animo is a numerical scale
    animo = st.selectbox("Animo", options=animo_options)

    input_df = pd.DataFrame({
        'Edad': [edad],
        'Genero': [genero],
        'Estado': [estado],
        'Acceso': [acceso],
        'Actualidad': [actualidad],
        'Canal': [canal],
        'Medio': [medio],
        'Equipo': [equipo],
        'Resultado_Equipo': [resultado_equipo],
        'Clima': [clima],
        'Animo': [animo],
    })

    if st.button("Predict"):
        predicted_level, probability = predict_financial_level(input_df)
        if predicted_level:
            st.subheader("Prediction:")
            st.write(f"The predicted financial level is: **{predicted_level}**")
            st.write(f"Probability of being 'Passing': {probability:.4f}")
        else:
            st.warning("Prediction failed. Please check the input values (see error messages above).")

if __name__ == '__main__':
    main()
