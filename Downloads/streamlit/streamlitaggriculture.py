# app.py

import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import load_data, standardize_data
from prediction import predict_new_observation

# Titre de l'application
st.title("Aide a la prise de decision de la Cultures Agricoles")

# Charger le modèle
model = joblib.load("best_model.pkl")

# Entrée de l'utilisateur
st.sidebar.header("Paramètres d'entrée")
N = st.sidebar.number_input("Nitrogen content (kg/ha)", min_value=0.0, max_value=200.0, value=87.0)
P = st.sidebar.number_input("Phosphorous content (kg/ha)", min_value=0.0, max_value=200.0, value=35.0)
K = st.sidebar.number_input("Potassium content (kg/ha)", min_value=0.0, max_value=200.0, value=25.0)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=21.44)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=63.16)
ph = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.17)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=65.88)

# Créer un bouton pour prédire
if st.sidebar.button("Prédire"):
    # Préparer l'observation
    observation = [N, P, K, temperature, humidity, ph, rainfall]
    
    # Standardiser l'observation
    train_mean = pd.Series([87, 35, 25, 21.44, 63.16, 6.17, 65.88])  # Moyenne d'entraînement
    train_std = pd.Series([10, 10, 10, 5, 10, 1, 20])  # Écart type d'entraînement
    y_pred = predict_new_observation(model, observation, train_mean, train_std)
    
    # Afficher le résultat
    st.write(f"La culture prédite est : {y_pred[0]}")

