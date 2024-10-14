import streamlit as st
import numpy as np
import joblib as jb

# Ajouter des styles CSS personnalisés
st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
    }
    h1 {
        color: #4CAF50;
    }
    .subheader {
        color: #FF6347;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Titre de l'application
st.title("Prédiction utilisant les données iris pour le déploiement via Streamlit")
st.subheader("APPLICATION RÉALISÉE PAR NIANG GOUMACK")
st.markdown("*CETTE APPLICATION UTILISE UN MODÈLE DE MACHINE LEARNING DES DONNÉES IRIS POUR FAIRE LA PRÉDICTION*")

# Ajouter une image (par exemple, le logo de votre application)
#st.image("path/to/your/image.jpg", caption="Votre légende ici", use_column_width=True)

# Standarisation
train_moy = 3.4452083333333334
train_std = 1.9627834079686994

# Chargement du modèle
model = jb.load(filename="bestmodelepredict.joblib")

# Définition d'une fonction d'inférence
def inference(petal_long, petal_larg, sepal_long, sepal_larg):
    new_data = np.array([petal_long, petal_larg, sepal_long, sepal_larg])
    new_data_reshape = new_data.reshape(1, -1)
    new_data_reshape_stand = (new_data_reshape - train_moy) / train_std
    pred = model.predict(new_data_reshape_stand)
    return pred

# Initialisation des valeurs des champs
if 'petal_long' not in st.session_state:
    st.session_state.petal_long = 0.0
if 'petal_larg' not in st.session_state:
    st.session_state.petal_larg = 0.0
if 'sepal_long' not in st.session_state:
    st.session_state.sepal_long = 0.0
if 'sepal_larg' not in st.session_state:
    st.session_state.sepal_larg = 0.0

# L'utilisateur saisit une valeur pour chaque caractéristique
st.session_state.petal_long = st.number_input(label='Petal length:', min_value=-10.0, value=st.session_state.petal_long)
st.session_state.petal_larg = st.number_input(label='Petal width:', min_value=-10.0, value=st.session_state.petal_larg)
st.session_state.sepal_long = st.number_input(label='Sepal length:', min_value=-10.0, value=st.session_state.sepal_long)
st.session_state.sepal_larg = st.number_input(label='Sepal width:', min_value=-10.0, value=st.session_state.sepal_larg)

# Création du bouton de prédiction
if st.button("Predict"):
    prediction = inference(st.session_state.petal_long, st.session_state.petal_larg, st.session_state.sepal_long, st.session_state.sepal_larg)
    st.success(f"La prédiction est : {prediction[0]}")

    # Réinitialiser les champs à zéro
    st.session_state.petal_long = 0.0
    st.session_state.petal_larg = 0.0
    st.session_state.sepal_long = 0.0
    st.session_state.sepal_larg = 0.0
