# main.py

import joblib
from data_preprocessing import load_data, check_null_values, standardize_data
from visualization import plot_correlation_heatmap
from model_training import train_and_evaluate_models
from prediction import predict_new_observation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Charger les données
data = load_data('C:/Users/baye.niang/Documents/PROJET-AGRICULTURE/aggriculturepredictionIA.csv')

# Vérifier les valeurs manquantes
print(check_null_values(data))

# Visualisation de la corrélation
plot_correlation_heatmap(data)

# Séparer les caractéristiques et la cible
X = data.drop('label', axis=1)
y = data['label']

# Séparer les données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardiser les données
X_train, X_test, scaler = standardize_data(X_train, X_test)

# Modèles et grilles de paramètres
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

param_grids = {
    "Logistic Regression": {'C': [0.01, 0.1, 1, 10, 100]},
    "Decision Tree": {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
    "SVM": {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']},
    "K-Nearest Neighbors": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    "Naive Bayes": {}
}

# Entraîner et évaluer les modèles
results, best_model_name, best_accuracy, best_model_instance = train_and_evaluate_models(X_train, y_train, X_test, y_test, models, param_grids)

# Afficher les résultats
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Best Parameters: {result['Best Parameters']}")
    print(f"Test Accuracy: {result['Test Accuracy']:.4f}")
    print("Confusion Matrix:")
    print(result['Confusion Matrix'])
    print("Classification Report:")
    print(result['Classification Report'])
    print("-" * 80)

# Sauvegarder le meilleur modèle
joblib.dump(best_model_instance, "best_model.pkl")

# Charger le modèle et prédire une nouvelle observation
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
new_observation = [87, 35, 25, 21.44, 63.16, 6.17, 65.88]

y_pred = predict_new_observation(best_model_instance, new_observation, train_mean, train_std)
print(f"Prédiction pour la nouvelle observation: {y_pred}")
