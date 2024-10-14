from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_and_evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    results = {}
    best_accuracy = 0
    best_model_name = None
    best_model_instance = None

    for name, model in models.items():
        print(f"Optimizing model: {name}")
        if name in param_grids:
            grid_search = GridSearchCV(model, param_grids[name], cv=kf, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = {}

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        results[name] = {
            "Best Parameters": best_params,
            "Test Accuracy": accuracy,
            "Confusion Matrix": conf_matrix,
            "Classification Report": class_report
        }

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model_instance = best_model

    return results, best_model_name, best_accuracy, best_model_instance
