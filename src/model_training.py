import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_absolute_error, r2_score, mean_squared_error


CLF_MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

REG_MODELS = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'SVR': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}


def train_classification_models(X_scaled, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    results = []
    trained_models = {}
    for name, model in CLF_MODELS.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
        results.append({
            'Modelo': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'F1-Score': round(f1_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'CV_F1': round(cv_scores.mean(), 4),
        })
    df_results = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
    return df_results, trained_models, (X_train, X_test, y_train, y_test)


def train_regression_models(X_scaled, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    results = []
    trained_models = {}
    for name, model in REG_MODELS.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        results.append({
            'Modelo': name,
            'MAE': round(mean_absolute_error(y_test, y_pred), 2),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            'R²': round(r2_score(y_test, y_pred), 4),
            'CV_R²': round(cv_scores.mean(), 4),
        })
    df_results = pd.DataFrame(results).sort_values('R²', ascending=False)
    return df_results, trained_models, (X_train, X_test, y_train, y_test)


def save_model(model, name, models_path="models/trained_models"):
    os.makedirs(models_path, exist_ok=True)
    path = os.path.join(models_path, f"{name}.joblib")
    joblib.dump(model, path)
    return path


def load_model(name, models_path="models/trained_models"):
    path = os.path.join(models_path, f"{name}.joblib")
    return joblib.load(path)
