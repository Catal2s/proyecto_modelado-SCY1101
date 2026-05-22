from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


PARAM_GRID_CLF = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
}

PARAM_GRID_REG = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
}


def optimize_classifier(X_scaled, y, model_class=RandomForestClassifier, param_grid=None, random_state=42):
    if param_grid is None:
        param_grid = PARAM_GRID_CLF
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state, stratify=y
    )
    grid = GridSearchCV(
        model_class(random_state=random_state),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def optimize_regressor(X_scaled, y, model_class=RandomForestRegressor, param_grid=None, random_state=42):
    if param_grid is None:
        param_grid = PARAM_GRID_REG
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state
    )
    grid = GridSearchCV(
        model_class(random_state=random_state),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid
