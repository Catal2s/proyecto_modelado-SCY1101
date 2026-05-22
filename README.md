# Proyecto de Machine Learning - Optimización Logística

## Descripción
Proyecto para predecir:
- Si un vehículo es adecuado para un envío (clasificación)
- La eficiencia de una ruta en km/h (regresión)

## Modelos
- Random Forest Classifier (Optimizado con GridSearchCV)
- Random Forest Regressor (Optimizado con GridSearchCV)

## Resultados
- Clasificación: F1 Score de 0.8333 (+9.85% de mejora)
- Regresión: R² de 0.9873, MAE de 5.30 km/h

## Estructura
- 01_eda_limpieza.ipynb - Exploración y limpieza de datos
- 02_feature_engineering.ipynb - Creación de variables
- 03_modelado_base.ipynb - Modelos base
- 04_hyperparameter_optimization.ipynb - Optimización con GridSearchCV
- 05_final_analysis.ipynb - Evaluación final y conclusiones
