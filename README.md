# Proyecto de Machine Learning - Optimización Logística

## Descripción
Proyecto para predecir:
- Si un vehículo es adecuado para un envío (clasificación)
- La eficiencia de una ruta en km/h (regresión)

## Modelos Probados
### Clasificación (vehiculo_adecuado):
- Gradient Boosting, Random Forest, Decision Tree, KNN, SVM, Logistic Regression

### Regresión (eficiencia_ruta):
- Gradient Boosting, Random Forest, Decision Tree, Linear Regression, KNN, SVR

## Resultados
- Clasificación: Gradient Boosting con F1 Score de 0.8333
- Regresión: Gradient Boosting con R² de 0.9883

## Requisitos
- Python 3.10+
- Ver `requirements.txt` para dependencias

## Instalación
```bash
python -m venv venv
# Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso
### Pipeline completo (recomendado):
```bash
python run_pipeline.py
```
Esto ejecuta: carga/limpieza → feature engineering → modelos → evaluación → clustering

### Notebooks (orden sugerido):
1. `01_EDA.ipynb` - Exploración y limpieza de datos
2. `02_supervising_modeling.ipynb` - Modelos supervisados base
3. `03_model_evaluation.ipynb` - Evaluación comparativa de modelos
4. `04_hyperparameter_optimization.ipynb` - Optimización con GridSearchCV
5. `05_final_analysis.ipynb` - Análisis final y conclusiones
6. `06_unsupervised_analysis.ipynb` - Clustering no supervisado (KMeans + PCA)

## Estructura del proyecto
```
proyecto_modelado/
├── data/                    # Datos crudos y limpios (CSV)
├── models/
│   └── trained_models/      # Modelos .joblib entrenados
├── notebooks/               # 6 Jupyter notebooks
├── results/
│   ├── metrics/             # CSVs con métricas
│   ├── plots/               # Gráficos generados (PNG)
│   └── reports/             # Reportes
├── src/                     # Módulos Python reutilizables
│   ├── make_dataset.py      # Carga y limpieza
│   ├── feature_engineering.py # Merges, targets, features
│   ├── models.py            # Entrenamiento multi-modelo + GridSearch
│   └── evaluation.py        # Evaluación, gráficos, clustering
├── run_pipeline.py          # Pipeline ejecutable
├── requirements.txt         # Dependencias
└── README.md
