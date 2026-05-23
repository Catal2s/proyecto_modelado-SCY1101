# Proyecto de Machine Learning — Optimización Logística

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Pipeline completo de Machine Learning para predecir la **asignación óptima de vehículos** y la **eficiencia de rutas** en una empresa de logística. Incluye modelos supervisados (clasificación + regresión), optimización de hiperparámetros y análisis no supervisado con clustering.

---

## Resultados Clave

| Tarea | Mejor Modelo | Métrica Principal | Resultado |
|-------|-------------|-------------------|-----------|
| Clasificación (vehículo adecuado) | Gradient Boosting | F1-Score | **0.8333** |
| Regresión (eficiencia de ruta) | Gradient Boosting | R² | **0.9883** |
| Clustering | KMeans (k=3) | — | 3 segmentos identificados |

---

## Dataset

Se utilizaron **4 datasets reales** de operaciones logísticas:

| Dataset | Registros | Descripción |
|---------|-----------|-------------|
| `envios.csv` | 1.030 | Envíos con peso, volumen, fechas |
| `vehiculos.csv` | 61 | Flota con capacidades y estado |
| `rutas.csv` | 82 | Rutas con distancia, tiempo, peajes |
| `incidencias.csv` | 206 | Incidentes con tipo y costo |

Los datos requerían limpieza significativa: valores no numéricos en columnas numéricas (`~500`, `$1.200`, `aprox 300`), fechas en formatos mixtos, valores nulos en múltiples columnas.

---

## Pipeline

```
data/                    notebooks/                results/
  envios.csv ─────────── 01_EDA.ipynb ──────────── metrics/
  vehiculos.csv ──────── 02_supervised_modeling/    plots/
  rutas.csv ──────────── 03_model_evaluation/       reports/
  incidencias.csv ────── 04_hyperparameter_opt/
                         05_final_analysis/
src/                     06_unsupervised_analysis/
  data_preprocessing.py
  model_training.py
  hyperparameter_tuning.py
  model_evaluation.py
```

Ejecución completa con un solo comando:

```bash
python run_pipeline.py
```

---

## Modelos Evaluados

### Clasificación — 6 modelos
Random Forest, Gradient Boosting, Decision Tree, KNN, SVM, Logistic Regression

### Regresión — 6 modelos
Random Forest, Gradient Boosting, Decision Tree, Linear Regression, KNN, SVR

### Optimización
GridSearchCV con validación cruzada de 5 folds sobre Random Forest:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [5, 10, None]
- `min_samples_split`: [2, 5, 10]

### No Supervisado
KMeans con k=3 determinado por método del codo, visualizado con PCA.

---

## Visualizaciones

| Gráfico | Descripción |
|---------|-------------|
| `comparativa_modelos.png` | F1-Score de todos los clasificadores |
| `matriz_confusion.png` | Matriz de confusión del mejor clasificador |
| `curva_roc.png` | Curva ROC con AUC=0.95 |
| `comparativa_regresores.png` | R² de todos los regresores |
| `importancia_variables.png` | Feature importance del mejor regresor |
| `top5_variables.png` | Top 5 variables para clasificación |
| `elbow_method.png` | Método del codo para KMeans |
| `clusters_pca.png` | Clusters en 2D con PCA |
| `clusters_vs_target.png` | % adecuado y eficiencia por cluster |

---

## Requisitos

- Python 3.10+
- Ver `requirements.txt`

```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

---

## Reproducibilidad

```bash
# Pipeline completo (genera todo desde cero)
python run_pipeline.py
```

O paso a paso con notebooks:
1. `01_exploratory_analysis.ipynb` — EDA y limpieza
2. `02_supervised_modeling.ipynb` — Modelos supervisados
3. `03_model_evaluation.ipynb` — Evaluación comparativa
4. `04_hyperparameter_optimization.ipynb` — GridSearchCV
5. `05_final_analysis.ipynb` — Análisis final
6. `06_unsupervised_analysis.ipynb` — Clustering

---

## Tecnologías

- **Python** 3.13 — `pandas`, `numpy`
- **Machine Learning** — `scikit-learn` (StandardScaler, 6 clasificadores, 6 regresores, KMeans, PCA, GridSearchCV)
- **Visualización** — `matplotlib`, `seaborn`
- **Serialización** — `joblib`

---

## Autor

[Tu Nombre] — Proyecto académico para Programación para la Ciencia de Datos, DUOC UC 2026.
