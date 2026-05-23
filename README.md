# Proyecto de Machine Learning — Optimización Logística

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Pipeline completo de Machine Learning para predecir la **asignación óptima de vehículos** y la **eficiencia de rutas** en una empresa de logística. Incluye modelos supervisados (clasificación + regresión), optimización de hiperparámetros con GridSearchCV y análisis no supervisado con KMeans + PCA.

---

## Resultados Clave

| Tarea | Mejor Modelo | Métrica | Resultado |
|-------|-------------|---------|-----------|
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

Los datos requerían limpieza significativa: valores no numéricos en columnas numéricas (`~500`, `$1.200`, `aprox 300`), fechas en formatos mixtos, valores nulos en múltiples columnas. Se implementó una función `clean_numeric()` con expresiones regulares para extraer solo los dígitos válidos.

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

### Clasificación — 6 modelos (target: vehiculo_adecuado)

| Modelo | ¿Cómo funciona? | Resultado |
|--------|----------------|-----------|
| **Gradient Boosting** | Árboles en secuencia, cada uno corrige el error del anterior. El más preciso. | **F1: 0.8333** |
| Random Forest | 100 árboles en paralelo, votación mayoritaria. Robusto. | F1: 0.7826 |
| Decision Tree | Un solo árbol con reglas if/else. Interpretable. | F1: 0.7500 |
| KNN | Busca los 5 vecinos más parecidos y mira su clase. | F1: 0.4000 |
| SVM | Encuentra un hiperplano que separa las clases con kernel RBF. | F1: 0.3077 |
| Logistic Regression | Fórmula lineal + curva sigmoide que da probabilidad. | F1: 0.2857 |

**Ganó Gradient Boosting.** Los modelos lineales (Logistic Regression, SVM) fallaron porque el problema NO es lineal. KNN sufrió por la maldición de la dimensionalidad (9 variables).

### Regresión — 6 modelos (target: eficiencia_ruta)

| Modelo | R² | MAE (km/h) |
|--------|-----|------------|
| **Gradient Boosting** | **0.9883** | **3.55** |
| Random Forest | 0.9872 | 5.24 |
| Decision Tree | 0.9681 | 4.35 |
| Linear Regression | 0.6943 | 36.37 |
| KNN | 0.6882 | 30.82 |
| SVR | 0.0621 | 51.44 |

**R²** mide cuánto explica el modelo la realidad (1.0 = perfecto). **MAE** mide el error promedio en km/h. Gradient Boosting explica el 98.8% de la eficiencia con solo 3.55 km/h de error. Linear Regression con R²=0.69 confirma que la relación no es lineal.

---

## Optimización de Hiperparámetros — GridSearchCV

GridSearchCV prueba **todas las combinaciones posibles** de hiperparámetros con validación cruzada de 5 folds, para encontrar la configuración óptima sin overfitting.

```python
param_grid = {
    'n_estimators': [50, 100, 200],    # cantidad de árboles
    'max_depth': [5, 10, None],         # profundidad máxima de cada árbol
    'min_samples_split': [2, 5, 10],    # mínimo de muestras para dividir un nodo
}
```

**27 combinaciones × 5 folds = 135 entrenamientos por tarea.**

| Tarea | Mejores parámetros | Impacto |
|-------|-------------------|---------|
| Clasificación | max_depth=5, n_estimators=50, min_samples_split=2 | Mejora +9.85% en F1 |
| Regresión | max_depth=10, n_estimators=200, min_samples_split=2 | Mejora marginal (0.9872 → 0.9883) |

- **n_estimators:** más árboles = más preciso pero más lento. Puede sobreajustar.
- **max_depth:** árboles profundos aprenden mejor pero memorizan los datos (overfitting). None = sin límite.
- **min_samples_split:** valores bajos (2) dividen más, valores altos (10) son más conservadores.

---

## No Supervisado — KMeans + PCA

KMeans agrupa envíos según sus características **sin usar etiquetas**. Se determinó **k=3** con el método del codo (Elbow Method). PCA redujo las 9 dimensiones a 2 para visualizar.

| Cluster | Envíos | Peso promedio | Distancia prom. | % Vehículo Adecuado |
|---------|--------|--------------|----------------|-------------------|
| 0 | 112 | 8.742 kg | 704 km | 17.9% |
| 1 | 109 | 18.654 kg | 1.398 km | 33.9% |
| 2 | 4 | 9.338 kg | **18.627 km** | 0.0% |

El Cluster 2 contiene 4 outliers con distancia extrema (18.627 km), probablemente errores de datos o rutas internacionales especiales.

---

## Visualizaciones

| Gráfico | Propósito |
|---------|-----------|
| `comparativa_modelos.png` | Barra horizontal comparando F1-Score de los 6 clasificadores |
| `matriz_confusion.png` | Aciertos vs errores del Gradient Boosting |
| `curva_roc.png` | Capacidad discriminativa del modelo (AUC=0.95) |
| `comparativa_regresores.png` | Barra horizontal comparando R² de los 6 regresores |
| `importancia_variables.png` | Qué variables pesan más en eficiencia de ruta |
| `top5_variables.png` | Top 5 variables para clasificación |
| `elbow_method.png` | Curva de inercia para elegir k óptimo en KMeans |
| `clusters_pca.png` | Clusters proyectados en 2D con centroides |
| `clusters_vs_target.png` | % adecuado y eficiencia promedio por cluster |

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
