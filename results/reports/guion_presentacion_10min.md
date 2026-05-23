# Guion de Presentación — 10 minutos
## Proyecto: Optimización Logística con Machine Learning

---

### [0:00 – 0:45] INTRODUCCIÓN

**Slide: Portada + Objetivos**
**En pantalla:** README.md (sección Resultados Clave)

"Buenos días, soy [tu nombre] y les presento mi proyecto de Machine Learning para optimización logística.

El objetivo fue desarrollar modelos predictivos para una empresa de logística usando sus datos reales. Planteamos **tres tareas**:

1. **Clasificación supervisada** — predecir si un vehículo es adecuado para un envío según capacidad de peso y volumen.
2. **Regresión supervisada** — predecir la eficiencia de una ruta en km/h.
3. **Clustering no supervisado** — segmentar los envíos para encontrar patrones ocultos sin usar etiquetas.

Trabajamos con **4 datasets**: envíos (1.030 registros), vehículos (61), rutas (82) e incidencias (206)."

---

### [0:45 – 2:15] DATOS Y PREPROCESAMIENTO

**Slide: Calidad de datos**
**En pantalla:** Notebook `01_EDA.ipynb` — celdas con valores sucios (~500, $1.200, aprox 300)

"Lo primero fue enfrentar la calidad de los datos. Teníamos valores como *'~500'*, *'$1.200'*, *'aprox 300'* en columnas que deberían ser numéricas, fechas en formatos mezclados, valores nulos en IDs de ruta y vehículo. Básicamente, datos reales, no datos de laboratorio.

Implementé una función `clean_numeric()` que usa expresiones regulares para extraer solo los dígitos válidos. También parseé fechas con `dayfirst=True` porque el formato chileno es DD/MM/YYYY, imputé valores nulos con la mediana y filtré outliers como envíos con más de 30 días de entrega."

**Slide: Feature Engineering**
**En pantalla:** Notebook `01_EDA.ipynb` — celdas donde se crean los targets

"Después de limpiar los 4 datasets por separado, los fusioné con **left joins** usando `id_ruta` e `id_vehiculo`. Esto nos dejó ~225 registros.

Luego creé los targets:
- `vehiculo_adecuado`: 1 si el vehículo cumple capacidad de peso Y volumen. Dio ~25% de casos positivos — hay desbalance.
- `eficiencia_ruta`: distancia dividida por tiempo estimado más 0.1 para evitar división por cero.

Las **features finales** fueron 9 variables numéricas escaladas con StandardScaler."

---

### [2:15 – 4:15] CLASIFICACIÓN

**Slide: Modelos de clasificación**
**En pantalla:** Notebook `02_supervised_modeling.ipynb` — celda con los 6 modelos

"Para clasificación probamos **6 algoritmos** para ver cuál funcionaba mejor:

| Modelo | Cómo funciona |
|--------|--------------|
| **Gradient Boosting** | Árboles en secuencia, cada uno corrige el error del anterior |
| Random Forest | 100 árboles en paralelo, votación mayoritaria |
| Decision Tree | Un solo árbol con reglas if/else |
| KNN | Busca los vecinos más parecidos y mira su clase |
| SVM | Encuentra un hiperplano que separa las clases |
| Logistic Regression | Fórmula lineal que da una probabilidad |

Usamos **F1-Score** como métrica principal porque hay desbalance de clases. Accuracy podía ser engañoso: un modelo que siempre prediga 'no adecuado' tendría ~75% de accuracy pero serviría de nada. F1 balancea precisión y recall."

**▶ MOSTRAR: `results/plots/comparativa_modelos.png`**

"Acá los resultados:

**Gradient Boosting ganó con F1=0.8333**, seguido por Random Forest con 0.7826.

**¿Por qué los otros fallaron?**
- **SVM y Logistic Regression** dieron F1 de 0.30 y 0.28. Esto prueba que **el problema NO es lineal** — si lo fuera, la regresión logística habría dado un resultado similar a Gradient Boosting.
- **KNN** dio solo 0.40. Con 9 variables, la distancia euclideana pierde efectividad — es la maldición de la dimensionalidad.
- La **validación cruzada** de Gradient Boosting dio 0.8657, incluso mejor que el F1 de test, lo que significa que el modelo generaliza bien."

**▶ MOSTRAR: `results/plots/matriz_confusion.png`**

"La matriz de confusión del Gradient Boosting muestra la diagonal principal bien cargada — la mayoría de predicciones son correctas, con solo ~4-5 errores de ~45 test. Tiende a cometer más falsos positivos que falsos negativos, que es el error menos riesgoso para el negocio."

**▶ MOSTRAR: `results/plots/curva_roc.png`**

"La curva ROC tiene un **AUC de 0.95**, muy cerca del 1.0 ideal. El modelo discrimina excelentemente entre ambas clases."

---

### [4:15 – 6:15] REGRESIÓN

**Slide: Modelos de regresión**
**En pantalla:** Notebook `02_supervised_modeling.ipynb` — celda de regresores

"Para regresión probamos otros 6 modelos para predecir eficiencia de ruta en km/h."

**▶ MOSTRAR: `results/plots/comparativa_regresores.png`**

"**Gradient Boosting ganó otra vez** con R²=0.9883 y MAE de solo 3.55 km/h.

**¿Qué es R² y MAE?**
- **R²** mide qué tanto explica el modelo la realidad. 1.0 es perfecto, 0.0 es no servir de nada. 0.9883 significa que el modelo explica el 98.8% de lo que pasa en los datos.
- **MAE** es el error promedio: 3.55 km/h significa que cuando predices eficiencia, te equivocas en promedio por 3.55 km/h.

Analicemos los resultados:
- **Linear Regression** dio R²=0.69. Si el problema fuera lineal, Gradient Boosting y Regresión Lineal habrían dado resultados parecidos. La diferencia enorme (0.99 vs 0.69) **confirma que la relación no es lineal**.
- **SVR** dio R²=0.06, básicamente no aprendió nada — necesitaba optimización de sus parámetros C y gamma.
- **Random Forest** casi empata en R² pero pierde en MAE (5.24 vs 3.55), se equivoca 1.7 km/h más."

**▶ MOSTRAR: `results/plots/importancia_variables.png`**

"¿Qué variables importan más para la eficiencia?
- **tiempo_estimado_hrs: 65.4%** — domina completamente.
- **distancia_km: 31.8%** — segundo factor.
- Las otras 7 variables juntas pesan menos del 3%.

Para planificar rutas eficientes, lo que importa son el tiempo estimado y la distancia. El vehículo y los peajes son irrelevantes."

**▶ MOSTRAR: `results/plots/top5_variables.png`**

"Para clasificación, las variables top son **peso_kg (32.6%)** y **volumen_m3 (21.7%)** — lógico porque el target se define comparando peso y volumen contra capacidades."

---

### [6:15 – 7:15] OPTIMIZACIÓN — GridSearchCV

**Slide: GridSearchCV**
**En pantalla:** Notebook `04_hyperparameter_optimization.ipynb` — celda con el grid

"Después de encontrar los mejores modelos, aplicamos **GridSearchCV** para optimizar Random Forest en ambas tareas.

**¿Qué es GridSearchCV?** Le dices al modelo 'prueba todas estas combinaciones y dime cuál funciona mejor'. CV = validación cruzada de 5 folds, así no confía en un solo resultado.

**Los parámetros que optimizamos:**
- **n_estimators:** 50, 100 o 200 — ¿cuántos árboles pongo? Más árboles = más precisión pero más lento.
- **max_depth:** 5, 10 o sin límite — ¿qué tan profundo crece cada árbol? Poco profundo = no aprende bien, muy profundo = memoriza los datos (overfitting).
- **min_samples_split:** 2, 5 o 10 — ¿cuántos datos mínimos para dividir un nodo? 2 divide mucho (overfitting), 10 es más conservador.

27 combinaciones × 5 folds = 135 entrenamientos por tarea.

**Resultados:**
- **Clasificación:** ganó `max_depth=5, n_estimators=50` — con ~180 datos de entrenamiento, árboles profundos causaban overfitting. Mejor simple.
- **Regresión:** ganó `max_depth=10, n_estimators=200` — al predecir un número continuo se necesita más capacidad.

El impacto: en clasificación subimos de F1 0.78 a 0.83 (+9.85%). En regresión la mejora fue marginal porque el modelo base ya era excelente."

---

### [7:15 – 8:45] CLUSTERING NO SUPERVISADO

**Slide: KMeans + PCA**
**En pantalla:** Notebook `06_unsupervised_analysis.ipynb` — celda de KMeans

"Como análisis complementario, aplicamos **KMeans** para segmentar los envíos sin usar las etiquetas.

**¿Qué hace KMeans?** Agrupa datos según su cercanía, sin saber qué son. Encuentra patrones ocultos."

**▶ MOSTRAR: `results/plots/elbow_method.png`**

"Para elegir k, usamos el **método del codo**. Graficamos la inercia vs número de clusters. Donde la curva se dobla (el codo) está el k ideal. Acá se dobla en **k=3**."

**▶ MOSTRAR: `results/plots/clusters_pca.png`**

"PCA reduce las 9 dimensiones a 2 para poder graficar. Vemos 3 clusters con sus centroides (marcados con X roja). Los clusters 0 y 1 están cerca pero separados. El cluster 2 son 4 puntos alejados."

**▶ MOSTRAR: `results/plots/clusters_vs_target.png`**

"Este gráfico es el más interesante para el negocio — dos paneles:

**Panel izquierdo — % vehículo adecuado:**
- Cluster 0: 17.9% — solo 1 de cada 5 tiene vehículo adecuado.
- Cluster 1: 33.9% — casi el doble.
- Cluster 2: 0% — ninguno.

**Panel derecho — Eficiencia promedio:**
- Cluster 0: ~153 km/h
- Cluster 1: ~193 km/h
- Cluster 2: ~1.129 km/h (valor irreal)

**Perfil de cada cluster:**
- **Cluster 0** (112 envíos): carga media, poca adecuación — la flota no está bien dimensionada.
- **Cluster 1** (109 envíos): carga pesada, larga distancia — cuando el envío es grande, asignan mejor el vehículo.
- **Cluster 2** (4 envíos): **distancia de 18.627 km** — cruzar Chile 30 veces. Son errores de datos o rutas especiales."

---

### [8:45 – 10:00] CONCLUSIONES Y RECOMENDACIONES

**Slide: Conclusiones**
**En pantalla:** README.md — Resultados Clave

"**Tres conclusiones principales:**

1. **Gradient Boosting fue el mejor modelo** tanto en clasificación como en regresión. Esto confirma la literatura: para datos tabulares, los gradient boosted trees son la mejor opción.

2. **La limpieza de datos fue clave.** Sin una función robusta para limpiar valores numéricos, los modelos habrían fallado. En proyectos reales esto toma ~60-80% del tiempo.

3. **Intentamos predecir `tiene_incidencia` pero lo descartamos** por desbalance extremo (18 casos positivos vs 194 negativos). Fue una lección: hay que diagnosticar la viabilidad del target antes de modelar."

**Slide: Recomendaciones**

"**Cuatro recomendaciones para el negocio:**
1. **Implementar el clasificador** en el sistema de asignación de flota — F1=0.83 es sólido para producción.
2. **Usar el regresor para planificación** — error de solo 3.55 km/h.
3. **Mejorar el registro de peso_kg y volumen_m3** — son las variables más importantes pero las que vienen más sucias.
4. **Investigar los 4 outliers del cluster 2** — pueden ser errores de datos o rutas especiales."

**Slide: Cierre**

"El proyecto está **100% reproducible**: solo ejecutar `python run_pipeline.py` y se regeneran métricas, gráficos y modelos.

Próximos pasos: probar XGBoost, incorporar datos temporales y construir un dashboard.

Muchas gracias. ¿Preguntas?"
