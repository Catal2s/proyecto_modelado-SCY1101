# Guion de Presentación — 10 minutos
## Proyecto: Optimización Logística con Machine Learning

---

### [0:00 – 0:45] APERTURA — Objetivo y contexto

**Slide: Portada + Objetivos**

"Buenos días, mi nombre es [tu nombre] y voy a presentar mi proyecto de Machine Learning para optimización logística.

El objetivo fue desarrollar modelos predictivos para una empresa de logística usando sus datos reales. Planteamos **tres tareas de modelado**:

1. **Clasificación supervisada** — predecir si un vehículo es adecuado para un envío según su capacidad de peso y volumen.
2. **Regresión supervisada** — predecir la eficiencia de una ruta en km/h.
3. **Clustering no supervisado** — segmentar los envíos para encontrar patrones ocultos.

Trabajamos con **4 datasets**: envíos (1.030 registros), vehículos (61), rutas (82) e incidencias (206)."

---

### [0:45 – 2:15] ETAPA 1 — Preprocesamiento y Feature Engineering

**Slide: Pipeline de limpieza**

"Lo primero fue enfrentar la **calidad de los datos**. Teníamos problemas comunes en datos reales: valores como *'~500'*, *'$1.200'*, *'aprox 300'* en columnas numéricas, fechas en formatos mezclados, y nulos. Implementé una función `clean_numeric()` que usa expresiones regulares para extraer solo los valores numéricos válidos.

**Pasos clave de limpieza:**
- Fechas parseadas con `dayfirst=True` para detectar formato chileno DD/MM/YYYY.
- Valores nulos imputados con la mediana de cada columna (robusto ante outliers).
- Envíos con tiempo de entrega >30 días filtrados como outliers.
- IDs de ruta y vehículo nulos reemplazados con 0 para preservar el merge."

**Slide: Feature Engineering**

"Luego de limpiar los 4 datasets por separado, los fusioné mediante **left joins** usando `id_ruta` e `id_vehiculo` como llaves — esto nos dejó con un dataset unificado de ~225 registros después de los merges.

**Creación de targets:**
- `vehiculo_adecuado` [binario]: 1 si `capacidad_kg >= peso_kg` AND `capacidad_m3 >= volumen_m3`. Esto nos dio ~25% de casos positivos (desbalance moderado).
- `eficiencia_ruta` [continuo]: `distancia_km / (tiempo_estimado_hrs + 0.1)`. El +0.1 evita división por cero.

**Features finales usadas (9 variables):**
| Variable | Tipo | Descripción |
|----------|------|-------------|
| peso_kg | numérica | Peso de la carga |
| volumen_m3 | numérica | Volumen de la carga |
| distancia_km | numérica | Distancia de la ruta |
| tiempo_estimado_hrs | numérica | Tiempo estimado de viaje |
| peaje_total | numérica | Costo de peajes |
| capacidad_kg | numérica | Capacidad de peso del vehículo |
| capacidad_m3 | numérica | Capacidad de volumen del vehículo |
| km_recorridos | numérica | Kilometraje del vehículo |
| año_fabricacion | numérica | Año de fabricación del vehículo |

Todas fueron **escaladas con StandardScaler** (media 0, desviación 1) porque modelos como SVM y KNN son sensibles a escalas diferentes."

---

### [2:15 – 4:15] ETAPA 2 — Clasificación: Modelos y Resultados

**Slide: Modelos de clasificación**

"Para clasificación probamos **6 algoritmos**: Random Forest, Gradient Boosting, Decision Tree, KNN, SVM y Logistic Regression. Usamos **80/20 train/test split estratificado** para mantener la proporción de ~25% positivos en ambos conjuntos, más **validación cruzada de 5 folds** para robustez.

¿Por qué **F1-Score** como métrica principal? Porque tenemos desbalance de clases. Accuracy podría ser engañoso: un modelo que siempre prediga 'no adecuado' tendría ~75% de accuracy pero serviría de nada. F1 balancea precisión y recall."

**▶ MOSTRAR: `results/plots/comparativa_modelos.png`**

"Acá vemos la comparación de F1-Score:

| Modelo | F1-Score | CV F1 |
|--------|----------|-------|
| **Gradient Boosting** | **0.8333** | **0.8657** |
| Random Forest | 0.7826 | 0.6935 |
| Decision Tree | 0.7500 | 0.7344 |
| KNN | 0.4000 | 0.3582 |
| SVM | 0.3077 | 0.3531 |
| Logistic Regression | 0.2857 | 0.4472 |

**Interpretación:**
- **Gradient Boosting domina** con F1=0.8333 y CV F1=0.8657 — esto significa que generaliza bien, no hay overfitting significativo.
- La **brecha grande** entre Gradient Boosting/Random Forest vs el resto (>0.4 de diferencia con SVM, KNN, LR) indica que la frontera de decisión es compleja y no lineal.
- SVM tiene **Precision=1.0 pero Recall=0.18** — predice 'adecuado' solo en los casos más seguros, clasificando casi todo como negativo.
- KNN sufre porque con 9 dimensiones la distancia euclideana pierde efectividad (maldición de la dimensionalidad)."

**▶ MOSTRAR: `results/plots/matriz_confusion.png`**

"Esta es la matriz de confusión del Gradient Boosting. La diagonal principal concentra los aciertos. Podemos destacar:
- El modelo **solo se equivoca en ~4-5 casos** de ~45 de test (~91% accuracy).
- Tiende a cometer más **falsos positivos que falsos negativos** (predecir que hay vehículo adecuado cuando no lo hay). Para el negocio, este error es menos riesgoso: asignar un vehículo inadecuado es mejor que rechazar un envío viable."

**▶ MOSTRAR: `results/plots/curva_roc.png`**

"La curva ROC del Gradient Boosting muestra un **AUC de 0.95**, muy cerca del 1.0 ideal. La curva se dispara hacia arriba rápidamente, lo que indica que el modelo ordena muy bien las probabilidades: los positivos tienen puntuaciones altas y los negativos bajas, con mínima superposición.

Esto confirma que el modelo es **altamente discriminatorio** y apto para producción con un umbral ajustable según la tolerancia al riesgo del negocio."

---

### [4:15 – 6:15] ETAPA 3 — Regresión: Modelos y Resultados

**Slide: Modelos de regresión**

"Para regresión probamos otros 6 algoritmos: Gradient Boosting, Random Forest, Decision Tree, Linear Regression, KNN y SVR. El target fue `eficiencia_ruta` con valores entre ~0 y 500 km/h (filtramos outliers >500)."

**▶ MOSTRAR: `results/plots/comparativa_regresores.png`**

"Acá los resultados son muy claros:

| Modelo | R² | MAE | CV R² |
|--------|-----|-----|-------|
| **Gradient Boosting** | **0.9883** | **3.55** | **0.9875** |
| Random Forest | 0.9872 | 5.24 | 0.9628 |
| Decision Tree | 0.9681 | 4.35 | 0.9357 |
| Linear Regression | 0.6943 | 36.37 | 0.6920 |
| KNN | 0.6882 | 30.82 | 0.6659 |
| SVR | 0.0621 | 51.44 | 0.0161 |

**Análisis crítico:**
- Gradient Boosting y Random Forest son prácticamente **equivalentes en R²** (>0.987), pero Gradient Boosting gana en **MAE: 3.55 vs 5.24** — se equivoca 1.7 km/h menos en promedio.
- **Linear Regression con R²=0.69** confirma que la relación NO es lineal. Si lo fuera, la regresión lineal habría dado un resultado similar a Gradient Boosting.
- **SVR funciona pésimo (R²=0.06)** porque el kernel RBF sin optimización de hiperparámetros (C, gamma) no logra adaptarse a la distribución de los datos. Esto era esperable.
- La **validación cruzada** (CV R²) muestra que Gradient Boosting mantiene su rendimiento (0.9875), mientras que Random Forest baja a 0.9628 — indicio de que Random Forest tiene mayor varianza."

**▶ MOSTRAR: `results/plots/importancia_variables.png`**

"Para entender qué impulsa la eficiencia de ruta, veamos la importancia de variables del Gradient Boosting:

- **tiempo_estimado_hrs: 65.4%** — domina completamente. Tiene sentido: rutas con poco tiempo estimado pero larga distancia son más eficientes.
- **distancia_km: 31.8%** — segundo factor. Rutas más largas tienden a ser en autopista, más eficientes.
- **peaje_total, peso_kg, año_fabricación:** < 1% cada una — irrelevantes para predecir eficiencia.

Esto implica que para planificar rutas, las variables clave son tiempo y distancia, no el vehículo ni los peajes."

**▶ MOSTRAR: `results/plots/top5_variables.png`**

"Para clasificación (vehículo adecuado), las variables más importantes son:
- **peso_kg: 32.6%** y **volumen_m3: 21.7%** — lógico, porque el target se define directamente comparando peso y volumen contra capacidades.
- **km_recorridos: 9.3%** — vehículos con más kilometraje podrían tener más desgaste pero también ser más usados para ciertos tipos de carga.
- **capacidad_m3: 8.5%** y **capacidad_kg: 7.1%** — sorprendentemente no son las más importantes, pero el modelo las usa en interacción con peso y volumen."

---

### [6:15 – 7:15] ETAPA 4 — Optimización de Hiperparámetros

**Slide: GridSearchCV**

"Aplicamos **GridSearchCV con validación cruzada de 5 folds** sobre Random Forest para ambas tareas. ¿Por qué Random Forest y no Gradient Boosting? Porque Random Forest tiene menos hiperparámetros y es más rápido de optimizar. El espacio de búsqueda fue:

```python
param_grid = {
    'n_estimators': [50, 100, 200],     # número de árboles
    'max_depth': [5, 10, None],          # profundidad máxima
    'min_samples_split': [2, 5, 10],     # mínimo para dividir un nodo
}
```

**27 combinaciones × 5 folds = 135 entrenamientos** por tarea.

**Resultados clasificación:**
- Mejores params: `max_depth=5, min_samples_split=2, n_estimators=50`
- Interpretación: con **pocos árboles (50) y poca profundidad (5)** se evita overfitting en un dataset de ~180 registros de entrenamiento. Árboles más profundos aprendían ruido.
- Impacto: Random Forest base F1=0.7826 → Gradient Boosting (mejor general) F1=**0.8333** → mejora de **+9.85%**.

**Resultados regresión:**
- Mejores params: `max_depth=10, min_samples_split=2, n_estimators=200`
- Interpretación: para regresión se necesitan **más árboles (200) y más profundidad (10)** porque la variable continua requiere mayor capacidad de representación.
- Impacto: mejora marginal (R² 0.9872 → 0.9883). Cuando el modelo base ya es excelente, la optimización fina aporta poco.

**Conclusión sobre optimización:** GridSearch es ideal para proyectos académicos porque es exhaustivo y trazable. En producción usaría RandomizedSearchCV para eficiencia computacional."

---

### [7:15 – 8:45] ETAPA 5 — Clustering No Supervisado

**Slide: KMeans + PCA**

"Como análisis complementario, aplicamos **KMeans** para segmentar los envíos sin usar las etiquetas. Primero determinamos el número óptimo de clusters."

**▶ MOSTRAR: `results/plots/elbow_method.png`**

"Este es el **método del codo**. La inercia (suma de distancias intra-cluster) disminuye a medida que aumentamos k. El 'codo' se forma alrededor de **k=3**: después de ese punto, agregar más clusters reduce la inercia muy poco. Seleccionamos k=3."

**▶ MOSTRAR: `results/plots/clusters_pca.png`**

"Proyectamos los datos a 2 dimensiones con PCA para visualización. Las componentes explican el [xx]% y [xx]% de la varianza respectivamente.

Vemos **3 clusters** con los centroides marcados con X roja:
- **Cluster 0 (morado)** y **Cluster 1 (verde)** están cerca pero separados.
- **Cluster 2 (amarillo)** son 4 puntos alejados — claramente outliers."

**▶ MOSTRAR: `results/plots/clusters_vs_target.png`**

"Este gráfico de dos paneles es el más revelador para el negocio:

**Panel izquierdo — % Vehículo Adecuado por Cluster:**
- Cluster 0: **17.9%** — solo 1 de cada 5 envíos tiene vehículo adecuado.
- Cluster 1: **33.9%** — ~1 de cada 3, casi el doble.
- Cluster 2: **0%** — ninguno, consistente con ser outliers.

**Panel derecho — Eficiencia promedio por Cluster:**
- Cluster 0: ~153 km/h
- Cluster 1: ~193 km/h (mayor eficiencia)
- Cluster 2: ~1.129 km/h (valor irreal — 18.627 km de distancia con 16.4 hrs estimadas)

**Perfil de cada cluster:**

| Cluster | n | Peso promedio | Distancia prom. | % Adecuado | Eficiencia |
|---------|---|-------------|----------------|-----------|-----------|
| 0 | 112 | 8.742 kg | 704 km | 17.9% | 153 km/h |
| 1 | 109 | 18.654 kg | 1.398 km | 33.9% | 193 km/h |
| 2 | 4 | 9.338 kg | **18.627 km** | 0.0% | 1.129 km/h |

**Interpretación de negocio:**
- **Cluster 0**: envíos 'típicos' de peso y distancia medios — la flota actual no está bien dimensionada (baja tasa de adecuación).
- **Cluster 1**: cargas pesadas y largas distancias — cuando el envío es grande, se asigna mejor el vehículo (mayor adecuación).
- **Cluster 2**: 4 envíos con **distancia de 18.627 km** (¡cruzar Chile 30 veces!). Son claramente errores de datos o rutas internacionales especiales. Recomendación: investigar y corregir o excluir."

---

### [8:45 – 10:00] CIERRE — Lecciones, Recomendaciones y Conclusiones

**Slide: Lecciones aprendidas**

"**Tres lecciones técnicas importantes:**

1. **La limpieza de datos es lo que más tiempo tomó pero lo que más impacto tuvo.** Sin una función robusta `clean_numeric()`, los modelos habrían fallado. En proyectos reales, esta etapa ocupa ~60-80% del tiempo.

2. **Gradient Boosting fue consistentemente superior** tanto en clasificación (F1=0.8333) como en regresión (R²=0.9883). Esto confirma la literatura: para datos tabulares estructurados, los gradient boosted trees suelen ser la mejor opción.

3. **Target 'tiene_incidencia' fue descartado** por desbalance extremo (18 positivos vs 194 negativos). Intentamos modelarlo pero el F1 máximo fue 0.33. Esto enseñó que hay que diagnosticar la viabilidad del target antes de invertir tiempo en modelar."

**Slide: Recomendaciones para el negocio**

"**Cuatro recomendaciones concretas:**

1. **Implementar el clasificador Gradient Boosting** en el sistema de asignación de flota. Con F1=0.83, puede reducir significativamente errores de asignación. Sugiero empezar con un piloto en una región.

2. **Usar el regresor para planificación de rutas** — error de solo 3.55 km/h. Permite estimar tiempos de viaje con precisión y optimizar programación de entregas.

3. **Mejorar calidad del registro de datos:** peso_kg y volumen_m3 son las variables más determinantes pero justo son las que vienen más sucias. Si el negocio mejora la captura de estos datos, los modelos mejorarán automáticamente.

4. **Investigar los 4 envíos outlier del cluster 2.** Pueden ser errores de carga de datos (distancia mal registrada) o casos especiales que merecen tratamiento aparte."

**Slide: Cierre**

"En resumen: construimos un pipeline completo y reproducible de ML que aborda clasificación, regresión y clustering con resultados sólidos. El proyecto está **100% reproducible** — solo ejecutar `python run_pipeline.py` y se regeneran todos los resultados, métricas y gráficos.

**Posibles extensiones futuras:**
- Probar XGBoost y LightGBM como alternativas más rápidas.
- Incorporar variables temporales (estacionalidad, festivos).
- Construir dashboard interactivo con Streamlit.
- Recolectar más datos históricos para mejorar generalización.

Muchas gracias. ¿Preguntas?"
