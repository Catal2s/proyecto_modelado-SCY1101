# Resumen Ejecutivo - Proyecto Optimización Logística

## Objetivo
Desarrollar modelos predictivos para:
1. **Clasificación**: Predecir si un vehículo es adecuado para un envío
2. **Regresión**: Predecir la eficiencia de una ruta (km/h)

## Principales Hallazgos

### Clasificación (vehiculo_adecuado)
- **Mejor modelo**: Gradient Boosting (F1: 0.8333)
- Random Forest base: F1 0.7586
- Mejora vs base: +9.85%
- Hiperparámetros óptimos RF: max_depth=5, min_samples_split=2, n_estimators=50

### Regresión (eficiencia_ruta)
- **Mejor modelo**: Gradient Boosting (R²: 0.9883, MAE: 3.55 km/h)
- Random Forest base: R² 0.9872
- El modelo base ya era excelente; la mejora fue marginal

### Variables Clave
1. tiempo_estimado_hrs (65.7%)
2. distancia_km (33.7%)
3. peaje_total, peso_kg, año_fabricacion (< 1% cada una)

### Clustering (No Supervisado)
- 3 clusters identificados con KMeans
- Cluster 0: 112 envíos, peso medio, 17.9% vehículo adecuado
- Cluster 1: 109 envíos, peso alto, 33.9% vehículo adecuado
- Cluster 2: 4 envíos outlier, distancia extrema, 0% vehículo adecuado

## Recomendaciones para el Negocio
1. Implementar el clasificador Gradient Boosting para asignación automática de vehículos
2. Usar el regresor para planificación de rutas (error promedio de 3.55 km/h)
3. Registrar peso_kg y volumen_m3 con alta precisión (variables más determinantes)
4. Investigar los 4 envíos del cluster 2 (posibles errores de datos o rutas especiales)
