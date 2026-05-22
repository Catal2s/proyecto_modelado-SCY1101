import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'src')
import pandas as pd

from data_preprocessing import run_make_dataset, run_feature_engineering, prepare_regression_data, FEATURES
from model_training import (
    train_classification_models,
    train_regression_models,
    save_model,
)
from hyperparameter_tuning import (
    optimize_classifier,
    optimize_regressor,
)
from model_evaluation import (
    plot_classifier_comparison,
    plot_regressor_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    save_metrics_table,
    run_clustering,
)

def run_pipeline():
    print("=" * 70)
    print("  PIPELINE COMPLETO - PROYECTO OPTIMIZACIÓN LOGÍSTICA")
    print("=" * 70)

    os.makedirs("models/trained_models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    print("\n[1/6] Cargando y limpiando datos...")
    envios, incidencias, rutas, vehiculos = run_make_dataset("data")
    print(f"  OK Envios: {len(envios)} | Incidencias: {len(incidencias)} | Rutas: {len(rutas)} | Vehiculos: {len(vehiculos)}")

    print("\n[2/6] Ingeniería de características...")
    df, X, X_scaled, scaler, features = run_feature_engineering("data")
    _, X_reg_scaled, y_reg, scaler_reg = prepare_regression_data(df)
    print(f"  OK Dataset unificado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"  OK Features: {len(features)} variables")

    print("\n[3/6] Entrenando modelos de CLASIFICACIÓN (vehiculo_adecuado)...")
    y_clf = df['vehiculo_adecuado']
    clf_results, clf_models, (X_tr, X_te, y_tr, y_te) = train_classification_models(X_scaled, y_clf)
    print(f"  Mejor modelo: {clf_results.iloc[0]['Modelo']} (F1: {clf_results.iloc[0]['F1-Score']})")
    print(f"\n{clf_results.to_string(index=False)}")
    save_metrics_table(clf_results, "clasificacion_resultados.csv")
    plot_classifier_comparison(clf_results)

    best_clf_name = clf_results.iloc[0]['Modelo']
    best_clf = clf_models[best_clf_name]
    save_model(best_clf, "mejor_clasificador")
    plot_confusion_matrix(best_clf, X_te, y_te, labels=['No adecuado', 'Adecuado'])

    if hasattr(best_clf, "predict_proba"):
        plot_roc_curve(best_clf, X_te, y_te)

    print("\n  Optimizando hiperparámetros del Random Forest Classifier...")
    grid_clf = optimize_classifier(X_scaled, y_clf)
    print(f"  Mejores parámetros: {grid_clf.best_params_}")
    print(f"  Mejor F1 (CV): {grid_clf.best_score_:.4f}")
    best_clf_optimized = grid_clf.best_estimator_
    save_model(best_clf_optimized, "clasificador_optimizado")

    print("\n[4/6] Entrenando modelos de REGRESIÓN (eficiencia_ruta)...")
    reg_results, reg_models, (X_tr_r, X_te_r, y_tr_r, y_te_r) = train_regression_models(X_reg_scaled, y_reg)
    print(f"  Mejor modelo: {reg_results.iloc[0]['Modelo']} (R²: {reg_results.iloc[0]['R²']})")
    print(f"\n{reg_results.to_string(index=False)}")
    save_metrics_table(reg_results, "regresion_resultados.csv")
    plot_regressor_comparison(reg_results)

    best_reg_name = reg_results.iloc[0]['Modelo']
    best_reg = reg_models[best_reg_name]
    save_model(best_reg, "mejor_regresor")

    imp = plot_feature_importance(best_reg, features)
    if imp is not None:
        print(f"\n  Top 5 variables importantes:")
        for _, row in imp.head(5).iterrows():
            print(f"    {row['Variable']}: {row['Importancia']:.4f}")

    print("\n  Optimizando hiperparámetros del Random Forest Regressor...")
    grid_reg = optimize_regressor(X_reg_scaled, y_reg)
    print(f"  Mejores parámetros: {grid_reg.best_params_}")
    print(f"  Mejor R² (CV): {grid_reg.best_score_:.4f}")
    best_reg_optimized = grid_reg.best_estimator_
    save_model(best_reg_optimized, "regresor_optimizado")

    print("\n[5/6] Análisis NO SUPERVISADO (Clustering)...")
    kmeans, clusters = run_clustering(X_scaled, df.copy(), features, n_clusters=3)
    print(f"  OK Clusters generados: 3")
    save_model(kmeans, "kmeans_clustering")
    print(f"  Distribución: {pd.Series(clusters).value_counts().to_dict()}")

    print("\n[6/6] Resumen final...")
    clf_base_f1 = clf_results[clf_results['Modelo'] == 'Random Forest']['F1-Score'].values[0] if 'Random Forest' in clf_results['Modelo'].values else 0
    mejor_clf_f1 = clf_results.iloc[0]['F1-Score']
    mejora_clf = ((mejor_clf_f1 - clf_base_f1) / clf_base_f1 * 100) if clf_base_f1 > 0 else 0

    reg_base_r2 = reg_results[reg_results['Modelo'] == 'Random Forest']['R²'].values[0] if 'Random Forest' in reg_results['Modelo'].values else 0
    mejor_reg_r2 = reg_results.iloc[0]['R²']
    mejora_reg = ((mejor_reg_r2 - reg_base_r2) / reg_base_r2 * 100) if reg_base_r2 > 0 else 0

    print(f"""
  ========================================
  RESUMEN DEL PIPELINE
  ========================================
  CLASIFICACIÓN (vehiculo_adecuado):
    Mejor modelo: {best_clf_name} (F1: {mejor_clf_f1:.4f})
    Mejora vs Random Forest base: {mejora_clf:+.2f}%

  REGRESIÓN (eficiencia_ruta):
    Mejor modelo: {best_reg_name} (R²: {mejor_reg_r2:.4f})
    Mejora vs Random Forest base: {mejora_reg:+.2f}%

  MODELOS GUARDADOS:
    - models/trained_models/mejor_clasificador.joblib
    - models/trained_models/clasificador_optimizado.joblib
    - models/trained_models/mejor_regresor.joblib
    - models/trained_models/regresor_optimizado.joblib
    - models/trained_models/kmeans_clustering.joblib

  GRÁFICOS EN results/plots/
  MÉTRICAS EN results/metrics/
  ========================================
  """)

if __name__ == "__main__":
    run_pipeline()
