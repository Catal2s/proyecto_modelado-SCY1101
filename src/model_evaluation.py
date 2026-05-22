import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

def plot_classifier_comparison(df_results, save_path="results/plots/comparativa_modelos.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df_results.sort_values('F1-Score')
    ax.barh(df_plot['Modelo'], df_plot['F1-Score'], color='steelblue')
    ax.set_xlabel('F1-Score')
    ax.set_title('Clasificación - Comparación de F1-Score')
    ax.set_xlim(0, 1)
    ax.axvline(x=df_plot['F1-Score'].max(), color='green', linestyle='--',
               label=f"Mejor: {df_plot.iloc[-1]['Modelo']} ({df_plot.iloc[-1]['F1-Score']:.4f})")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_regressor_comparison(df_results, save_path="results/plots/comparativa_regresores.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df_results.sort_values('R²')
    ax.barh(df_plot['Modelo'], df_plot['R²'], color='coral')
    ax.set_xlabel('R²')
    ax.set_title('Regresión - Comparación de R²')
    ax.axvline(x=df_plot['R²'].max(), color='green', linestyle='--',
               label=f"Mejor: {df_plot.iloc[-1]['Modelo']} ({df_plot.iloc[-1]['R²']:.4f})")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(model, X_test, y_test, labels, save_path="results/plots/matriz_confusion.png"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, save_path="results/plots/importancia_variables.png"):
    if hasattr(model, 'feature_importances_'):
        importancias = pd.DataFrame({
            'Variable': feature_names,
            'Importancia': model.feature_importances_
        }).sort_values('Importancia', ascending=False)

        plt.figure(figsize=(10, 5))
        plt.barh(importancias['Variable'], importancias['Importancia'], color='steelblue')
        plt.xlabel('Importancia')
        plt.title('Importancia de Variables')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return importancias
    return None

def plot_roc_curve(model, X_test, y_test, save_path="results/plots/curva_roc.png"):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def save_metrics_table(df, filename, save_path="results/metrics"):
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, filename), index=False)

def run_clustering(X_scaled, df, features, n_clusters=3, random_state=42, save_dir="results"):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters

    inercia = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        inercia.append(km.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inercia, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.xticks(range(1, 11))
    plt.grid(alpha=0.3, linestyle='--')
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    plt.savefig(f"{save_dir}/plots/elbow_method.png", dpi=300, bbox_inches='tight')
    plt.close()

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis',
                         alpha=0.7, edgecolors='black', linewidth=0.5, s=80)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               marker='X', s=250, c='red', edgecolors='black', linewidth=2, label='Centroides')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Visualización de Clusters (KMeans + PCA)')
    ax.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plots/clusters_pca.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cluster_adecuado = df.groupby('cluster')['vehiculo_adecuado'].mean() * 100
    axes[0].bar(range(n_clusters), cluster_adecuado, color=plt.cm.viridis(np.linspace(0.2, 0.8, n_clusters)))
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Vehículo Adecuado (%)')
    axes[0].set_title('Vehículos Adecuados por Cluster')
    axes[0].set_xticks(range(n_clusters))
    axes[0].set_ylim(0, 100)

    cluster_eficiencia = df.groupby('cluster')['eficiencia_ruta'].mean()
    axes[1].bar(range(n_clusters), cluster_eficiencia, color=plt.cm.plasma(np.linspace(0.2, 0.8, n_clusters)))
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Eficiencia (km/h)')
    axes[1].set_title('Eficiencia Promedio por Cluster')
    axes[1].set_xticks(range(n_clusters))

    plt.tight_layout()
    plt.savefig(f"{save_dir}/plots/clusters_vs_target.png", dpi=300, bbox_inches='tight')
    plt.close()

    os.makedirs(f"{save_dir}/metrics", exist_ok=True)
    df[['id_envio', 'cluster'] + features].to_csv(f"{save_dir}/metrics/clusters_envios.csv", index=False)

    return kmeans, clusters
