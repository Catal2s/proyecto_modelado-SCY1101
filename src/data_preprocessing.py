import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURES = [
    'peso_kg', 'volumen_m3', 'distancia_km', 'tiempo_estimado_hrs',
    'peaje_total', 'capacidad_kg', 'capacidad_m3', 'km_recorridos',
    'año_fabricacion'
]


def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).replace('~', '').replace('$', '').replace('aprox', '').replace('aproximadamente', '').strip()
    try:
        return float(x)
    except:
        return np.nan


def load_raw_data(data_path="data"):
    envios = pd.read_csv(os.path.join(data_path, "envios.csv"))
    incidencias = pd.read_csv(os.path.join(data_path, "incidencias.csv"))
    rutas = pd.read_csv(os.path.join(data_path, "rutas.csv"))
    vehiculos = pd.read_csv(os.path.join(data_path, "vehiculos.csv"))
    return envios, incidencias, rutas, vehiculos


def clean_envios(envios):
    df = envios.copy()
    df['fecha_envio'] = pd.to_datetime(df['fecha_envio'], errors='coerce', dayfirst=True)
    df['fecha_entrega'] = pd.to_datetime(df['fecha_entrega'], errors='coerce', dayfirst=True)
    df['peso_kg'] = df['peso_kg'].apply(clean_numeric)
    df['volumen_m3'] = df['volumen_m3'].apply(clean_numeric)
    df = df.dropna(subset=['id_envio'])
    df = df.dropna(subset=['fecha_envio', 'fecha_entrega'])
    df['tiempo_entrega_dias'] = (df['fecha_entrega'] - df['fecha_envio']).dt.days
    df = df[df['tiempo_entrega_dias'] >= 0]
    df = df[df['tiempo_entrega_dias'] <= 30]
    df['id_ruta'] = df['id_ruta'].fillna(0)
    df['id_vehiculo'] = df['id_vehiculo'].fillna(0)
    df['peso_kg'] = df['peso_kg'].fillna(df['peso_kg'].median())
    df['volumen_m3'] = df['volumen_m3'].fillna(df['volumen_m3'].median())
    df['tipo_carga'] = df['tipo_carga'].fillna('DESCONOCIDO')
    df['estado'] = df['estado'].fillna('DESCONOCIDO')
    df['id_envio'] = df['id_envio'].astype(int)
    return df


def clean_incidencias(incidencias):
    df = incidencias.copy()
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['fecha'])
    df['costo_impacto'] = df['costo_impacto'].apply(clean_numeric)
    df['costo_impacto'] = df['costo_impacto'].fillna(0)
    df = df.dropna(subset=['id_envio'])
    df = df.dropna(subset=['id_incidencia'])
    df['tipo_incidencia'] = df['tipo_incidencia'].fillna('Desconocido')
    df['id_envio'] = df['id_envio'].astype(int)
    df['descripcion'] = df['descripcion'].fillna('Sin descripcion')
    return df


def clean_rutas(rutas):
    df = rutas.copy()
    df['distancia_km'] = pd.to_numeric(df['distancia_km'], errors='coerce')
    df['peaje_total'] = pd.to_numeric(df['peaje_total'], errors='coerce')
    df['peaje_total'] = df['peaje_total'].fillna(0)
    df['tiempo_estimado_hrs'] = pd.to_numeric(df['tiempo_estimado_hrs'], errors='coerce')
    df['tipo_via'] = df['tipo_via'].str.strip().str.upper()
    df['tipo_via'] = df['tipo_via'].fillna('DESCONOCIDO')
    df = df.dropna(subset=['id_ruta'])
    df['origen'] = df['origen'].fillna('DESCONOCIDO').str.strip().str.upper()
    df['destino'] = df['destino'].fillna('DESCONOCIDO').str.strip().str.upper()
    df['distancia_km'] = df['distancia_km'].fillna(df['distancia_km'].median())
    df['tiempo_estimado_hrs'] = df['tiempo_estimado_hrs'].fillna(df['tiempo_estimado_hrs'].median())
    return df


def clean_vehiculos(vehiculos):
    df = vehiculos.copy()
    df = df.dropna(subset=['id_vehiculo'])
    df['id_vehiculo'] = df['id_vehiculo'].astype(int)
    numeric_cols = ['capacidad_kg', 'capacidad_m3', 'km_recorridos', 'año_fabricacion']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    if 'tipo' in df.columns:
        df['tipo'] = df['tipo'].str.strip().str.upper().fillna('DESCONOCIDO')
    if 'estado_vehiculo' in df.columns:
        df['estado_vehiculo'] = df['estado_vehiculo'].str.strip().str.upper().fillna('DESCONOCIDO')
    return df


def save_clean_data(envios, incidencias, rutas, vehiculos, data_path="data"):
    os.makedirs(data_path, exist_ok=True)
    envios.to_csv(os.path.join(data_path, "envios_limpio.csv"), index=False)
    incidencias.to_csv(os.path.join(data_path, "incidencias_limpio.csv"), index=False)
    rutas.to_csv(os.path.join(data_path, "rutas_limpio.csv"), index=False)
    vehiculos.to_csv(os.path.join(data_path, "vehiculos_limpio.csv"), index=False)


def run_make_dataset(data_path="data"):
    envios, incidencias, rutas, vehiculos = load_raw_data(data_path)
    envios_clean = clean_envios(envios)
    incidencias_clean = clean_incidencias(incidencias)
    rutas_clean = clean_rutas(rutas)
    vehiculos_clean = clean_vehiculos(vehiculos)
    save_clean_data(envios_clean, incidencias_clean, rutas_clean, vehiculos_clean, data_path)
    return envios_clean, incidencias_clean, rutas_clean, vehiculos_clean


def load_clean_data(data_path="data"):
    envios = pd.read_csv(os.path.join(data_path, "envios_limpio.csv"))
    incidencias = pd.read_csv(os.path.join(data_path, "incidencias_limpio.csv"))
    rutas = pd.read_csv(os.path.join(data_path, "rutas_limpio.csv"))
    vehiculos = pd.read_csv(os.path.join(data_path, "vehiculos_limpio.csv"))
    return envios, incidencias, rutas, vehiculos


def merge_tables(envios, rutas, vehiculos):
    df = envios.merge(rutas, on='id_ruta', how='left')
    df = df.merge(vehiculos, on='id_vehiculo', how='left')
    return df


def create_targets(df, incidencias):
    ids_con_incidencia = incidencias['id_envio'].unique()
    df['tiene_incidencia'] = df['id_envio'].isin(ids_con_incidencia).astype(int)
    df['capacidad_kg'] = df['capacidad_kg'].fillna(df['capacidad_kg'].median())
    df['capacidad_m3'] = df['capacidad_m3'].fillna(df['capacidad_m3'].median())
    df['peso_kg'] = df['peso_kg'].fillna(df['peso_kg'].median())
    df['volumen_m3'] = df['volumen_m3'].fillna(df['volumen_m3'].median())
    df['vehiculo_adecuado'] = (
        (df['capacidad_kg'] >= df['peso_kg']) &
        (df['capacidad_m3'] >= df['volumen_m3'])
    ).astype(int)
    df['distancia_km'] = df['distancia_km'].fillna(df['distancia_km'].median())
    df['tiempo_estimado_hrs'] = df['tiempo_estimado_hrs'].fillna(df['tiempo_estimado_hrs'].median())
    df['eficiencia_ruta'] = df['distancia_km'] / (df['tiempo_estimado_hrs'] + 0.1)
    return df


def prepare_features(df, scaler=None):
    X = df[FEATURES].fillna(df[FEATURES].median())
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X, X_scaled, scaler


def prepare_regression_data(df, scaler=None):
    df_reg = df[(df['eficiencia_ruta'] > 0) & (df['eficiencia_ruta'] < 500)]
    X = df_reg[FEATURES].fillna(df_reg[FEATURES].median())
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    y = df_reg['eficiencia_ruta']
    return X, X_scaled, y, scaler


def run_feature_engineering(data_path="data"):
    envios, incidencias, rutas, vehiculos = load_clean_data(data_path)
    df = merge_tables(envios, rutas, vehiculos)
    df = create_targets(df, incidencias)
    X, X_scaled, scaler = prepare_features(df)
    return df, X, X_scaled, scaler, FEATURES
