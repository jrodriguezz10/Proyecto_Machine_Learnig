import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# =========================================================
# 1. CARGA DEL DATASET CON FEATURE ENGINEERING
# =========================================================

base = Path(__file__).parent

ruta_entrada = (
    base
    / "data_procesada"
    / "smoke_detection_features.csv"
)

df = pd.read_csv(ruta_entrada)

print("=" * 70)
print("1. DATASET DE ENTRADA")
print("=" * 70)

print(f"Filas     : {df.shape[0]}")
print(f"Columnas  : {df.shape[1]}")


# =========================================================
# 2. SELECCIÓN DE VARIABLES PREDICTORAS
# =========================================================

# UTC y CNT se excluyen del entrenamiento porque corresponden
# a información temporal/secuencial y podrían favorecer
# la memorización del orden de adquisición.

features = [
    "Temperature[C]",
    "Humidity[%]",
    "TVOC[ppb]",
    "eCO2[ppm]",
    "Raw H2",
    "Raw Ethanol",
    "Pressure[hPa]",
    "PM1.0",
    "PM2.5",
    "NC0.5",
    "NC1.0",
    "NC2.5",
    "pm_ratio",
    "pm25_delta",
    "tvoc_delta",
    "eco2_delta",
    "temp_delta",
    "humidity_delta",
    "pm25_mean_5",
    "tvoc_mean_5",
    "eco2_mean_5"
]

target = "Fire Alarm"

X = df[features].copy()
y = df[target].copy()

print("\n" + "=" * 70)
print("2. VARIABLES UTILIZADAS")
print("=" * 70)

print(f"Cantidad de características X: {X.shape[1]}")
print(f"Cantidad de observaciones     : {X.shape[0]}")

print("\nCaracterísticas seleccionadas:")
for feature in features:
    print("-", feature)

print("\nVariable objetivo:")
print("-", target)


# =========================================================
# 3. PARTICIÓN PREVIA PARA EVITAR DATA LEAKAGE
# =========================================================

# 70 % entrenamiento
# 15 % validación
# 15 % prueba

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\n" + "=" * 70)
print("3. PARTICIÓN PREVIA")
print("=" * 70)

print("Entrenamiento:", X_train.shape)
print("Validación   :", X_val.shape)
print("Prueba       :", X_test.shape)


# =========================================================
# 4. ANÁLISIS DE ESCALAS ANTES DE NORMALIZAR
# =========================================================

print("\n" + "=" * 70)
print("4. ESCALAS ANTES DE LA ESTANDARIZACIÓN")
print("=" * 70)

resumen_antes = X_train[
    [
        "Temperature[C]",
        "Humidity[%]",
        "TVOC[ppb]",
        "eCO2[ppm]",
        "PM1.0",
        "PM2.5"
    ]
].agg(["mean", "std", "min", "max"]).T

print(resumen_antes.round(3))


# =========================================================
# 5. ESTANDARIZACIÓN CON STANDARD SCALER
# =========================================================

scaler = StandardScaler()

# IMPORTANTE:
# El escalador se ajusta ÚNICAMENTE con los datos de entrenamiento.
X_train_scaled = scaler.fit_transform(X_train)

# Validación y prueba utilizan los parámetros aprendidos
# únicamente del conjunto de entrenamiento.
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# 6. CONVERSIÓN A DATAFRAME
# =========================================================

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=features,
    index=X_train.index
)

X_val_scaled = pd.DataFrame(
    X_val_scaled,
    columns=features,
    index=X_val.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=features,
    index=X_test.index
)


# =========================================================
# 7. VALIDACIÓN DE LA NORMALIZACIÓN
# =========================================================

print("\n" + "=" * 70)
print("5. VALIDACIÓN DE LA ESTANDARIZACIÓN")
print("=" * 70)

print("Media promedio de X_train escalado:")
print(round(X_train_scaled.mean().mean(), 6))

print("\nDesviación estándar promedio de X_train escalado:")
print(round(X_train_scaled.std(ddof=0).mean(), 6))

print(
    "\nValores faltantes después de escalar:",
    X_train_scaled.isnull().sum().sum()
)

print(
    "Valores infinitos después de escalar:",
    np.isinf(X_train_scaled).sum().sum()
)


# =========================================================
# 8. EJEMPLO ANTES Y DESPUÉS
# =========================================================

print("\n" + "=" * 70)
print("6. EJEMPLO ANTES Y DESPUÉS DE NORMALIZAR")
print("=" * 70)

columnas_ejemplo = [
    "Temperature[C]",
    "Humidity[%]",
    "TVOC[ppb]",
    "eCO2[ppm]",
    "PM1.0",
    "PM2.5"
]

print("\nANTES:")
print(
    X_train[columnas_ejemplo]
    .head(5)
    .round(3)
    .to_string(index=False)
)

print("\nDESPUÉS:")
print(
    X_train_scaled[columnas_ejemplo]
    .head(5)
    .round(3)
    .to_string(index=False)
)


# =========================================================
# 9. GUARDAR PARTICIONES
# =========================================================

carpeta_salida = base / "data_procesada" / "particiones"
carpeta_salida.mkdir(exist_ok=True)

# Datos SIN normalizar:
# se utilizarán para modelos basados en árboles.
X_train.to_csv(
    carpeta_salida / "X_train.csv",
    index=False
)

X_val.to_csv(
    carpeta_salida / "X_val.csv",
    index=False
)

X_test.to_csv(
    carpeta_salida / "X_test.csv",
    index=False
)

# Datos normalizados:
# se utilizarán especialmente para SVM.
X_train_scaled.to_csv(
    carpeta_salida / "X_train_scaled.csv",
    index=False
)

X_val_scaled.to_csv(
    carpeta_salida / "X_val_scaled.csv",
    index=False
)

X_test_scaled.to_csv(
    carpeta_salida / "X_test_scaled.csv",
    index=False
)

# Etiquetas
y_train.to_csv(
    carpeta_salida / "y_train.csv",
    index=False
)

y_val.to_csv(
    carpeta_salida / "y_val.csv",
    index=False
)

y_test.to_csv(
    carpeta_salida / "y_test.csv",
    index=False
)


# =========================================================
# 10. GUARDAR EL ESCALADOR
# =========================================================

ruta_scaler = (
    carpeta_salida
    / "standard_scaler.joblib"
)

joblib.dump(
    scaler,
    ruta_scaler
)


# =========================================================
# 11. RESULTADO FINAL
# =========================================================

print("\n" + "=" * 70)
print("7. RESULTADO FINAL")
print("=" * 70)

print("Características utilizadas:", len(features))

print("\nArchivos generados:")
print("- X_train.csv")
print("- X_val.csv")
print("- X_test.csv")
print("- X_train_scaled.csv")
print("- X_val_scaled.csv")
print("- X_test_scaled.csv")
print("- y_train.csv")
print("- y_val.csv")
print("- y_test.csv")
print("- standard_scaler.joblib")

print("\nArchivos guardados en:")
print(carpeta_salida)