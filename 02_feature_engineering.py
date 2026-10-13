import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# 1. CARGA DEL DATASET LIMPIO
# =========================================================

base = Path(__file__).parent

ruta_entrada = (
    base
    / "data_procesada"
    / "smoke_detection_clean.csv"
)

df = pd.read_csv(ruta_entrada)

print("=" * 65)
print("1. DATASET DE ENTRADA")
print("=" * 65)

print(f"Filas iniciales   : {df.shape[0]}")
print(f"Columnas iniciales: {df.shape[1]}")


# =========================================================
# 2. IDENTIFICACIÓN DE SECUENCIAS
# =========================================================

# CNT es un contador de observaciones.
# Si el contador disminuye, consideramos que comienza
# una nueva secuencia de adquisición.
df["session_id"] = (
    df["CNT"].diff().fillna(0) < 0
).cumsum()

print("\n" + "=" * 65)
print("2. SECUENCIAS IDENTIFICADAS")
print("=" * 65)

print(
    "Cantidad de secuencias:",
    df["session_id"].nunique()
)


# =========================================================
# 3. RELACIÓN ENTRE MATERIAL PARTICULADO
# =========================================================

print("\n" + "=" * 65)
print("3. CREACIÓN DE PROPORCIÓN PM")
print("=" * 65)

# Se agrega un valor muy pequeño para evitar división entre cero.
epsilon = 1e-6

df["pm_ratio"] = (
    df["PM2.5"]
    / (df["PM1.0"] + epsilon)
)

print("Variable creada: pm_ratio")


# =========================================================
# 4. CAMBIOS ENTRE OBSERVACIONES CONSECUTIVAS
# =========================================================

print("\n" + "=" * 65)
print("4. CREACIÓN DE DELTAS")
print("=" * 65)

df["pm25_delta"] = (
    df.groupby("session_id")["PM2.5"]
    .diff()
    .fillna(0)
)

df["tvoc_delta"] = (
    df.groupby("session_id")["TVOC[ppb]"]
    .diff()
    .fillna(0)
)

df["eco2_delta"] = (
    df.groupby("session_id")["eCO2[ppm]"]
    .diff()
    .fillna(0)
)

df["temp_delta"] = (
    df.groupby("session_id")["Temperature[C]"]
    .diff()
    .fillna(0)
)

df["humidity_delta"] = (
    df.groupby("session_id")["Humidity[%]"]
    .diff()
    .fillna(0)
)

print("Variables creadas:")
print("- pm25_delta")
print("- tvoc_delta")
print("- eco2_delta")
print("- temp_delta")
print("- humidity_delta")


# =========================================================
# 5. PROMEDIOS MÓVILES
# =========================================================

print("\n" + "=" * 65)
print("5. CREACIÓN DE PROMEDIOS MÓVILES")
print("=" * 65)

# Ventana de 5 observaciones consecutivas.
# Solo utiliza la observación actual y las anteriores.

df["pm25_mean_5"] = (
    df.groupby("session_id")["PM2.5"]
    .transform(
        lambda x: x.rolling(
            window=5,
            min_periods=1
        ).mean()
    )
)

df["tvoc_mean_5"] = (
    df.groupby("session_id")["TVOC[ppb]"]
    .transform(
        lambda x: x.rolling(
            window=5,
            min_periods=1
        ).mean()
    )
)

df["eco2_mean_5"] = (
    df.groupby("session_id")["eCO2[ppm]"]
    .transform(
        lambda x: x.rolling(
            window=5,
            min_periods=1
        ).mean()
    )
)

print("Variables creadas:")
print("- pm25_mean_5")
print("- tvoc_mean_5")
print("- eco2_mean_5")


# =========================================================
# 6. VALIDACIÓN DE LAS NUEVAS VARIABLES
# =========================================================

print("\n" + "=" * 65)
print("6. VALIDACIÓN DE FEATURE ENGINEERING")
print("=" * 65)

features_nuevas = [
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

print(
    "Valores faltantes en nuevas variables:",
    df[features_nuevas].isnull().sum().sum()
)

print(
    "Valores infinitos en nuevas variables:",
    np.isinf(
        df[features_nuevas]
    ).sum().sum()
)


# =========================================================
# 7. EJEMPLO DE VARIABLES GENERADAS
# =========================================================

print("\n" + "=" * 65)
print("7. EJEMPLO DE CARACTERÍSTICAS GENERADAS")
print("=" * 65)

columnas_muestra = [
    "PM1.0",
    "PM2.5",
    "pm_ratio",
    "pm25_delta",
    "TVOC[ppb]",
    "tvoc_delta",
    "eCO2[ppm]",
    "eco2_delta",
    "pm25_mean_5"
]

print(
    df[columnas_muestra]
    .head(10)
    .round(3)
    .to_string(index=False)
)


# =========================================================
# 8. ELIMINACIÓN DEL IDENTIFICADOR AUXILIAR
# =========================================================

# session_id solo se utilizó para calcular correctamente
# las características temporales.
# No se utilizará como entrada del modelo.

df = df.drop(columns=["session_id"])


# =========================================================
# 9. RESULTADO FINAL
# =========================================================

print("\n" + "=" * 65)
print("8. RESULTADO DE INGENIERÍA DE CARACTERÍSTICAS")
print("=" * 65)

print(f"Filas finales   : {df.shape[0]}")
print(f"Columnas finales: {df.shape[1]}")

print("\nCaracterísticas nuevas:")

for feature in features_nuevas:
    print("-", feature)


# =========================================================
# 10. GUARDAR DATASET CON FEATURES
# =========================================================

ruta_salida = (
    base
    / "data_procesada"
    / "smoke_detection_features.csv"
)

df.to_csv(
    ruta_salida,
    index=False
)

print("\nDataset con ingeniería de características guardado en:")
print(ruta_salida)