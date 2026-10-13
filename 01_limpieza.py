import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# 1. CARGA DEL DATASET ORIGINAL
# =========================================================

# Busca el CSV en la misma carpeta donde se encuentra este script
ruta = Path(__file__).parent / "smoke_detection_iot.csv"

df = pd.read_csv(ruta)

print("=" * 60)
print("1. INFORMACIÓN INICIAL DEL DATASET")
print("=" * 60)

print(f"Filas originales   : {df.shape[0]}")
print(f"Columnas originales: {df.shape[1]}")

print("\nColumnas:")
for columna in df.columns:
    print("-", columna)


# =========================================================
# 2. VERIFICACIÓN DE VALORES FALTANTES
# =========================================================

print("\n" + "=" * 60)
print("2. VALORES FALTANTES")
print("=" * 60)

faltantes = df.isnull().sum()

print(faltantes)
print("\nTotal de valores faltantes:", faltantes.sum())


# =========================================================
# 3. VERIFICACIÓN DE DUPLICADOS
# =========================================================

print("\n" + "=" * 60)
print("3. REGISTROS DUPLICADOS")
print("=" * 60)

duplicados = df.duplicated().sum()

print("Cantidad de registros duplicados:", duplicados)


# =========================================================
# 4. VERIFICACIÓN DE VALORES INFINITOS
# =========================================================

print("\n" + "=" * 60)
print("4. VALORES INFINITOS")
print("=" * 60)

columnas_numericas = df.select_dtypes(include=np.number)

infinitos = np.isinf(columnas_numericas).sum().sum()

print("Cantidad total de valores infinitos:", infinitos)


# =========================================================
# 5. VALIDACIÓN BÁSICA DE VARIABLES
# =========================================================

print("\n" + "=" * 60)
print("5. VALIDACIÓN DE RANGOS")
print("=" * 60)

humedad_invalida = (
    (df["Humidity[%]"] < 0)
    | (df["Humidity[%]"] > 100)
).sum()

pm1_negativo = (df["PM1.0"] < 0).sum()
pm25_negativo = (df["PM2.5"] < 0).sum()

fire_alarm_invalido = (
    ~df["Fire Alarm"].isin([0, 1])
).sum()

print("Humedad fuera de 0-100 %   :", humedad_invalida)
print("PM1.0 negativo             :", pm1_negativo)
print("PM2.5 negativo             :", pm25_negativo)
print("Fire Alarm distinto de 0/1 :", fire_alarm_invalido)

print("\nRangos observados:")

print(
    "Temperatura:",
    df["Temperature[C]"].min(),
    "a",
    df["Temperature[C]"].max(),
    "°C"
)

print(
    "Humedad:",
    df["Humidity[%]"].min(),
    "a",
    df["Humidity[%]"].max(),
    "%"
)

print(
    "PM1.0:",
    df["PM1.0"].min(),
    "a",
    df["PM1.0"].max()
)

print(
    "PM2.5:",
    df["PM2.5"].min(),
    "a",
    df["PM2.5"].max()
)


# =========================================================
# 6. ELIMINACIÓN DE COLUMNA INNECESARIA
# =========================================================

print("\n" + "=" * 60)
print("6. ELIMINACIÓN DE COLUMNA AUXILIAR")
print("=" * 60)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("Columna 'Unnamed: 0' eliminada correctamente.")
else:
    print("La columna 'Unnamed: 0' no existe.")


# =========================================================
# 7. RESULTADO DESPUÉS DE LA LIMPIEZA
# =========================================================

print("\n" + "=" * 60)
print("7. DATASET DESPUÉS DE LA LIMPIEZA")
print("=" * 60)

print(f"Filas finales   : {df.shape[0]}")
print(f"Columnas finales: {df.shape[1]}")

print("\nValores faltantes finales:", df.isnull().sum().sum())
print("Duplicados finales:", df.duplicated().sum())


# =========================================================
# 8. GUARDAR DATASET LIMPIO
# =========================================================

# Crea la carpeta de salida dentro de la misma carpeta del script
carpeta_salida = Path(__file__).parent / "data_procesada"
carpeta_salida.mkdir(exist_ok=True)

ruta_salida = carpeta_salida / "smoke_detection_clean.csv"

df.to_csv(
    ruta_salida,
    index=False
)

print("\nDataset limpio guardado correctamente en:")
print(ruta_salida)