import os
import sys
import time
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====================================================================
# CONFIGURACIÓN Y RUTAS
# ====================================================================
BASE_DIR = Path(__file__).resolve().parent
DIR_MONITOREO = BASE_DIR / "monitoreo"
DIR_MONITOREO.mkdir(parents=True, exist_ok=True)

RUTA_REGISTRO = DIR_MONITOREO / "registro_inferencias.csv"
RUTA_TRAIN = BASE_DIR / "data_procesada" / "smoke_detection_features.csv"
RUTA_MODELO = BASE_DIR / "modelos" / "modelo_mejor_validacion.joblib"
RUTA_SALIDA_TXT = DIR_MONITOREO / "salida_09.txt"

FEATURE_NAMES = [
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

COLUMNAS_REGISTRO = [
    "timestamp",
    "ip_origen",
    "temperature",
    "humidity",
    "raw_aire",
    "tvoc",
    "pm2_5",
    "eco2",
    "tvoc_mean_5",
    "prediccion",
    "probabilidad",
    "latencia_ms"
]


# ====================================================================
# VALIDACIÓN DE REGISTRO DE DATOS REALES
# ====================================================================
def validar_archivo_registro():
    """
    Verifica que el archivo de inferencias reales del ESP32 exista
    y contenga datos antes de iniciar la auditoría.
    """
    if not RUTA_REGISTRO.exists() or os.path.getsize(RUTA_REGISTRO) == 0:
        raise FileNotFoundError(
            f"[!] Error: No se encontró el archivo de registro en {RUTA_REGISTRO}. "
            "Asegúrese de generar lecturas con el ESP32 o el servidor antes de auditar."
        )


# ====================================================================
# CÁLCULO DE POPULATION STABILITY INDEX (PSI)
# ====================================================================
def calcular_psi(serie_esperada, serie_real, num_bins=10):
    """
    Calcula el Population Stability Index (PSI) usando deciles del conjunto
    de entrenamiento.
    """
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(serie_esperada, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    if len(bins) < 3:
        min_v = min(serie_esperada.min(), serie_real.min())
        max_v = max(serie_esperada.max(), serie_real.max())
        bins = np.linspace(min_v, max_v, num_bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.inf

    esperado_counts, _ = np.histogram(serie_esperada, bins=bins)
    real_counts, _ = np.histogram(serie_real, bins=bins)

    esperado_pct = esperado_counts / len(serie_esperada)
    real_pct = real_counts / len(serie_real)

    # Evitar divisiones por cero o logaritmos de cero
    eps = 1e-4
    esperado_pct = np.where(esperado_pct == 0, eps, esperado_pct)
    real_pct = np.where(real_pct == 0, eps, real_pct)

    esperado_pct /= np.sum(esperado_pct)
    real_pct /= np.sum(real_pct)

    psi_val = np.sum((real_pct - esperado_pct) * np.log(real_pct / esperado_pct))
    return float(psi_val)


# ====================================================================
# EVALUACIÓN DE RACHAS CONSECUTIVAS (RUN LENGTH ENCODING)
# ====================================================================
def obtener_racha_maxima(df, col):
    """
    Evalúa la racha más larga de valores repetidos consecutivos para una columna.
    Retorna:
      - max_run (int): longitud de la racha máxima.
      - valor_repetido: el valor repetido en dicha racha.
      - racha_df (DataFrame): subset con las filas completas de esa racha.
    """
    if col not in df.columns or len(df) == 0:
        return 0, None, df.iloc[0:0]

    serie = df[col]
    es_cambio = (serie != serie.shift(1))
    bloques = es_cambio.cumsum()
    tamanios = df.groupby(bloques).size()
    id_max = tamanios.idxmax()
    racha_df = df[bloques == id_max]
    max_run = len(racha_df)
    valor_repetido = racha_df[col].iloc[0]
    return max_run, valor_repetido, racha_df


# ====================================================================
# PROGRAMA PRINCIPAL DE AUDITORÍA Y MONITOREO
# ====================================================================
def ejecutar_monitoreo():
    validar_archivo_registro()

    salida_texto = []

    def registrar(texto=""):
        print(texto)
        salida_texto.append(str(texto))

    # Cargar datos
    df_live = pd.read_csv(RUTA_REGISTRO)
    df_train = pd.read_csv(RUTA_TRAIN)

    df_live["ts_dt"] = pd.to_datetime(df_live["timestamp"])

    registrar("=" * 85)
    registrar("SISTEMA DE MONITOREO CONTINUO DE INFERENCIA Y DRIFT (09_monitoreo.py)")
    registrar("=" * 85)

    # ----------------------------------------------------------------
    # 1. RESUMEN DE SESIÓN Y RENDIMIENTO
    # ----------------------------------------------------------------
    n_inferencias = len(df_live)
    t_inicio = df_live["ts_dt"].min()
    t_fin = df_live["ts_dt"].max()
    duracion = t_fin - t_inicio
    total_segundos = duracion.total_seconds()
    mins, segs = divmod(int(total_segundos), 60)
    freq_media = (n_inferencias / total_segundos) if total_segundos > 0 else 0.0

    n_alarmas = int((df_live["prediccion"] == 1).sum())
    pct_alarmas = (n_alarmas / n_inferencias * 100.0) if n_inferencias > 0 else 0.0

    lat_media = df_live["latencia_ms"].mean()
    lat_p95 = df_live["latencia_ms"].quantile(0.95)
    lat_max = df_live["latencia_ms"].max()

    # Detección de huecos > 5 segundos
    deltas_tiempo = df_live["ts_dt"].diff().dt.total_seconds()
    huecos = deltas_tiempo[deltas_tiempo > 5.0]
    n_huecos = len(huecos)

    registrar("\n1. RESUMEN GENERAL DE OPERACIÓN")
    registrar("-" * 65)
    registrar(f"Número total de inferencias       : {n_inferencias}")
    registrar(f"Inicio de sesión                  : {t_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    registrar(f"Fin de sesión                     : {t_fin.strftime('%Y-%m-%d %H:%M:%S')}")
    registrar(f"Duración de la sesión             : {mins} min {segs} s ({total_segundos:.1f} s)")
    registrar(f"Frecuencia media de inferencia    : {freq_media:.2f} lecturas/s")
    registrar(f"Eventos de alarma disparados      : {n_alarmas} ({pct_alarmas:.2f} %)")
    registrar(f"Latencia de inferencia media      : {lat_media:.4f} ms")
    registrar(f"Latencia Percentil 95 (P95)       : {lat_p95:.4f} ms")
    registrar(f"Latencia máxima observada         : {lat_max:.4f} ms")
    registrar(f"Huecos > 5 s sin lecturas         : {n_huecos} evento(s)")
    if n_huecos > 0:
        for idx_h, val_h in huecos.items():
            ts_gap = df_live.loc[idx_h, "timestamp"]
            registrar(f"  -> Hueco detectado en {ts_gap}: {val_h:.1f} s transcurridos")

    # Resumen de episodios de alarma (bloques consecutivos de prediccion == 1)
    is_alarm = df_live["prediccion"] == 1
    blocks = (~is_alarm).cumsum()
    alarm_blocks = df_live[is_alarm].groupby(blocks)
    n_episodios = alarm_blocks.ngroups

    registrar(f"\nResumen de episodios de alarma : {n_episodios} episodio(s) detectado(s)")
    if n_episodios > 0:
        for idx_ep, (_, group) in enumerate(alarm_blocks, start=1):
            t_ep_start = group["ts_dt"].iloc[0]
            t_ep_end = group["ts_dt"].iloc[-1]
            dur_ep = (t_ep_end - t_ep_start).total_seconds()
            max_tvoc_ep = group["tvoc"].max()
            max_raw_ep = group["raw_aire"].max() if "raw_aire" in group.columns else np.nan
            ts_start_str = group["timestamp"].iloc[0]
            ts_end_str = group["timestamp"].iloc[-1]
            registrar(f"  - Episodio {idx_ep}:")
            registrar(f"      Hora de inicio : {ts_start_str}")
            registrar(f"      Hora de fin    : {ts_end_str}")
            registrar(f"      Duración       : {dur_ep:.1f} s ({len(group)} lecturas consecutivas)")
            registrar(f"      TVOC máximo    : {max_tvoc_ep:.2f} ppb")
            registrar(f"      raw_aire máximo: {max_raw_ep:.1f}")

    # ----------------------------------------------------------------
    # 2. SALUD Y TELEMETRÍA DE SENSORES
    # ----------------------------------------------------------------
    temp_fuera = ((df_live["temperature"] < 0) | (df_live["temperature"] > 50)).sum()
    hum_fuera = ((df_live["humidity"] < 20) | (df_live["humidity"] > 95)).sum()

    registrar("\n2. ESTADO DE SALUD DE SENSORES FÍSICOS")
    registrar("-" * 65)
    registrar(f"Temperatura fuera de rango (0 - 50 °C) : {temp_fuera} muestra(s) " +
              (f"[{temp_fuera/n_inferencias*100:.2f} %]" if temp_fuera > 0 else "[CORRECTO]"))
    registrar(f"Humedad fuera de rango (20 - 95 %)     : {hum_fuera} muestra(s) " +
              (f"[{hum_fuera/n_inferencias*100:.2f} %]" if hum_fuera > 0 else "[CORRECTO]"))

    registrar("\nAuditoría de congelamiento de señal (umbral > 30 lecturas idénticas):")

    # Evaluar si raw_aire también se repite más de 30 veces seguidas
    max_run_raw, _, _ = obtener_racha_maxima(df_live, "raw_aire")
    raw_aire_congelado = bool(max_run_raw > 30)

    sensores_reposo = {
        "tvoc": 10.0,
        "eco2": 400.0,
        "pm2_5": 0.05
    }

    variables_congelamiento = ["temperature", "humidity", "tvoc", "eco2", "raw_aire", "pm2_5"]
    for v in variables_congelamiento:
        if v in df_live.columns:
            max_run, valor_rep, racha_df = obtener_racha_maxima(df_live, v)
            congelado = (max_run > 30)

            if not congelado:
                estado_cong = "Estable / Normal"
            else:
                if v in sensores_reposo:
                    val_esperado = sensores_reposo[v]
                    es_reposo = np.isclose(float(valor_rep), val_esperado, atol=0.01)
                    raw_en_reposo = bool((racha_df["raw_aire"] < 1500).all()) if "raw_aire" in racha_df.columns else False
                    if es_reposo and raw_en_reposo:
                        estado_cong = "Línea base (sensor en reposo)"
                    elif raw_aire_congelado:
                        estado_cong = "ALERTA: CONGELADO"
                    else:
                        estado_cong = "Estable / Normal"
                elif v == "humidity":
                    temp_varia = bool(racha_df["temperature"].nunique() > 1) if "temperature" in racha_df.columns else False
                    raw_varia = bool(racha_df["raw_aire"].nunique() > 1) if "raw_aire" in racha_df.columns else False
                    if temp_varia or raw_varia:
                        estado_cong = "Estable (resolución DHT11 1 %)"
                    elif raw_aire_congelado:
                        estado_cong = "ALERTA: CONGELADO"
                    else:
                        estado_cong = "Estable / Normal"
                elif v == "raw_aire":
                    estado_cong = "ALERTA: CONGELADO" if raw_aire_congelado else "Estable / Normal"
                else:
                    estado_cong = "ALERTA: CONGELADO" if raw_aire_congelado else "Estable / Normal"

            registrar(f"  - {v:<14}: {estado_cong:<34} (máx. repeticiones consecutivas: {max_run})")

    # ----------------------------------------------------------------
    # 3. DETECCIÓN DE DATA DRIFT (KS TEST Y PSI DE 10 BINS)
    # ----------------------------------------------------------------
    registrar("\n3. EVALUACIÓN DE DATA DRIFT RESPECTO AL DATASET DE ENTRENAMIENTO")
    registrar("-" * 85)

    mapeo_variables = [
        ("Temperature", "Temperature[C]", "temperature"),
        ("Humidity", "Humidity[%]", "humidity"),
        ("TVOC", "TVOC[ppb]", "tvoc"),
        ("eCO2", "eCO2[ppm]", "eco2")
    ]

    header_tabla = f"{'Variable':<14} {'Media Train':<14} {'Media Vivo':<14} {'KS Stat':<10} {'p-valor':<12} {'PSI (10)':<10} {'Estado':<15}"
    registrar(header_tabla)
    registrar("-" * 85)

    archivos_generados = []

    for nom_mostrar, col_train, col_live in mapeo_variables:
        train_vals = df_train[col_train].dropna()
        live_vals = df_live[col_live].dropna()

        m_train = float(train_vals.mean())
        m_live = float(live_vals.mean())

        ks_stat, p_val = ks_2samp(train_vals, live_vals)
        psi_val = calcular_psi(train_vals, live_vals, num_bins=10)

        if psi_val < 0.10:
            estado_drift = "Estable"
        elif 0.10 <= psi_val <= 0.25:
            estado_drift = "Moderado"
        else:
            estado_drift = "Drift"

        p_str = f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
        registrar(f"{nom_mostrar:<14} {m_train:<14.4f} {m_live:<14.4f} {ks_stat:<10.4f} {p_str:<12} {psi_val:<10.4f} {estado_drift:<15}")

    registrar("-" * 85)
    registrar("Criterio PSI: Estable (< 0.10) | Moderado (0.10 - 0.25) | Drift (> 0.25)")

    # ----------------------------------------------------------------
    # 4. GENERACIÓN DE GRÁFICOS DE MONITOREO
    # ----------------------------------------------------------------
    registrar("\n4. GENERACIÓN DE GRÁFICOS DIAGNÓSTICOS EN monitoreo/")
    registrar("-" * 65)

    indices = np.arange(len(df_live))

    # Gráfico 1: Probabilidad y predicción en el tiempo
    plt.figure(figsize=(10, 5.5))
    plt.subplot(2, 1, 1)
    plt.plot(indices, df_live["probabilidad"], color="#d9534f", linewidth=1.5, label="Probabilidad de Fuego")
    plt.axhline(0.5, color="#6c757d", linestyle="--", alpha=0.7, label="Umbral Decisión (0.5)")
    plt.ylabel("Probabilidad", fontsize=9)
    plt.title("Evolución de Probabilidad y Predicción de Alarma en Tiempo Real", fontsize=11, pad=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=8)

    plt.subplot(2, 1, 2)
    plt.step(indices, df_live["prediccion"], color="#b02a37", where="mid", linewidth=1.5, label="Predicción (0: Normal, 1: Alarma)")
    plt.xlabel("Muestra de Inferencia", fontsize=9)
    plt.ylabel("Clase Binaria", fontsize=9)
    plt.yticks([0, 1], ["0 (Normal)", "1 (Alarma)"])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    ruta_g1 = DIR_MONITOREO / "probabilidad_prediccion_tiempo.png"
    plt.savefig(ruta_g1, dpi=150)
    plt.close()
    archivos_generados.append(str(ruta_g1))
    registrar(f"  [OK] Guardado: {ruta_g1.name}")

    # Gráfico 2: Latencia en el tiempo
    plt.figure(figsize=(10, 4.5))
    plt.plot(indices, df_live["latencia_ms"], color="#0284c7", linewidth=1.2, label="Latencia por Inferencia")
    plt.axhline(lat_media, color="#16a34a", linestyle="--", linewidth=1.5, label=f"Media: {lat_media:.4f} ms")
    plt.axhline(lat_p95, color="#d97706", linestyle=":", linewidth=1.5, label=f"P95: {lat_p95:.4f} ms")
    plt.xlabel("Muestra de Inferencia", fontsize=9)
    plt.ylabel("Latencia (ms)", fontsize=9)
    plt.title("Latencia de Inferencia del Modelo en Tiempo Real", fontsize=11, pad=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    ruta_g2 = DIR_MONITOREO / "latencia_tiempo.png"
    plt.savefig(ruta_g2, dpi=150)
    plt.close()
    archivos_generados.append(str(ruta_g2))
    registrar(f"  [OK] Guardado: {ruta_g2.name}")

    # Gráfico 3: Histograma comparativo de entrenamiento contra vivo (Temperatura y TVOC)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # Subplot Temperatura
    axes[0].hist(df_train["Temperature[C]"].dropna(), bins=30, density=True, alpha=0.5, color="#3b82f6", label="Entrenamiento")
    axes[0].hist(df_live["temperature"].dropna(), bins=25, density=True, alpha=0.6, color="#ef4444", label="En Vivo")
    axes[0].set_title("Comparación de Distribución: Temperatura (°C)", fontsize=10)
    axes[0].set_xlabel("Temperatura [°C]", fontsize=9)
    axes[0].set_ylabel("Densidad de Probabilidad", fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8)

    # Subplot TVOC
    tvoc_train = np.clip(df_train["TVOC[ppb]"].dropna(), 0, 5000)
    tvoc_live = np.clip(df_live["tvoc"].dropna(), 0, 5000)
    axes[1].hist(tvoc_train, bins=30, density=True, alpha=0.5, color="#3b82f6", label="Entrenamiento")
    axes[1].hist(tvoc_live, bins=25, density=True, alpha=0.6, color="#ef4444", label="En Vivo")
    axes[1].set_title("Comparación de Distribución: TVOC (ppb)", fontsize=10)
    axes[1].set_xlabel("TVOC [ppb]", fontsize=9)
    axes[1].set_ylabel("Densidad de Probabilidad", fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    ruta_g3 = DIR_MONITOREO / "histograma_comparativo_temperatura_tvoc.png"
    plt.savefig(ruta_g3, dpi=150)
    plt.close()
    archivos_generados.append(str(ruta_g3))
    registrar(f"  [OK] Guardado: {ruta_g3.name}")

    # Guardar en salida_09.txt
    archivos_generados.append(str(RUTA_SALIDA_TXT))
    archivos_generados.append(str(RUTA_REGISTRO))
    with open(RUTA_SALIDA_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(salida_texto) + "\n")

    registrar(f"  [OK] Salida de consola guardada en: {RUTA_SALIDA_TXT.name}")
    registrar("\n" + "=" * 85)
    registrar("EJECUCIÓN DE 09_MONITOREO.PY COMPLETADA CON ÉXITO")
    registrar("=" * 85)


if __name__ == "__main__":
    ejecutar_monitoreo()

