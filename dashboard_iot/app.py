import os
import sys
import time
import socket
import csv
from threading import Lock
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ====================================================================
# 1. CONFIGURACIÓN Y RUTAS
# ====================================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_PATH = PROJECT_DIR / "modelos" / "modelo_mejor_validacion.joblib"

if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_DIR / "modelos" / "decision_tree.joblib"

print(f"[*] Cargando modelo desde: {MODEL_PATH}")
modelo = joblib.load(MODEL_PATH)
print(f"[OK] Modelo cargado con exito: {type(modelo).__name__}")

# Nombres exactos de las 21 características esperadas por el modelo
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

# ====================================================================
# 2. DETECCIÓN DE IP LOCAL
# ====================================================================
def obtener_ip_local():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

LOCAL_IP = obtener_ip_local()
PORT = 8000

# ====================================================================
# 3. BUFFER TEMPORAL PARA INGENIERÍA DE CARACTERÍSTICAS EN VIVO
# ====================================================================
class HistorialSensores:
    def __init__(self, ventana: int = 5):
        self.ventana = ventana
        self.buffer = deque(maxlen=ventana)
        self.ultima_lectura = None
        self.historial_graficas = deque(maxlen=50)

    def procesar_lectura(self, lectura: dict):
        # 1. Cálculo de pm_ratio
        pm25 = float(lectura.get("pm2_5", 0.0))
        pm10 = float(lectura.get("pm1_0", 0.0))
        pm_ratio = pm25 / (pm10 + 1e-6)

        # 2. Cálculo de deltas respecto a la última observación
        if self.ultima_lectura is not None:
            pm25_delta = pm25 - self.ultima_lectura["pm2_5"]
            tvoc_delta = float(lectura.get("tvoc", 0.0)) - self.ultima_lectura["tvoc"]
            eco2_delta = float(lectura.get("eco2", 0.0)) - self.ultima_lectura["eco2"]
            temp_delta = float(lectura.get("temperature", 0.0)) - self.ultima_lectura["temperature"]
            humidity_delta = float(lectura.get("humidity", 0.0)) - self.ultima_lectura["humidity"]
        else:
            pm25_delta = 0.0
            tvoc_delta = 0.0
            eco2_delta = 0.0
            temp_delta = 0.0
            humidity_delta = 0.0

        # Guardar en buffer móvil
        self.buffer.append({
            "pm2_5": pm25,
            "tvoc": float(lectura.get("tvoc", 0.0)),
            "eco2": float(lectura.get("eco2", 0.0))
        })
        self.ultima_lectura = {
            "pm2_5": pm25,
            "tvoc": float(lectura.get("tvoc", 0.0)),
            "eco2": float(lectura.get("eco2", 0.0)),
            "temperature": float(lectura.get("temperature", 0.0)),
            "humidity": float(lectura.get("humidity", 0.0))
        }

        # 3. Cálculo de promedios móviles (últimas 5 observaciones)
        pm25_mean_5 = float(np.mean([x["pm2_5"] for x in self.buffer]))
        tvoc_mean_5 = float(np.mean([x["tvoc"] for x in self.buffer]))
        eco2_mean_5 = float(np.mean([x["eco2"] for x in self.buffer]))

        # Construcción del vector de 21 características
        fila_features = {
            "Temperature[C]": float(lectura.get("temperature", 20.0)),
            "Humidity[%]": float(lectura.get("humidity", 50.0)),
            "TVOC[ppb]": float(lectura.get("tvoc", 0.0)),
            "eCO2[ppm]": float(lectura.get("eco2", 400.0)),
            "Raw H2": float(lectura.get("raw_h2", 12500.0)),
            "Raw Ethanol": float(lectura.get("raw_ethanol", 19000.0)),
            "Pressure[hPa]": float(lectura.get("pressure", 939.0)),
            "PM1.0": pm10,
            "PM2.5": pm25,
            "NC0.5": float(lectura.get("nc0_5", 0.0)),
            "NC1.0": float(lectura.get("nc1_0", 0.0)),
            "NC2.5": float(lectura.get("nc2_5", 0.0)),
            "pm_ratio": pm_ratio,
            "pm25_delta": pm25_delta,
            "tvoc_delta": tvoc_delta,
            "eco2_delta": eco2_delta,
            "temp_delta": temp_delta,
            "humidity_delta": humidity_delta,
            "pm25_mean_5": pm25_mean_5,
            "tvoc_mean_5": tvoc_mean_5,
            "eco2_mean_5": eco2_mean_5
        }

        df_features = pd.DataFrame([fila_features])[FEATURE_NAMES]
        return df_features, fila_features

historial = HistorialSensores()

# ====================================================================
# 4. APLICACIÓN FASTAPI Y GESTIÓN WEBSOCKETS
# ====================================================================
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Monitor de Calidad de Aire y Alarma IoT - UPAO")

# Montar carpeta estática local para servir Chart.js 100% offline
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

# ====================================================================
# MONITOREO LOCAL Y REGISTRO DE INFERENCIAS
# ====================================================================
MONITOR_DIR = PROJECT_DIR / "monitoreo"
REGISTRO_CSV = MONITOR_DIR / "registro_inferencias.csv"
csv_lock = Lock()

ultima_lectura_esp32: Optional[float] = None
total_inferencias: int = 0
total_alarmas: int = 0
acumulador_latencias: List[float] = []

COLUMNAS_MONITOREO = [
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

def registrar_inferencia_csv(
    timestamp: str,
    ip_origen: str,
    temperature: float,
    humidity: float,
    raw_aire: Optional[float],
    tvoc: float,
    pm2_5: float,
    eco2: float,
    tvoc_mean_5: float,
    prediccion: int,
    probabilidad: float,
    latencia_ms: float
):
    global total_inferencias, total_alarmas, acumulador_latencias, ultima_lectura_esp32
    ultima_lectura_esp32 = time.time()
    total_inferencias += 1
    if prediccion == 1:
        total_alarmas += 1
    acumulador_latencias.append(latencia_ms)

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    with csv_lock:
        es_nuevo = not REGISTRO_CSV.exists()
        with open(REGISTRO_CSV, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if es_nuevo:
                writer.writerow(COLUMNAS_MONITOREO)
            writer.writerow([
                timestamp,
                ip_origen,
                round(temperature, 2),
                round(humidity, 2),
                raw_aire if raw_aire is not None else "",
                round(tvoc, 2),
                round(pm2_5, 3),
                round(eco2, 2),
                round(tvoc_mean_5, 2),
                prediccion,
                round(probabilidad, 4),
                round(latencia_ms, 4)
            ])

# Modelo de datos de entrada desde ESP32
class LecturaSensor(BaseModel):
    temperature: float = Field(..., example=24.5)
    humidity: float = Field(..., example=55.2)
    tvoc: float = Field(..., example=45.0)
    eco2: float = Field(..., example=412.0)
    raw_h2: float = Field(default=12800.0, example=12850.0)
    raw_ethanol: float = Field(default=19400.0, example=19450.0)
    pressure: float = Field(default=939.5, example=939.5)
    pm1_0: float = Field(..., example=1.5)
    pm2_5: float = Field(default=0.0, example=1.8)
    pm25: Optional[float] = Field(default=None, example=1.8)
    nc0_5: float = Field(default=10.0, example=11.2)
    nc1_0: float = Field(default=1.5, example=1.7)
    nc2_5: float = Field(default=0.04, example=0.04)
    raw_aire: Optional[float] = Field(default=None, example=1260.0)
    device_id: Optional[str] = "ESP32_UPAO"

# ====================================================================
# 5. ENDPOINTS DE LA API
# ====================================================================
@app.get("/api/info")
def get_info():
    return {
        "local_ip": LOCAL_IP,
        "port": PORT,
        "endpoint_post": f"http://{LOCAL_IP}:{PORT}/api/predict",
        "model_name": type(modelo).__name__,
        "features_count": len(FEATURE_NAMES)
    }

@app.get("/api/health")
def get_health():
    """
    Monitoreo de salud del servicio de inferencia y conectividad del ESP32.
    """
    ahora = time.time()
    if ultima_lectura_esp32 is not None:
        segundos_desde_ultima = round(ahora - ultima_lectura_esp32, 2)
        esp32_conectado = bool(segundos_desde_ultima <= 5.0)
    else:
        segundos_desde_ultima = None
        esp32_conectado = False

    latencia_promedio = round(float(np.mean(acumulador_latencias)), 4) if acumulador_latencias else 0.0
    porcentaje_alarmas = round((total_alarmas / total_inferencias * 100.0), 2) if total_inferencias > 0 else 0.0

    return {
        "estado_modelo": "cargado" if modelo is not None else "no_cargado",
        "modelo": type(modelo).__name__,
        "segundos_desde_ultima_lectura": segundos_desde_ultima,
        "esp32_conectado": esp32_conectado,
        "total_inferencias": total_inferencias,
        "latencia_promedio": latencia_promedio,
        "porcentaje_alarmas": porcentaje_alarmas
    }

@app.post("/api/predict")
async def predict_sensor_data(lectura: LecturaSensor, request: Request):
    """
    Endpoint principal para recibir datos desde el ESP32 o simulador.
    """
    datos_dict = lectura.model_dump() if hasattr(lectura, "model_dump") else lectura.dict()
    df_features, features_calc = historial.procesar_lectura(datos_dict)

    # Inferencia del modelo de ML
    prediccion = int(modelo.predict(df_features)[0])
    
    # Medición de latencia_ms con time.perf_counter() alrededor de predict_proba
    t0 = time.perf_counter()
    probabilidad = 1.0 if prediccion == 1 else 0.0
    if hasattr(modelo, "predict_proba"):
        try:
            prob_arr = modelo.predict_proba(df_features)[0]
            probabilidad = float(prob_arr[1])
        except Exception:
            pass
    latencia_ms = (time.perf_counter() - t0) * 1000.0

    hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ip_cliente = request.client.host if request.client else "local"

    # Registrar en monitoreo/registro_inferencias.csv
    raw_aire_val = datos_dict.get("raw_aire")
    if raw_aire_val is None:
        raw_aire_val = datos_dict.get("rawAire")

    pm25_val = datos_dict.get("pm2_5")
    if pm25_val is None:
        pm25_val = datos_dict.get("pm25", 0.0)

    registrar_inferencia_csv(
        timestamp=hora_actual,
        ip_origen=ip_cliente,
        temperature=float(datos_dict.get("temperature", 0.0)),
        humidity=float(datos_dict.get("humidity", 0.0)),
        raw_aire=float(raw_aire_val) if raw_aire_val is not None else None,
        tvoc=float(datos_dict.get("tvoc", 0.0)),
        pm2_5=float(pm25_val),
        eco2=float(datos_dict.get("eco2", 0.0)),
        tvoc_mean_5=float(features_calc.get("tvoc_mean_5", 0.0)),
        prediccion=prediccion,
        probabilidad=probabilidad,
        latencia_ms=latencia_ms
    )
    
    # Respuesta estructurada para el ESP32 y el Dashboard
    resultado = {
        "status": "ok",
        "fire_alarm": prediccion,
        "alarm_active": bool(prediccion == 1),
        "probability_fire": round(probabilidad, 4),
        "action": "TRIGGER_BUZZER_AND_LED" if prediccion == 1 else "NORMAL_MONITORING",
        "message": "PELIGRO: Humo o combustión detectada!" if prediccion == 1 else "Aire limpio / Condiciones normales",
        "timestamp": hora_actual,
        "origin_ip": ip_cliente,
        "device_id": lectura.device_id,
        "readings": datos_dict
    }

    # Guardar en historial de memoria para recargas rápidas
    historial.historial_graficas.append(resultado)

    # Transmitir a todos los clientes web conectados al Dashboard
    await manager.broadcast({
        "type": "NEW_READING",
        "payload": resultado
    })

    return resultado

@app.get("/api/history")
def get_history():
    return list(historial.historial_graficas)

# Datos de muestra del dataset original para simular con exactitud
MUESTRAS_LIMPIAS = [
    {"temperature": 20.0, "humidity": 57.36, "tvoc": 0.0, "eco2": 400.0, "raw_h2": 12306.0, "raw_ethanol": 18520.0, "pressure": 939.735, "pm1_0": 0.0, "pm2_5": 0.0, "nc0_5": 0.0, "nc1_0": 0.0, "nc2_5": 0.0},
    {"temperature": 20.05, "humidity": 55.34, "tvoc": 0.0, "eco2": 400.0, "raw_h2": 12560.0, "raw_ethanol": 19604.0, "pressure": 939.810, "pm1_0": 0.01, "pm2_5": 0.01, "nc0_5": 0.05, "nc1_0": 0.01, "nc2_5": 0.0},
    {"temperature": 20.10, "humidity": 54.12, "tvoc": 1.0, "eco2": 400.0, "raw_h2": 12612.0, "raw_ethanol": 19680.0, "pressure": 939.780, "pm1_0": 0.02, "pm2_5": 0.02, "nc0_5": 0.10, "nc1_0": 0.02, "nc2_5": 0.0}
]

MUESTRAS_FUEGO = [
    {"temperature": 15.2, "humidity": 52.8, "tvoc": 850.0, "eco2": 520.0, "raw_h2": 13340.0, "raw_ethanol": 20160.0, "pressure": 939.5, "pm1_0": 2.15, "pm2_5": 2.24, "nc0_5": 14.8, "nc1_0": 2.31, "nc2_5": 0.052},
    {"temperature": 24.8, "humidity": 48.5, "tvoc": 1420.0, "eco2": 680.0, "raw_h2": 12909.0, "raw_ethanol": 19428.0, "pressure": 938.7, "pm1_0": 3.85, "pm2_5": 4.12, "nc0_5": 26.5, "nc1_0": 4.13, "nc2_5": 0.093},
    {"temperature": 26.5, "humidity": 45.1, "tvoc": 2150.0, "eco2": 890.0, "raw_h2": 12450.0, "raw_ethanol": 19120.0, "pressure": 937.9, "pm1_0": 6.42, "pm2_5": 7.10, "nc0_5": 44.2, "nc1_0": 7.02, "nc2_5": 0.180}
]

indice_clean = 0
indice_fuego = 0

@app.get("/api/simulate/{scenario}")
async def simulate(scenario: str, request: Request):
    """
    Permite probar el sistema directamente desde el navegador sin hardware conectado.
    """
    global indice_clean, indice_fuego
    if scenario == "fire":
        base = MUESTRAS_FUEGO[indice_fuego % len(MUESTRAS_FUEGO)].copy()
        indice_fuego += 1
        # Pequeña variación estocástica
        base["tvoc"] += np.random.uniform(-20, 20)
        base["pm2_5"] += np.random.uniform(-0.1, 0.2)
    else:
        base = MUESTRAS_LIMPIAS[indice_clean % len(MUESTRAS_LIMPIAS)].copy()
        indice_clean += 1
        base["temperature"] += np.random.uniform(-0.2, 0.2)
        base["humidity"] += np.random.uniform(-0.5, 0.5)

    base["device_id"] = "SIMULADOR_WEB"
    lectura = LecturaSensor(**base)
    return await predict_sensor_data(lectura, request)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Enviar estado actual de bienvenida
        await websocket.send_json({
            "type": "INIT",
            "local_ip": LOCAL_IP,
            "port": PORT,
            "endpoint": f"http://{LOCAL_IP}:{PORT}/api/predict",
            "history": list(historial.historial_graficas)
        })
        while True:
            # Mantener conexión viva
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

# ====================================================================
# 6. TAREA EN SEGUNDO PLANO: ENLACE AUTOMÁTICO CON EL ESP32 AP (192.168.4.1)
# ====================================================================
import asyncio
import requests

async def polling_esp32_ap():
    """
    Si la PC se conecta al Wi-Fi 'ML' del ESP32, esta tarea lee los sensores
    desde http://192.168.4.1/api/sensors automaticamente y manda la prediccion.
    """
    await asyncio.sleep(2)
    while True:
        try:
            # Consultar en hilo separado para no bloquear el bucle de eventos
            def fetch_data():
                return requests.get("http://192.168.4.1/api/sensors", timeout=0.8)

            resp = await asyncio.to_thread(fetch_data)
            if resp.status_code == 200:
                data = resp.json()
                lectura = LecturaSensor(**data)
                datos_dict = lectura.model_dump()
                df_features, features_calc = historial.procesar_lectura(datos_dict)

                prediccion = int(modelo.predict(df_features)[0])
                
                t0 = time.perf_counter()
                probabilidad = 1.0 if prediccion == 1 else 0.0
                if hasattr(modelo, "predict_proba"):
                    try:
                        probabilidad = float(modelo.predict_proba(df_features)[0][1])
                    except Exception:
                        pass
                latencia_ms = (time.perf_counter() - t0) * 1000.0

                hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                raw_aire_val = data.get("raw_aire", data.get("rawAire"))

                registrar_inferencia_csv(
                    timestamp=hora_actual,
                    ip_origen="192.168.4.1",
                    temperature=float(data.get("temperature", 0.0)),
                    humidity=float(data.get("humidity", 0.0)),
                    raw_aire=float(raw_aire_val) if raw_aire_val is not None else None,
                    tvoc=float(data.get("tvoc", 0.0)),
                    pm2_5=float(data.get("pm2_5", data.get("pm25", 0.0))),
                    eco2=float(data.get("eco2", 0.0)),
                    tvoc_mean_5=float(features_calc.get("tvoc_mean_5", 0.0)),
                    prediccion=prediccion,
                    probabilidad=probabilidad,
                    latencia_ms=latencia_ms
                )

                resultado = {
                    "status": "ok",
                    "fire_alarm": prediccion,
                    "alarm_active": bool(prediccion == 1),
                    "probability_fire": round(probabilidad, 4),
                    "action": "TRIGGER_BUZZER_AND_LED" if prediccion == 1 else "NORMAL_MONITORING",
                    "message": "PELIGRO: Humo o combustión detectada!" if prediccion == 1 else "Aire limpio / Condiciones normales",
                    "timestamp": hora_actual,
                    "origin_ip": "192.168.4.1",
                    "device_id": data.get("device_id", "ESP32_AP_ML"),
                    "readings": datos_dict
                }

                historial.historial_graficas.append(resultado)
                await manager.broadcast({"type": "NEW_READING", "payload": resultado})

                # Notificar al ESP32 para activar o silenciar buzzer/LEDs
                def send_alarm():
                    requests.get(f"http://192.168.4.1/set_alarm?fire={prediccion}", timeout=0.6)
                await asyncio.to_thread(send_alarm)

        except Exception:
            pass
        await asyncio.sleep(1.2)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(polling_esp32_ap())

# ====================================================================
# 7. RUTA RAÍZ - SERVIR EL DASHBOARD HTML
# ====================================================================
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = BASE_DIR / "templates" / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Error: Plantilla index.html no encontrada</h1>"

if __name__ == "__main__":
    print("=" * 70)
    print("  MONITOR DE CALIDAD DE AIRE & ALARMA IOT - UPAO ML")
    print("=" * 70)
    print(f"  - Servidor Web:      http://localhost:{PORT}")
    print(f"  - Dashboard en Red:  http://{LOCAL_IP}:{PORT}")
    print(f"  - Endpoint ESP32:    http://{LOCAL_IP}:{PORT}/api/predict")
    print("=" * 70)
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
