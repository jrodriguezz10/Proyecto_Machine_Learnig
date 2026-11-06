# Monitor de calidad del aire y detección temprana de quemas o humos tóxicos

Sistema de clasificación binaria para la predicción de activación de alarmas (`Fire Alarm`) ante presencia de humo o gases tóxicos.
Implementado y validado a partir del Smoke Detection Dataset disponible en Kaggle.

## Resultados en Test

| Modelo | F1-Score |
|---|---|
| Dummy Classifier | 0.4168 |
| Decision Tree | 1.0000 |
| Random Forest | 0.9999 |
| SVM | 0.9945 |

## Estructura del Proyecto

- `01_limpieza.py` a `06_evaluacion.py`: Flujo completo de preparación, entrenamiento y evaluación.
- `dashboard_iot/`:
  - `app.py`: Servidor FastAPI con inferencia en tiempo real y WebSocket.
  - `templates/index.html`: Dashboard web profesional con tema claro y gráficas en vivo.
  - `static/chart.umd.min.js`: Librería Chart.js servida localmente para funcionamiento 100% offline.
  - `esp32_firmware/esp32_monitor.ino`: Firmware para microcontrolador ESP32 (modo Access Point).
  - `iniciar_dashboard.bat`: Acceso directo para ejecutar el servidor local.
- `data_procesada/`: Datasets limpios, transformados y particiones generadas.
- `modelos/`: Modelos entrenados y serializados en `.joblib`.
- `resultados_evaluacion/`: Métricas, matrices de confusión y gráficos comparativos.

## Sistema IoT y Dashboard en Tiempo Real

El sistema integra el mejor modelo (`DecisionTreeClassifier`) con un nodo físico ESP32:
1. **Microcontrolador:** ESP32 DevKit V1 configurado como Access Point (`SSID: ML`, `PASS: 12345678`).
2. **Sensores:** Sensor de calidad del aire analógico (AOUT a GPIO 34) y sensor ambiental DHT11 (GPIO 27).
3. **Actuadores:** LED Verde (GPIO 5), LED Rojo (GPIO 4) y Buzzer acústico (GPIO 18).
4. **Dashboard:** Accesible localmente en `http://localhost:8000` con streaming de datos vía WebSockets.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

Ejecutar los scripts en orden secuencial:
```bash
python 01_limpieza.py
# ... hasta
python 06_evaluacion.py
```

## Autores

Rodriguez Zavaleta Jheyson, Silvestre Miguel Alex Paul — UPAO 2026
