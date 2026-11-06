# Monitor de Calidad del Aire y Detección Temprana de Incendios e Inferencia IoT

Sistema integral de Machine Learning e Internet de las Cosas (IoT) para la clasificación binaria y predicción en tiempo real de activación de alarmas (`Fire Alarm = 1`) ante la presencia de combustión, humo o gases tóxicos.

Desarrollado y validado en el marco del curso de **Machine Learning (Ciclo 8)** — **Universidad Privada Antenor Orrego (UPAO)**.

---

## 📑 Tabla de Contenidos

1. [Descripción General](#-descripción-general)
2. [Resultados y Comparativa de Modelos ML](#-resultados-y-comparativa-de-modelos-ml)
3. [Componentes de Hardware Utilizados](#-componentes-de-hardware-utilizados)
4. [Diagrama y Tabla Maestra de Conexiones](#-diagrama-y-tabla-maestra-de-conexiones)
5. [Arquitectura del Sistema y Flujo de Inferencia](#-arquitectura-del-sistema-y-flujo-de-inferencia)
6. [Estructura del Proyecto](#-estructura-del-proyecto)
7. [Guía de Instalación y Ejecución](#-guía-de-instalación-y-ejecución)
8. [Protocolo de Demostración en Vivo](#-protocolo-de-demostración-en-vivo)
9. [Autores](#-autores)

---

## 🎯 Descripción General

El proyecto aborda la detección temprana de eventos de combustión utilizando parámetros ambientales y químicos del aire. A partir del dataset benchmark *Smoke Detection Dataset IoT*, se construyó un pipeline completo de Machine Learning (limpieza, ingeniería de características temporales, partición estratificada, entrenamiento y evaluación) y se desplegó un prototipo físico IoT autónomo conectado a un servidor de inferencia y dashboard en tiempo real.

---

## 📊 Resultados y Comparativa de Modelos ML

Se entrenaron y validaron cuatro algoritmos de clasificación supervisada mediante **Validación Cruzada Estratificada (Stratified K-Fold)**:

| Modelo | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Test) | Tiempo de Inferencia | Estado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | 52.34% | 0.4168 | 0.5000 | 0.4168 | < 0.1 ms | Descartado |
| **Support Vector Machine (SVM)** | 99.45% | 0.9910 | 0.9956 | 0.9945 | ~15.2 ms | Descartado |
| **Random Forest Classifier** | 99.98% | 0.9998 | 0.9996 | 0.9999 | ~8.4 ms | Candidato |
| **Decision Tree Classifier (Elegido)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **< 0.2 ms** | **En Producción** |

### ¿Por qué se seleccionó el Árbol de Decisión?
1. **Rendimiento Óptimo:** Alcanzó un **F1-Score perfecto (1.0000)** en las particiones de validación y test.
2. **Latencia Sub-milisegundo:** Tiempo de inferencia inferior a **0.2 ms**, ideal para sistemas embebidos y seguridad en tiempo real.
3. **Interpretabilidad (White-Box):** Permite auditoría directa de las reglas de decisión, identificando que el promedio móvil de gases volátiles (`tvoc_mean_5`) y la concentración de partículas (`NC0.5` / `PM2.5`) son los predictores más relevantes de un conato de incendio.
4. **Eficiencia de Almacenamiento:** El modelo serializado ocupa apenas **4.4 KB** (`modelo_mejor_validacion.joblib`).

---

## 🛠️ Componentes de Hardware Utilizados

Para garantizar la viabilidad económica y escalabilidad en un despliegue real, se implementó un nodo físico de bajo costo:

| Componente | Modelo / Especificación | Función en el Proyecto |
| :--- | :--- | :--- |
| **Microcontrolador** | ESP32 DevKit V1 (30 pines) | Emisor Wi-Fi (Access Point), lectura de sensores ADC/Digital y control de actuadores. |
| **Sensor de Calidad de Aire** | Módulo Analógico MP503 / Micro-Gas | Detección de gases volátiles, humo, solventes y partículas de combustión (Salida AOUT). |
| **Sensor Ambiental** | DHT11 (Digital) | Medición de Temperatura (°C) y Humedad Relativa (%). |
| **Alarma Sonora** | Buzzer (Zumbador de 5V/3.3V) | Señal acústica de peligro (onda de 2.4 kHz con modulación `tone()`). |
| **Indicador Normal** | LED Verde (5mm) | Indicador visual de estado seguro / aire limpio (`Fire Alarm = 0`). |
| **Indicador Alarma** | LED Rojo (5mm) | Indicador visual de alarma crítica de fuego (`Fire Alarm = 1`). |
| **Resistencias** | 2x Resistencias de 220 Ω (o 330 Ω) | Protección limitadora de corriente para los LEDs. |
| **Conexiones** | Protoboard 400 pts + Cables Dupont | Interconexión física sin necesidad de soldadura. |

---

## 🔌 Diagrama y Tabla Maestra de Conexiones

### Tabla de Pines en ESP32 DevKit V1

> **Nota:** Todos los componentes comparten la misma línea de tierra (`GND`) del ESP32.

| Dispositivo | Pin Físico del Sensor | Pin en ESP32 | Tipo de Señal | Notas |
| :--- | :--- | :--- | :--- | :--- |
| **Sensor de Aire (MP503)** | **VCC** | **3V3** | Alimentación | 3.3V constante |
| | **GND** | **GND** | Tierra común | Riel negativo |
| | **AOUT** | **GPIO 34 (D34)** | Entrada Analógica | Rango ADC 12 bits (0 a 4095) |
| | **DOUT** | *(Sin conectar)* | Digital | No requerido |
| **Sensor DHT11** | **VCC / (+)** | **3V3** | Alimentación | 3.3V constante |
| | **GND / (-)** | **GND** | Tierra común | Riel negativo |
| | **DATA / OUT / S**| **GPIO 27 (D27)** | Entrada Digital | Protocolo 1-Wire DHT |
| **LED Verde (Normal)** | **Ánodo (+)** | **GPIO 5 (D5)** | Salida Digital | En serie con resistencia 220 Ω |
| | **Cátodo (-)** | **GND** | Tierra común | Pata corta |
| **LED Rojo (Alarma)** | **Ánodo (+)** | **GPIO 4 (D4)** | Salida Digital | En serie con resistencia 220 Ω |
| | **Cátodo (-)** | **GND** | Tierra común | Pata corta |
| **Buzzer** | **Positivo (+)** | **GPIO 18 (D18)** | Salida PWM/Tone | Frecuencia de 2400 Hz |
| | **Negativo (-)** | **GND** | Tierra común | Pata corta |

### Diagrama Esquemático

```
                          ESP32 DEVKIT V1
                       +-------------------+
  (Riel 3.3V) <--------| 3V3           VIN |
  (Riel GND)  <--------| GND           GND |--------> (Riel GND Común)
                       |                   |
 (AOUT Aire) --------->| GPIO 34 (D34)     |
 (DATA DHT11) -------->| GPIO 27 (D27)     |
                       |                   |
                       | GPIO 18 (D18) --->|--------> (+) Buzzer Activo/Pasivo
                       | GPIO 4  (D4)  --->|--[220Ω]-> (+) LED Rojo (Alarma)
                       | GPIO 5  (D5)  --->|--[220Ω]-> (+) LED Verde (Seguro)
                       +-------------------+
```

---

## 🌐 Arquitectura del Sistema y Flujo de Inferencia

El sistema opera mediante una red privada local independiente generada por el propio microcontrolador:

```
[ SENSORES FÍSICOS ]
  DHT11 (Temp/Hum)
  Sensor Aire (AOUT)
          │
          ▼ (Lectura cada 1.0 s)
[ NODO ESP32 (Access Point: "ML" | IP: 192.168.4.1) ]
          │
          ▼ HTTP POST JSON (Wi-Fi Local)
[ SERVIDOR BACKEND (PC: 192.168.4.2:8000) ]
  1. Cálculo de Deltas y Promedios Móviles (Feature Engineering)
  2. Inferencia con DecisionTreeClassifier (.joblib)
  3. Emisión WebSocket a los clientes web
          │
          ├──────────────────────────────┬──────────────────────────────┐
          ▼                              ▼                              ▼
  [ ACTUADORES EN ESP32 ]       [ DASHBOARD WEB OFFLINE ]      [ BASE DE DATOS/LOGS ]
  - Fire Alarm = 0: LED Verde   - Métricas en vivo (6 KPIs)    - Registro histórico
  - Fire Alarm = 1: LED Rojo    - 2 Gráficas Chart.js          - Auditoría en tabla
                    + Buzzer      (PM2.5/TVOC & Temp/Hum)
```

---

## 📂 Estructura del Proyecto

```text
Proyecto_Machine_Learning/
│
├── 01_limpieza.py                 # Pipeline de limpieza y validación de datos
├── 02_feature_engineering.py      # Creación de deltas, ratios y promedios móviles
├── 03_normalizacion.py            # Estandarización y preparación de datos
├── 04_particion.py                # División estratificada Train/Val/Test
├── 05_entrenamiento.py            # Entrenamiento y ajuste de hiperparámetros
├── 06_evaluacion.py               # Métricas, matrices de confusión y validación
├── 07_graficos_evaluacion.py      # Gráficos comparativos de rendimiento
│
├── dashboard_iot/                 # Sistema de Despliegue IoT y Visualización
│   ├── app.py                     # Servidor FastAPI con WebSocket e inferencia
│   ├── iniciar_dashboard.bat      # Script de lanzamiento con un solo clic
│   │
│   ├── templates/
│   │   └── index.html             # Dashboard web profesional (tema claro, 100% offline)
│   │
│   ├── static/
│   │   └── chart.umd.min.js       # Librería Chart.js empaquetada localmente
│   │
│   └── esp32_firmware/
│       └── esp32_monitor.ino      # Código en C++ para microcontrolador ESP32
│
├── data_procesada/                # Datasets limpios y particiones generadas
├── modelos/                       # Modelos entrenados exportados en .joblib
│   ├── decision_tree.joblib
│   ├── modelo_mejor_validacion.joblib
│   ├── random_forest.joblib
│   └── svm.joblib
│
├── resultados_evaluacion/         # Matrices de confusión y reportes de clasificación
├── requirements.txt               # Dependencias del entorno Python
└── README.md                      # Documentación técnica completa
```

---

## 🚀 Guía de Instalación y Ejecución

### 1. Requisitos Previos
* **Python 3.10 o superior** instalado en el sistema.
* **Arduino IDE 2.x** con el paquete de placas **ESP32 by Espressif Systems** instalado.
* Librería de Arduino: **DHT sensor library by Adafruit**.

### 2. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 3. Cargar el Firmware al ESP32
1. Conecta el ESP32 a tu computadora mediante cable micro-USB.
2. Abre en Arduino IDE el archivo [`dashboard_iot/esp32_firmware/esp32_monitor.ino`](dashboard_iot/esp32_firmware/esp32_monitor.ino).
3. Selecciona la placa **ESP32 Dev Module** y el puerto COM correspondiente.
4. Presiona **Subir (Upload)**.
5. Al terminar, el ESP32 comenzará a emitir la red Wi-Fi:
   * **Nombre de Red (SSID):** `ML`
   * **Contraseña:** `12345678`

### 4. Iniciar el Dashboard en la Computadora
1. Conéctate con tu computadora a la red Wi-Fi emitida por el ESP32: **`ML`**.
2. Ejecuta el archivo [`dashboard_iot/iniciar_dashboard.bat`](dashboard_iot/iniciar_dashboard.bat) (o ejecuta `python dashboard_iot/app.py`).
3. Abre tu navegador web en:  
   👉 **`http://localhost:8000`** (o `http://192.168.4.2:8000`).

---

## 🧪 Protocolo de Demostración en Vivo

Para validar la reacción del modelo ante situaciones reales de peligro:

### Prueba Limpia (Recomendada para aulas): Vapor de Alcohol
1. Humedece un hisopo o la yema de tu dedo con **alcohol en gel o líquido**.
2. Acércalo a 1 o 2 cm del orificio metálico del sensor de calidad del aire (sin tocarlo).
3. **Respuesta Inmediata (< 1 s):**
   * El sensor detecta los compuestos volátiles (TVOC y etanol).
   * El modelo cambia a `Fire Alarm = 1` (Probabilidad: 100%).
   * Se apaga el LED Verde, se enciende el **LED Rojo** y suena el **Buzzer**.
   * En el Dashboard, la barra superior cambia a rojo y la gráfica eleva el pico de concentración.

### Retorno al Estado Seguro
1. Retira el vapor y sopla suavemente sobre el sensor durante 3 segundos.
2. El promedio móvil se restablece por debajo del umbral de corte.
3. El sistema apaga la alarma automáticamente y **vuelve a encender el LED Verde**.

---

## 👥 Autores

* **Rodriguez Zavaleta Jheyson**
* **Silvestre Miguel Alex Paul**

*Escuela Profesional de Ingeniería de Computación y Sistemas — UPAO 2026*
