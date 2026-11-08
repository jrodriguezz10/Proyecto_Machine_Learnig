# Monitor de Calidad del Aire y Detección Temprana de Incendios con Inferencia IoT

Sistema integral de **Machine Learning (ML)** e **Internet de las Cosas (IoT)** para la detección temprana y clasificación binaria en tiempo real de eventos de combustión e incendios (`Fire Alarm = 1`) ante la presencia de humo, gases tóxicos y compuestos orgánicos volátiles.

Desarrollado y validado en el marco del curso de **Machine Learning (Ciclo 8)** — **Universidad Privada Antenor Orrego (UPAO)**.

---

## 📑 Tabla de Contenidos

1. [Descripción General](#-descripción-general)
2. [Arquitectura del Sistema y Principio de Funcionamiento](#-arquitectura-del-sistema-y-principio-de-funcionamiento)
3. [Componentes de Hardware y Materiales](#-componentes-de-hardware-y-materiales)
4. [Diagrama y Tabla Maestra de Conexiones](#-diagrama-y-tabla-maestra-de-conexiones)
5. [Pipeline Completo de Machine Learning](#-pipeline-completo-de-machine-learning)
6. [Resultados y Comparativa de Modelos](#-resultados-y-comparativa-de-modelos)
7. [Interpretabilidad y Validación del Modelo (08_interpretabilidad.py)](#-interpretabilidad-y-validación-del-modelo-08_interpretabilidadpy)
8. [Monitoreo Continuo, Salud de Sensores y Data Drift (09_monitoreo.py)](#-monitoreo-continuo-salud-de-sensores-y-data-drift-09_monitoreopy)
9. [Estructura del Repositorio](#-estructura-del-repositorio)
10. [Guía de Instalación y Ejecución Paso a Paso](#-guía-de-instalación-y-ejecución-paso-a-paso)
11. [Protocolo de Demostración en Vivo](#-protocolo-de-demostración-en-vivo)
12. [Autores](#-autores)

---

## 🎯 Descripción General

El objetivo del proyecto es prevenir conatos de incendio mediante la detección ultrarrápida de firmas químicas y ambientales antes de que se produzcan llamas visibles o incrementos térmicos descontrolados. 

A partir de 62,630 registros experimentales del *Smoke Detection Dataset IoT*, se construyó un pipeline estandarizado de Machine Learning capaz de inferir con **21 características (incluyendo ingeniería temporal de deltas y ventanas móviles)**. El modelo en producción está integrado a un nodo físico embebido basado en el microcontrolador **ESP32**, el cual envía telemetría vía Wi-Fi a un servidor **FastAPI**, despliega un dashboard web en tiempo real con WebSockets y ejecuta auditorías continuas de salud de sensores y *Data Drift* (deriva de datos).

---

## 🌐 Arquitectura del Sistema y Principio de Funcionamiento

El sistema opera como una solución *Edge-Fog-Cloud* en red local totalmente autónoma:

```
[ NODO SENSOR FÍSICO (ESP32) ]
  ├── Sensor de Calidad de Aire (MQ-2 / MP503) -> ADC GPIO 34
  ├── Sensor Ambiental DHT11                   -> GPIO 27 (1-Wire)
  └── Actuadores: LED Verde (GPIO 5), LED Rojo (GPIO 4), Buzzer (GPIO 18)
           │
           ▼  Wi-Fi Local (Access Point: "ML" | IP: 192.168.4.1)
           │  HTTP POST JSON cada ~500 ms (telemetría en vivo)
           │
[ SERVIDOR BACKEND (PC: 192.168.4.2:8000) ]
  ├── 1. Feature Engineering en Buffer FIFO (Deltas y Medias Móviles de 5 s)
  ├── 2. Vectorización a 21 características (FEATURE_NAMES)
  ├── 3. Inferencia ML con DecisionTreeClassifier (< 0.2 ms)
  ├── 4. Registro continuo en CSV (latencia_ms, IP, probabilidad, variables)
  └── 5. Streaming por WebSocket a clientes conectados
           │
           ├──────────────────────────────┬──────────────────────────────┐
           ▼                              ▼                              ▼
  [ RETROALIMENTACIÓN ACTUADORES ] [ DASHBOARD WEB OFFLINE ]   [ AUDITORÍA Y MONITOREO ]
  Respuesta HTTP JSON al ESP32:    Interfaz moderna Chart.js:  - GET /api/health
  - fire_alarm = 0: LED Verde      - 6 KPIs en vivo            - Detección de Drift (KS, PSI)
  - fire_alarm = 1: LED Rojo       - 2 Gráficas temporales     - Detección de congelamiento
                    + Buzzer PWM   - Tabla de últimas lecturas - Análisis de latencia
```

### Principio de Inferencia y Reconstrucción Sensorial
En el firmware [`esp32_monitor.ino`](dashboard_iot/esp32_firmware/esp32_monitor.ino), la señal analógica cruda del sensor de gas (`rawAire` de 0 a 4095 en ADC de 12 bits) se mapea con una función de transferencia lineal con saturación:
$$\text{factorHumo} = \text{clip}\left(\frac{\text{rawAire} - 1500}{1600},\, 0,\, 1\right)$$

- **En reposo / aire limpio (`rawAire` < 1500):** $\text{factorHumo} = 0$, estableciendo las líneas base físicas: $\text{TVOC} = 10\text{ ppb}$, $\text{eCO}_2 = 400\text{ ppm}$, $\text{PM}_{2.5} = 0.05\,\mu\text{g/m}^3$, $\text{PM}_{1.0} = 0.04\,\mu\text{g/m}^3$.
- **Ante presencia de humo/combustión (`rawAire` > 1500):** Los gases crecen proporcionalmente hasta saturar a 3100 ADC ($\text{TVOC} \to 1410\text{ ppb}$, $\text{eCO}_2 \to 850\text{ ppm}$, $\text{PM}_{2.5} \to 4.25\,\mu\text{g/m}^3$).
- El backend recibe estos datos brutos y calcula dinámicamente las tasas de variación temporal (`tvoc_delta`, `temp_delta`, etc.) y las medias móviles (`tvoc_mean_5`, `pm25_mean_5`, `eco2_mean_5`) para alimentar las 21 variables del modelo sin discrepancias de escala.

---

## 🛠️ Componentes de Hardware y Materiales

| Componente | Modelo / Especificación | Cantidad | Función Técnica en el Prototipo |
| :--- | :--- | :---: | :--- |
| **Microcontrolador** | ESP32 DevKit V1 (30 pines, Dual-Core 240 MHz) | 1 | Genera el Access Point Wi-Fi, digitaliza señales con ADC de 12 bits y acciona actuadores. |
| **Sensor de Calidad de Aire** | Módulo Analógico MQ-2 / MP503 (Micro-Gas) | 1 | Detección de gases inflamables, humo, alcohol, GLP y partículas volátiles (pin AOUT). |
| **Sensor Ambiental** | DHT11 (Digital con pull-up) | 1 | Monitoreo de Temperatura (0 a 50 °C) y Humedad Relativa (20 a 90 %) mediante 1-Wire. |
| **Alarma Acústica** | Buzzer (Zumbador pasivo/activo 3.3V-5V) | 1 | Alarma sonora modulada a 2400 Hz vía PWM (`tone()`) activada ante `Fire Alarm = 1`. |
| **Indicador Normal** | Diodo LED Verde (5 mm difuso) | 1 | Indicador visual de estado seguro / aire limpio (`Fire Alarm = 0`). |
| **Indicador Alarma** | Diodo LED Rojo (5 mm difuso) | 1 | Indicador visual de emergencia crítica de incendio (`Fire Alarm = 1`). |
| **Resistencias** | 220 Ω (1/4 W, tolerancia 5%) | 2 | Limitadoras de corriente para proteger los puertos GPIO del ESP32 conectados a los LEDs. |
| **Placa de Pruebas** | Protoboard de 400 puntos | 1 | Montaje y distribución del bus común de alimentación (3.3V) y tierra (GND). |
| **Cableado** | Jumpers Dupont Macho-Macho / Macho-Hembra | 10 | Interconexión modular rápida sin soldadura. |
| **Cable de Datos** | Cable USB a Micro-USB con líneas de datos | 1 | Programación del firmware y alimentación de 5V al ESP32 desde la PC. |

---

## 🔌 Diagrama y Tabla Maestra de Conexiones

### Tabla Maestra de Pines (ESP32 DevKit V1)

> **Regla de oro:** Todos los dispositivos comparten el mismo riel negativo común (`GND`) del ESP32 para asegurar referencia de tierra unificada.

| Dispositivo | Pin Físico del Módulo | Pin GPIO ESP32 | Tipo de Señal | Nivel Lógico / Voltaje | Función |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sensor MQ-2 / Aire** | **VCC** | **3V3** (o VIN 5V) | Alimentación | 3.3 V / 5.0 V | Alimentación del circuito calentador y puente divisor |
| | **GND** | **GND** | Tierra | 0.0 V | Retorno a tierra común |
| | **AOUT** | **GPIO 34 (D34)** | Entrada Analógica | 0.0 V - 3.3 V (ADC) | Lectura analógica continua de concentración de gas |
| | **DOUT** | *Sin conectar* | Digital | - | Comparador digital integrado (no requerido) |
| **Sensor DHT11** | **VCC / (+)** | **3V3** | Alimentación | 3.3 V | Alimentación lógica del sensor |
| | **GND / (-)** | **GND** | Tierra | 0.0 V | Retorno a tierra común |
| | **DATA / OUT / S**| **GPIO 27 (D27)** | Entrada Digital | 3.3 V (1-Wire) | Comunicación serial digital de temperatura y humedad |
| **LED Verde (Seguro)**| **Ánodo (+, pata larga)**| **GPIO 5 (D5)** | Salida Digital | 3.3 V (High/Low) | Se ilumina en serie con resistor de 220 Ω en estado normal |
| | **Cátodo (-, pata corta)**| **GND** | Tierra | 0.0 V | Conexión al riel de tierra común |
| **LED Rojo (Alarma)** | **Ánodo (+, pata larga)**| **GPIO 4 (D4)** | Salida Digital | 3.3 V (High/Low) | Se ilumina en serie con resistor de 220 Ω en caso de incendio |
| | **Cátodo (-, pata corta)**| **GND** | Tierra | 0.0 V | Conexión al riel de tierra común |
| **Buzzer** | **Positivo (+)** | **GPIO 18 (D18)** | Salida PWM | 3.3 V (2400 Hz) | Genera onda cuadrada acústica intermitente durante alarma |
| | **Negativo (-)** | **GND** | Tierra | 0.0 V | Retorno a tierra común |

### Esquema Eléctrico ASCII

```
                             ESP32 DEVKIT V1
                         +--------------------+
    (Riel 3.3V) <--------| 3V3            VIN |<-- (Alimentación USB 5V)
    (Riel GND)  <--------| GND            GND |--------> (Riel GND Común)
                         |                    |
   (AOUT MQ-2) --------->| GPIO 34 (D34)      |
   (DATA DHT11) -------->| GPIO 27 (D27)      |
                         |                    |
                         | GPIO 18 (D18) ---->|--------> (+) Buzzer Acústico
                         | GPIO 4  (D4)  ---->|--[220Ω]-> (+) LED Rojo (Alarma)
                         | GPIO 5  (D5)  ---->|--[220Ω]-> (+) LED Verde (Normal)
                         +--------------------+
```

---

## 🔬 Pipeline Completo de Machine Learning

El desarrollo experimental se organizó en 9 fases secuenciales y reproducibles:

```
01_limpieza.py ──> 02_feature_engineering.py ──> 03_normalizacion.py ──> 04_particion.py
                                                                                │
                                                                                ▼
07_graficos.py <── 06_evaluacion.py <── 05_entrenamiento.py (Stratified K-Fold)
      │
      ├─> 08_interpretabilidad.py (MDI, Permutación, Reglas del Árbol, Validación por Sesión)
      └─> 09_monitoreo.py        (Auditoría en Vivo, Salud Sensores, Deriva de Datos KS/PSI)
```

1. **`01_limpieza.py`:** Inspección de valores nulos, registros duplicados, tipos de datos y descarte de identificadores no predictivos (`UTC`).
2. **`02_feature_engineering.py`:** Reconstrucción temporal de sesiones continuas mediante saltos del contador (`CNT`). Cálculo del ratio de partículas `pm_ratio = PM2.5 / (PM1.0 + 1e-6)`, diferencias de primer orden (`pm25_delta`, `tvoc_delta`, `eco2_delta`, `temp_delta`, `humidity_delta`) y medias móviles en ventanas de 5 observaciones (`pm25_mean_5`, `tvoc_mean_5`, `eco2_mean_5`).
3. **`03_normalizacion.py`:** Generación del vector final de **21 características** y exportación de matrices con y sin escalamiento (`StandardScaler`).
4. **`04_particion.py`:** División estratificada 70% Entrenamiento, 15% Validación y 15% Prueba para preservar la proporción de clases de alarma.
5. **`05_entrenamiento.py`:** Entrenamiento y sintonización de hiperparámetros de cuatro algoritmos bajo validación cruzada (`StratifiedKFold(n_splits=5)`).
6. **`06_evaluacion.py`:** Evaluación ciega sobre `X_test.csv` reportando matrices de confusión, F1-Macro, Precision, Recall y ROC-AUC.
7. **`07_graficos_evaluacion.py`:** Generación de curvas ROC multimodelo y visualizaciones de matrices de confusión en `resultados_evaluacion/graficos/`.

---

## 📊 Resultados y Comparativa de Modelos

| Algoritmo | Accuracy (Test) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC | Latencia Inferencia | Veredicto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Línea Base)** | 52.34% | 0.4168 | 0.5000 | 0.4168 | 0.5000 | < 0.05 ms | Descartado |
| **Support Vector Machine (RBF)** | 99.45% | 0.9910 | 0.9956 | 0.9945 | 0.9982 | ~15.20 ms | Descartado (Alta latencia) |
| **Random Forest (300 árboles)** | 99.98% | 0.9998 | 0.9996 | 0.9999 | 1.0000 | ~8.40 ms | Excelente (Candidato) |
| **Decision Tree (Entropy, Depth=5)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **< 0.15 ms** | **SELECCIONADO (Producción)** |

### Justificación de Selección:
- **Cero Falsos Negativos:** En sistemas de seguridad de vida, no detectar un incendio real es crítico. El Árbol de Decisión obtuvo **Recall = 1.0000** sin comprometer precisión.
- **Ultra Baja Latencia:** Con `< 0.15 ms` por predicción, permite una tasa de actualización superior a 6,000 inferencias por segundo en CPU convencional.
- **Transparencia Total (Caja Blanca):** Puede exportarse directamente a sentencias `if-else` en microcontroladores de memoria ultralimitada sin necesidad de intérpretes pesados.

---

## 🔍 Interpretabilidad y Validación del Modelo (08_interpretabilidad.py)

El script [`08_interpretabilidad.py`](08_interpretabilidad.py) audita la toma de decisiones del modelo y su robustez ante transferibilidad de escenarios:

1. **Importancia de Características (MDI y Permutación):**
   - El promedio móvil `tvoc_mean_5` y la presión barométrica `Pressure[hPa]` concentran más del 65% del peso predictivo acumulado en MDI.
   - En la prueba de **Permutación en Test (`scoring='f1_macro'`)**, alterar `Pressure[hPa]` o `tvoc_mean_5` degrada el desempeño en más de 0.28 puntos de F1-Score, confirmando que el modelo se apoya en firmas físicas reales de combustión y despresurización en recintos cerrados.
   - Gráficos generados en [`resultados_evaluacion/interpretabilidad/`](resultados_evaluacion/interpretabilidad/):
     - `mdi_arbol_decision.png`
     - `mdi_random_forest.png`
     - `permutacion_comparativo_top10.png`

2. **Reglas Explícitas del Árbol de Decisión (`export_text`):**
   - Profundidad máxima real: **5 niveles**, 11 nodos hoja.
   - La regla raíz comprueba: `Pressure[hPa] <= 937.585` y `tvoc_mean_5 > 42.600 ppb`, discriminando inmediatamente el aire ambiente normal de una atmósfera con humo.

3. **Validación Leave-One-Group-Out (LOGO) por Sesiones:**
   - Se evaluó el modelo entrenándolo en $N-1$ sesiones y testeándolo en la sesión restante (reconstruida mediante reinicios de `CNT`).
   - Con las 21 variables originales, el F1-Macro promedio entre sesiones independientes fue de **95.84%**.
   - Al remover `Pressure[hPa]` y evaluar solo el subconjunto medible directamente por el prototipo (`Temperature`, `Humidity`, `TVOC`, `tvoc_delta`, `tvoc_mean_5`, `temp_delta`, `humidity_delta`), el modelo mantiene un F1-Score robusto superior a **91.4%**, confirmando su viabilidad práctica en entornos sin barómetro industrial.

4. **Simulación de Curva de Activación:**
   - Se simuló el barrido de la señal del sensor `rawAire` de 1000 a 3100 ADC:
   - **Punto de activación de alarma:** Se dispara exactamente cuando `rawAire >= 1575` (correspondiente a `TVOC >= 75.6 ppb`), reproduciendo fielmente el umbral físico programado en el prototipo.

---

## 📈 Monitoreo Continuo, Salud de Sensores y Data Drift (09_monitoreo.py)

Para la puesta en producción, [`dashboard_iot/app.py`](dashboard_iot/app.py) incorpora telemetría persistente en [`monitoreo/registro_inferencias.csv`](monitoreo/registro_inferencias.csv) y un endpoint de salud `GET /api/health`. El módulo [`09_monitoreo.py`](09_monitoreo.py) ejecuta auditorías estadísticas continuas:

### 1. Métricas de Operación de la Sesión
- **Inferencias reales analizadas:** 307 lecturas continuas del ESP32.
- **Frecuencia media de inferencia:** **1.75 lecturas/segundo** (muestreo uniforme cada ~570 ms).
- **Latencia de predicción:** Media de **0.134 ms**, P95 de **0.198 ms**, Máxima de **0.612 ms** (100% de inferencias en tiempo real estricto).
- **Episodios de Alarma:** Detección de **1 episodio continuo de combustión** (inicio `21:26:19`, fin `21:26:41`, duración `22.0 s`, con pico de `raw_aire = 2395` y `TVOC = 793.10 ppb`).

### 2. Salud de Sensores y Filtro Inteligente de Reposo
- **Temperatura y Humedad:** 0 muestras fuera de rango operativo (0–50 °C y 20–95%).
- **Línea base en reposo:** `tvoc` (10 ppb), `eco2` (400 ppm) y `pm2_5` (0.05) permanecieron 271 lecturas continuas en su valor mínimo mientras `raw_aire < 1500`, clasificándose correctamente como `"Línea base (sensor en reposo)"` en lugar de una falsa alarma de sensor congelado.
- **Resolución DHT11:** La humedad relativa reportó 137 lecturas constantes al 62%, clasificándose como `"Estable (resolución DHT11 1 %)"` debido a la cuantización entera del sensor mientras la temperatura y el gas continuaban variando normalmente.
- **Protección contra fallos:** Se marca `ALERTA: CONGELADO` únicamente si el canal analógico `raw_aire` permanece estancado por más de 30 lecturas consecutivas (lo cual indicaría microcontrolador colgado o cable de señal desconectado).

### 3. Detección de Data Drift (Deriva de Datos)
Se comparó la distribución en vivo del prototipo respecto a la distribución del dataset de entrenamiento mediante la prueba de **Kolmogorov-Smirnov (KS-test)** y el **Índice de Estabilidad Poblacional (PSI)** con 10 deciles:

| Variable | Media Entrenamiento | Media en Vivo (Prototipo) | Estadístico KS | p-valor | PSI (10 Bins) | Diagnóstico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temperature** | 15.97 °C | 29.88 °C | 0.9509 | < 1e-10 | 8.270 | Drift Ambiental (Clima local vs. laboratorio original) |
| **Humidity** | 48.54 % | 61.80 % | 0.9792 | < 1e-10 | 8.277 | Drift Ambiental (Humedad relativa local) |
| **TVOC** | 1942.06 ppb | 83.18 ppb | 0.8173 | < 1e-10 | 6.812 | Línea base limpia (sin fuego activo la mayor parte del tiempo) |
| **eCO2** | 670.02 ppm | 423.52 ppm | 0.3730 | < 1e-10 | 1.128 | En reposo normal (~400 ppm de fondo atmosférico) |

*Gráficos generados en [`monitoreo/`](monitoreo/):*
- `probabilidad_prediccion_tiempo.png`
- `latencia_tiempo.png`
- `histograma_comparativo_temperatura_tvoc.png`
- `salida_09.txt`

---

## 📂 Estructura del Repositorio

```text
Proyecto_Machine_Learning/
│
├── 01_limpieza.py                     # Pipeline de limpieza y descarte de metadatos
├── 02_feature_engineering.py          # Cálculo de deltas, ratios y ventanas móviles (21 features)
├── 03_normalizacion.py                # Definición de FEATURE_NAMES y normalización StandardScaler
├── 04_particion.py                    # Partición estratificada Train / Validation / Test
├── 05_entrenamiento.py                # Entrenamiento multimodelo bajo Stratified K-Fold
├── 06_evaluacion.py                   # Evaluación exhaustiva en conjunto ciego de prueba
├── 07_graficos_evaluacion.py          # Curvas ROC y matrices de confusión comparativas
├── 08_interpretabilidad.py            # MDI, Permutación, Reglas del Árbol, LOGO y simulación
├── 09_monitoreo.py                    # Auditoría de inferencias, salud de sensores y Data Drift
│
├── dashboard_iot/                     # Servidor Web e Inferencia Edge/Fog
│   ├── app.py                         # API FastAPI (POST /api/predict, GET /api/health, WebSocket)
│   ├── app_backup.py                  # Copia de seguridad del servidor
│   ├── iniciar_dashboard.bat          # Script de arranque automático en Windows
│   ├── templates/
│   │   └── index.html                 # Interfaz visual responsive (KPIs, gráficas en vivo, tabla)
│   ├── static/
│   │   └── chart.umd.min.js           # Librería Chart.js para visualización offline
│   └── esp32_firmware/
│       └── esp32_monitor.ino          # Firmware C++ para microcontrolador ESP32
│
├── data_procesada/                    # Datasets limpios y particiones (.csv)
│   ├── smoke_detection_features.csv   # Dataset maestro con las 21 características
│   └── particiones/                   # X_train, X_val, X_test, y_train, etc.
│
├── modelos/                           # Serialización binaria de modelos (.joblib)
│   ├── decision_tree.joblib
│   ├── modelo_mejor_validacion.joblib # Modelo en producción cargado por FastAPI
│   ├── random_forest.joblib
│   └── svm.joblib
│
├── monitoreo/                         # Datos en vivo de inferencia y reportes de drift
│   ├── registro_inferencias.csv       # Telemetría real con 307 inferencias del ESP32
│   ├── registro_SIMULADO_no_usar.csv  # Telemetría histórica simulada aislada
│   ├── probabilidad_prediccion_tiempo.png
│   ├── latencia_tiempo.png
│   ├── histograma_comparativo_temperatura_tvoc.png
│   └── salida_09.txt                  # Reporte consolidado de consola
│
├── resultados_evaluacion/             # Reportes gráficos de evaluación
│   ├── graficos/                      # Matrices de confusión y curvas ROC
│   └── interpretabilidad/             # Gráficos de MDI y Permutation Importance
│
├── requirements.txt                   # Dependencias Python (FastAPI, scikit-learn, etc.)
└── README.md                          # Manual técnico integral del proyecto
```

---

## 🚀 Guía de Instalación y Ejecución Paso a Paso

### 1. Preparación del Entorno
* **Python 3.10 o superior**.
* Clona o descarga el repositorio y accede a la carpeta raíz:
```bash
git clone https://github.com/jrodriguezz10/Proyecto_Machine_Learnig.git
cd Proyecto_Machine_Learnig
```
* Instala las dependencias necesarias:
```bash
pip install -r requirements.txt
```

### 2. Flujo de Entrenamiento y Evaluación Offline (Opcional si se desea reentrenar)
Los modelos ya se encuentran compilados en `modelos/`. Si deseas reproducir el pipeline completo desde cero:
```bash
python 01_limpieza.py
python 02_feature_engineering.py
python 03_normalizacion.py
python 04_particion.py
python 05_entrenamiento.py
python 06_evaluacion.py
python 07_graficos_evaluacion.py
python 08_interpretabilidad.py
```

### 3. Carga del Firmware al ESP32
1. Abre **Arduino IDE** (versión 2.x recomendada).
2. Instala el soporte de placas ESP32 desde el gestor de tarjetas (`esp32 by Espressif Systems`).
3. Instala la librería **DHT sensor library by Adafruit** desde el gestor de librerías.
4. Abre [`dashboard_iot/esp32_firmware/esp32_monitor.ino`](dashboard_iot/esp32_firmware/esp32_monitor.ino).
5. Conecta el ESP32 por USB, selecciona la placa **ESP32 Dev Module** y el puerto COM asignado.
6. Presiona **Subir (Upload)**.
7. Al finalizar, el ESP32 creará la red Wi-Fi:
   - **SSID:** `ML`
   - **Password:** `12345678`

### 4. Lanzamiento del Servidor y Dashboard IoT
1. Conéctate con tu laptop/PC a la red Wi-Fi emitida por el ESP32: **`ML`**.
2. Tu computadora recibirá la dirección IP `192.168.4.2` (o similar dentro del rango 192.168.4.x).
3. Haz doble clic en [`dashboard_iot/iniciar_dashboard.bat`](dashboard_iot/iniciar_dashboard.bat) (o ejecuta `python dashboard_iot/app.py`).
4. Abre tu navegador web en:  
   👉 **`http://localhost:8000`** (o `http://192.168.4.2:8000`).
5. El dashboard comenzará a actualizar inmediatamente las gráficas en vivo con cada transmisión del ESP32.

### 5. Ejecución del Monitoreo y Auditoría Continua
Para evaluar la telemetría acumulada en vivo, verificar el estado de los sensores y calcular el *Data Drift*:
```bash
python 09_monitoreo.py
```

---

## 🧪 Protocolo de Demostración en Vivo

Para validar experimentalmente la velocidad de detección y confiabilidad del prototipo:

### Prueba de Combustión / Vapor de Solvente (Segura para Interiores):
1. Humedece un hisopo o la yema de tu dedo con **alcohol medicinal (70% o 96%)** o líquido volátil.
2. Acércalo a 1 o 2 cm de la malla metálica del sensor de calidad del aire (MQ-2 / MP503).
3. **Respuesta Inmediata (< 1 segundo):**
   - El sensor detecta los compuestos volátiles elevando `rawAire` por encima de 1575 ADC.
   - El backend computa los deltas y predice **`Fire Alarm = 1`** con probabilidad del 100%.
   - El ESP32 apaga el **LED Verde**, enciende de inmediato el **LED Rojo** y activa el **Buzzer** acústico.
   - En el Dashboard Web, el banner de estado cambia a rojo de emergencia y las gráficas Chart.js reflejan el pico pronunciado.

### Retorno y Recuperación a Estado Seguro:
1. Retira la fuente de vapor y sopla suavemente sobre el sensor durante 3 a 5 segundos.
2. Al ventilarse, `rawAire` desciende por debajo de 1500 ADC y las medias móviles caen a la línea base.
3. El sistema apaga la alarma sonora, apaga el LED Rojo y **enciende nuevamente el LED Verde**, retornando a supervisión normal sin reinicio manual.

---

## 👥 Autores

* **Rodriguez Zavaleta Jheyson**
* **Silvestre Miguel Alex Paul**

*Escuela Profesional de Ingeniería de Computación y Sistemas — Universidad Privada Antenor Orrego (UPAO 2026)*
