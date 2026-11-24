# Monitor de Calidad del Aire y Detección Temprana de Incendios con Inferencia IoT y Servicio Cognitivo IA

Sistema integral de **Machine Learning (ML)**, **Internet de las Cosas (IoT)** y **Servicio Cognitivo (Generative AI)** para la detección temprana y clasificación binaria en tiempo real de eventos de combustión e incendios (`Fire Alarm = 1`) ante la presencia de humo y gases volátiles.

Desarrollado y validado en el marco del curso de **Machine Learning (Ciclo 8)** — **Universidad Privada Antenor Orrego (UPAO)**.

---

## 📑 Tabla de Contenidos

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Arquitectura del Sistema](#-arquitectura-del-sistema)
3. [Hardware y Sensores Reales Utilizados](#-hardware-y-sensores-reales-utilizados)
4. [Dataset y Preprocesamiento](#-dataset-y-preprocesamiento)
5. [Modelos Entrenados y Resultados Comparativos en TEST](#-modelos-entrenados-y-resultados-comparativos-en-test)
6. [Análisis de Interpretabilidad y Validación por Sesión (LOGO)](#-análisis-de-interpretabilidad-y-validación-por-sesión-logo)
7. [Modelos Mitigados sin Presión Atmosférica](#-modelos-mitigados-sin-presión-atmosférica)
8. [Servicio Cognitivo Gemini (Generative AI)](#-servicio-cognitivo-gemini-generative-ai)
9. [Monitoreo en Vivo y Data Drift](#-monitoreo-en-vivo-y-data-drift)
10. [Seguridad de la API](#-seguridad-de-la-api)
11. [Estructura del Repositorio](#-estructura-del-repositorio)
12. [Instalación y Puesta en Marcha](#-instalación-y-puesta-en-marcha)
13. [Limitaciones Conocidas y Trabajo Futuro](#-limitaciones-conocidas-y-trabajo-futuro)
14. [Autores y Créditos](#-autores-y-créditos)

---

## 🎯 Descripción del Proyecto

El sistema detecta firmas químicas y ambientales características de conatos de incendio antes de que existan llamas visibles o aumentos térmicos extremos.

A partir de 62,630 observaciones del *Smoke Detection Dataset*, se construyó un pipeline estandarizado de Machine Learning con **21 características (incluyendo ingeniería temporal de deltas y medias móviles de 5 segundos)**. El sistema combina:
- **Inferencia de borde / fog**: Modelo `DecisionTreeClassifier` desplegado en un servidor local `FastAPI` con inferencia en submiliegundos (0.0008 ms por observación).
- **Nodo Embebido IoT**: Microcontrolador `ESP32` en modo Access Point Wi-Fi, leyendo un sensor de calidad de aire (MP503 / gas analógico) y un sensor ambiental DHT11, gobernando actuadores locales (LED Verde, LED Rojo, Buzzer).
- **Dashboard Web en Tiempo Real**: Visualización interactiva con WebSockets (HTML5 / TailwindCSS / Chart.js local).
- **Servicio Cognitivo Explicativo**: Integración con Google Gemini (`gemini-2.5-flash` / API oficial `google-genai`) para generar explicaciones ciudadanas en lenguaje natural ante transiciones de alarma.
- **Auditoría Continua y Monitoreo de Data Drift**: Evaluación estadística continua (prueba Kolmogorov-Smirnov y Population Stability Index - PSI) y auditoría de congelamiento de sensores.

---

## 🌐 Arquitectura del Sistema

```
[ NODO EMBEBIDO ESP32 ]
  ├── Sensor Calidad de Aire (MP503 / gas analógico) -> ADC GPIO 34
  ├── Sensor Ambiental DHT11                         -> GPIO 27 (1-Wire)
  ├── LED Verde (Normal)                             -> GPIO 5
  ├── LED Rojo (Alarma)                              -> GPIO 4
  └── Buzzer (Acústico)                              -> GPIO 18
            │
            │  Wi-Fi AP ("ML") • HTTP POST /api/predict (cada ~1.0 s)
            │  Cabecera obligatoria: X-API-Key (Autenticación SHA-256/Hex)
            ▼
[ SERVIDOR FASTAPI (192.168.4.2:8000 o localhost:8000) ]
  ├── 1. Autenticación y Validación de Esquema Pydantic
  ├── 2. Buffer Temporal FIFO (Cálculo de deltas y tvoc_mean_5, pm25_mean_5, eco2_mean_5)
  ├── 3. Inferencia ML (DecisionTreeClassifier en modelos/modelo_mejor_validacion.joblib)
  ├── 4. Registro CSV en tiempo real (monitoreo/registro_inferencias.csv)
  ├── 5. Disparador Cognitivo Gemini (llamada asíncrona ante cambio 0 -> 1)
  └── 6. Difusión WebSocket (/ws) hacia clientes conectados
            │
            ▼
[ DASHBOARD WEB (Navegador) ]
  ├── KPIs en vivo, evolución temporal Chart.js, estado de actuadores
  └── Panel interactivo de Explicación IA con Gemini
```

---

## 🔌 Hardware y Sensores Reales Utilizados

| Componente | Conexión ESP32 | Función en el Prototipo |
| :--- | :--- | :--- |
| **Microcontrolador ESP32** | 30 pines / NodeMCU | Núcleo de adquisición, emisión Wi-Fi AP y control de actuadores |
| **Sensor de Aire (MP503 / Gas Analógico)** | VCC a 3V3, GND a GND, AOUT a **GPIO 34** (ADC1) | Detección de gases reductores y humo por variación analógica |
| **Sensor DHT11** | VCC a 3V3, GND a GND, DATA a **GPIO 27** | Monitoreo de temperatura (0–50 °C) y humedad relativa (20–90 %) |
| **LED Verde** | **GPIO 5** (con resistencia limitadora de 220 Ω) | Indicador visual de estado seguro / aire limpio |
| **LED Rojo** | **GPIO 4** (con resistencia limitadora de 220 Ω) | Indicador visual de alarma de incendio |
| **Buzzer Activo/Pasivo** | **GPIO 18** (+) a pin, (-) a GND | Señalización sonora a 2400 Hz al detectar fuego |

---

## 📊 Dataset y Preprocesamiento

- **Dataset Base**: *Smoke Detection Dataset* (Kaggle / IoT Environmental Sensors), con 62,630 observaciones y balance natural:
  - Clase 0 (Normal / Sin Alarma): 17,873 muestras (28.54 %)
  - Clase 1 (Alarma de Fuego)    : 44,757 muestras (71.46 %)
- **Particionado**: Train (70%, 43,841 muestras), Validation (15%, 9,394 muestras), Test (15%, 9,395 muestras) con estratificación (`random_state=42`).
- **Ingeniería de Características (21 variables)**:
  1. Ambientales y Gases: `Temperature[C]`, `Humidity[%]`, `TVOC[ppb]`, `eCO2[ppm]`, `Raw H2`, `Raw Ethanol`, `Pressure[hPa]`.
  2. Material Particulado: `PM1.0`, `PM2.5`, `NC0.5`, `NC1.0`, `NC2.5`, `pm_ratio`.
  3. Derivadas Temporales: `pm25_delta`, `tvoc_delta`, `eco2_delta`, `temp_delta`, `humidity_delta`.
  4. Ventanas Móviles (5 muestras): `pm25_mean_5`, `tvoc_mean_5`, `eco2_mean_5`.

---

## 🏆 Modelos Entrenados y Resultados Comparativos en TEST

Los modelos fueron entrenados y evaluados en el conjunto de prueba independiente (`X_test`, 9,395 muestras).

### Métricas en Conjunto TEST (Partición Aleatoria Estratificada)

| Modelo | Accuracy | Precision Macro | Recall Macro | F1 Macro | ROC-AUC | Latencia Inferencia / Obs. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Línea Base)** | 0.7146 | 0.3573 | 0.5000 | 0.4168 | 0.5000 | — |
| **Árbol de Decisión (`DecisionTreeClassifier`)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0008 ms** |
| **Random Forest (`RandomForestClassifier`)** | 0.9999 | 0.9998 | 0.9999 | 0.9999 | 1.0000 | 0.0250 ms |
| **SVM (`SVC` RBF)** | 0.9955 | 0.9927 | 0.9964 | 0.9945 | 0.9998 | — |
| **Red Neuronal MLP (`MLPClassifier`)** | 0.9995 | 0.9993 | 0.9994 | 0.9993 | 0.9999 | 0.0012 ms |

> **Nota sobre el Árbol de Decisión Desplegado**:
> - Hiperparámetros óptimos: `criterion='entropy'`, `max_depth=5`, `class_weight='balanced'`.
> - Estructura real del árbol: **17 nodos hoja** y **profundidad real = 5**.
> - En conjunto Test: 0 falsos negativos y 0 falsos positivos (100% de métricas con partición aleatoria).

---

## 🔍 Análisis de Interpretabilidad y Validación por Sesión (LOGO)

### El Fenómeno de Falsa Perfección: Partición Aleatoria vs. Sesiones Reales

Aunque con partición aleatoria estratificada (`train_test_split`) los algoritmos alcanzan F1 ≈ 1.00, este resultado enmascara una fuga temporal (*data leakage*): las lecturas del dataset provienen de series temporales continuas agrupadas en experimentos (sesiones identificables por reinicios del contador `CNT`).

Al evaluar con **Leave-One-Group-Out (LOGO)** —entrenando en unas sesiones y evaluando en una sesión no vista—, el rendimiento promedio cae significativamente:

| Esquema de Evaluación | Árbol de Decisión (F1 Macro Prom.) | Random Forest (F1 Macro Prom.) | Red Neuronal MLP (F1 Macro Prom.) |
| :--- | :---: | :---: | :---: |
| **Partición Aleatoria (Test)** | **1.0000** | **0.9999** | **0.9993** |
| **LOGO (21 Variables)** | **0.5086** | **0.5486** | **0.3364** |
| **LOGO (20 Variables, Sin Presión)** | **0.2952** | **0.4184** | **0.3583** |
| **LOGO (7 Variables Prototipo)** | **0.2350** | **0.1999** | **0.3003** |

### El Artefacto de la Presión Atmosférica (`Pressure[hPa]`)

El análisis de importancia reveló que la variable con mayor peso predictivo es `Pressure[hPa]`:
- **Importancia MDI**:
  - En Árbol de Decisión: `Pressure[hPa]` representa el **48.59 %** (0.4859) de la importancia total, seguido de `tvoc_mean_5` (32.28 %) y `NC0.5` (15.93 %).
  - En Random Forest: `Pressure[hPa]` representa el **21.09 %** (0.2109), seguido de `TVOC[ppb]` (16.38 %) y `tvoc_mean_5` (15.38 %).
- **Importancia por Permutación (caída media en F1 Macro)**:
  - Árbol de Decisión: `Pressure[hPa]` = 0.2993, `tvoc_mean_5` = 0.2916.
  - Random Forest: `Pressure[hPa]` = 0.1483, `tvoc_mean_5` = 0.0364.
  - SVM: `Pressure[hPa]` = 0.1886, `Raw Ethanol` = 0.1363.

**Conclusión Científica**: La presión barométrica fue el discriminador más fuerte en el dataset debido a las condiciones de ventilación o altitud de cada sesión de prueba del laboratorio, no porque la presión sea un indicador químico del humo. Por ello, el modelo depende críticamente de la ventana móvil de compuestos orgánicos volátiles (`tvoc_mean_5`) para operar de forma químicamente fundada.

### Comportamiento del Prototipo Físico y Umbral de Activación

En la calibración y simulación de la señal del sensor analógico MP503 (`rawAire` evaluado de 1000 a 3100 en pasos de 25):
- **Valor mínimo de activación de alarma**: `rawAire = 1725` (tanto para Árbol de Decisión como para Random Forest).
- Parámetros asociados a dicho punto: `factorHumo = 0.1406`, `TVOC = 206.88 ppb`, `PM2.5 = 0.64 ug/m³`.
- **Ruta de decisión recorrida en el Árbol**:
  1. `Pressure[hPa] (939.5) > 938.143` (Rama derecha)
  2. `tvoc_mean_5 (206.88) > 186.300` (Rama derecha)
  3. `tvoc_mean_5 (206.88) > 204.700` (Rama derecha)
  4. Nodo Hoja 32: `Fire Alarm = 1` (Pesos entrenamiento: [Normal=0, Fuego=26,389]).

---

## 🛡️ Modelos Mitigados sin Presión Atmosférica

Para evitar la dependencia del artefacto barométrico, se entrenaron modelos descartando `Pressure[hPa]` (20 variables):

| Modelo Mitigado (20 variables) | F1 Macro (Test) | Accuracy (Test) | ROC-AUC | F1 Promedio LOGO | Recall Alarma LOGO |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Árbol de Decisión (Sin Presión)** | 0.9995 | 0.9996 | 0.9995 | 0.2952 | 0.5240 |
| **Random Forest (Sin Presión)** | **0.9999** | **0.9999** | **1.0000** | **0.4184** | 0.2111 |

**Nueva Jerarquía MDI en Modelos Mitigados**:
- En Árbol de Decisión: `tvoc_mean_5` (35.00 %), `Humidity[%]` (24.80 %), `Raw H2` (10.35 %), `Raw Ethanol` (7.93 %), `TVOC[ppb]` (7.41 %).
- En Random Forest: `tvoc_mean_5` (18.22 %), `TVOC[ppb]` (17.48 %), `Raw Ethanol` (15.88 %), `Humidity[%]` (13.71 %), `Raw H2` (8.48 %).

---

## 🤖 Servicio Cognitivo Gemini (Generative AI)

El sistema incorpora un servicio cognitivo con el SDK oficial `google-genai` que traduce las variables técnicas de calidad de aire en diagnósticos comprensibles para un ciudadano o brigadista:
- **Disparador Automático**: Se activa asíncronamente cuando el sistema transiciona de estado seguro (`prediccion = 0`) a alarma de incendio (`prediccion = 1`). El texto generado se envía en tiempo real al navegador mediante WebSocket (`explicacion_ia`).
- **Consulta Manual**: Mediante el endpoint `GET /api/explicar` o el botón "Explicar ahora" del dashboard.
- **Registro de Auditoría**: Cada consulta se registra en `monitoreo/registro_gemini.csv` con fecha, latencia en ms, éxito y respuesta.
- **Tolerancia a Fallos**: Si no existe conexión o se agota la cuota, el endpoint devuelve de forma segura: `"Explicación no disponible: servicio cognitivo sin conexión"` sin interrumpir la inferencia ni bloquear el servidor.

---

## 📈 Monitoreo en Vivo y Data Drift

La herramienta de auditoría continua (`09_monitoreo.py`) analiza las sesiones de inferencia en vivo frente al dataset histórico de entrenamiento:

### Métricas de Sesión Real de Validación (307 inferencias consecutivas)
- **Duración**: 2 min 55 s (175.0 s)
- **Frecuencia media de inferencia**: 1.75 lecturas/s
- **Eventos de alarma**: 33 lecturas (10.75 %)
- **Latencia de inferencia**: Media: **0.8133 ms**, Percentil 95 (P95): **1.0987 ms**, Máxima: **4.7481 ms**.
- **Huecos > 5 s sin lecturas**: 0 eventos.
- **Episodios de alarma continuos**: 1 episodio (duración: 22.0 s, 33 lecturas; TVOC máximo: 793.10 ppb, `raw_aire` máximo: 2395.0).
- **Salud de Sensores**:
  - Fuera de rango (temperatura 0–50 °C, humedad 20–95 %): 0 muestras.
  - Auditoría de congelamiento: `tvoc` (10 ppb), `eco2` (400 ppm) y `pm2_5` (0.05) presentaron 271 lecturas repetidas clasificadas correctamente como **Línea base (sensor en reposo)** dado que `raw_aire < 1500`.

### Evaluación de Data Drift (Deriva de Datos)
| Variable | Media Entrenamiento | Media en Vivo | Estadístico KS | p-valor KS | PSI (10 bins) | Diagnóstico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temperature[C]** | 15.97 °C | 29.88 °C | 0.9509 | 0.0000 | 8.2703 | **Drift** |
| **Humidity[%]** | 48.54 % | 61.80 % | 0.9792 | 0.0000 | 8.2767 | **Drift** |
| **TVOC[ppb]** | 1942.06 ppb | 83.18 ppb | 0.8173 | 3.09e-224 | 6.8120 | **Drift** |
| **eCO2[ppm]** | 670.02 ppm | 423.52 ppm | 0.3730 | 9.81e-39 | 1.1276 | **Drift** |

*(Criterio PSI: Estable < 0.10, Moderado 0.10–0.25, Drift > 0.25)*.

---

## 🔒 Seguridad de la API

El endpoint de recepción de telemetría física (`POST /api/predict`) se encuentra protegido contra inyecciones no autorizadas mediante cabecera HTTP:
- **Cabecera requerida**: `X-API-Key: <clave_hexadecimal_32_caracteres>`
- **Comportamiento**:
  - Sin clave o con clave incorrecta: Retorna **HTTP 401 Unauthorized**.
  - Con clave válida: Retorna **HTTP 200 OK** con la inferencia correspondiente.
- **Endpoints Públicos**: `GET /` (dashboard), `GET /api/health` (estado del sistema), `GET /ws` (conexión WebSocket).

---

## 📁 Estructura del Repositorio

```text
archive (5)/
├── 01_eda.py                                       # Análisis exploratorio inicial
├── 02_limpieza_preprocesamiento.py                  # Imputación y limpieza de outliers
├── 03_ingenieria_caracteristicas.py                # Deltas, medias móviles y ratios temporales
├── 04_particionamiento.py                          # Partición train/val/test estratificada
├── 05_entrenamiento.py                             # Búsqueda y entrenamiento de DT, RF y SVM
├── 06_evaluacion_test.py                           # Métricas independientes sobre X_test
├── 07_exportacion_modelo.py                        # Serialización del modelo desplegado
├── 08_interpretabilidad.py                         # MDI, permutación, reglas y LOGO (21 vars)
├── 09_monitoreo.py                                 # Auditoría continua, salud sensores y PSI
├── 10_red_neuronal.py                              # MLPClassifier, curvas de pérdida y LOGO
├── 11_modelo_mitigado.py                           # DT y RF sin Pressure[hPa] y LOGO
├── requirements.txt                                # Dependencias exactas de Python
├── .env.example                                    # Plantilla de variables de entorno
├── .gitignore                                      # Exclusión de secretos, .venv y temporales
├── data_procesada/
│   ├── smoke_detection_features.csv               # Dataset procesado con 21 variables
│   └── particiones/                               # Particiones CSV (train, val, test y scaled)
├── modelos/
│   ├── modelo_mejor_validacion.joblib             # Modelo en producción (DecisionTree)
│   ├── decision_tree.joblib                       # Árbol de decisión base
│   ├── random_forest.joblib                       # Random Forest base
│   ├── svm.joblib                                 # Support Vector Machine base
│   ├── mlp.joblib                                 # Red neuronal MLP
│   ├── decision_tree_sin_presion.joblib           # Árbol mitigado sin Pressure[hPa]
│   └── random_forest_sin_presion.joblib          # Random Forest mitigado sin Pressure[hPa]
├── monitoreo/
│   ├── registro_inferencias.csv                   # Registro histórico completo de telemetría
│   ├── registro_sesion_informe.csv                # Sesión reproducida de 307 inferencias
│   ├── registro_gemini.csv                        # Auditoría de llamadas a la IA generativa
│   ├── prueba_seguridad.txt                       # Evidencia de validación HTTP 401 / 200
│   ├── prueba_gemini.txt                          # Evidencia de respuesta del servicio cognitivo
│   ├── salida_09_reproducida.txt                  # Salida completa de consola de 09_monitoreo.py
│   └── *.png                                      # Gráficas de latencia, PSI e inferencia
├── resultados_evaluacion/
│   ├── metricas_test.csv                          # Métricas tabuladas sobre conjunto Test
│   ├── interpretabilidad/
│   │   ├── salida_08.txt                          # Salida completa de 08_interpretabilidad.py
│   │   └── *.png                                  # Gráficas MDI y permutación
│   ├── red_neuronal/
│   │   ├── salida_10.txt                          # Salida completa de 10_red_neuronal.py
│   │   ├── matriz_confusion_mlp.png               # Matriz de confusión en test
│   │   └── curva_perdida_mlp.png                  # Curvas de pérdida y validación
│   └── mitigacion/
│       └── salida_11.txt                          # Salida completa de 11_modelo_mitigado.py
└── dashboard_iot/
    ├── app.py                                     # Servidor FastAPI + WebSockets + Gemini
    ├── templates/
    │   └── index.html                             # Dashboard web con Tailwind y Chart.js
    └── esp32_firmware/
        └── esp32_monitor.ino                      # Firmware C++ para ESP32 (AP Wi-Fi + ADC + DHT)
```

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el Repositorio y Crear el Entorno Virtual
```bash
git clone <URL_DEL_REPOSITORIO>
cd "archive (5)"

# Crear y activar entorno virtual con Python 3.10+
python -m venv .venv

# En Windows:
.\.venv\Scripts\activate
# En Linux/macOS:
# source .venv/bin/activate

# Instalar dependencias exactas
pip install -r requirements.txt
```

### 2. Configuración de Variables de Entorno (.env)
Copia la plantilla `.env.example` para crear tu archivo `.env`:
```bash
cp .env.example .env
```
Edita `.env` con tus valores:
```ini
GEMINI_API_KEY=tu_clave_de_google_ai_studio
API_KEY_ESP32=clave_hexadecimal_de_32_caracteres
```
> **Obtención de la clave de Gemini**:
> 1. Accede a [Google AI Studio](https://aistudio.google.com/apikey).
> 2. Inicia sesión con tu cuenta de Google y genera una API Key gratuita.
> 3. Pega la clave en `GEMINI_API_KEY`.

### 3. Orden de Ejecución de Scripts (Pipeline Completo)

Para reproducir todo el pipeline de investigación de 01 a 11:
```bash
python 01_eda.py
python 02_limpieza_preprocesamiento.py
python 03_ingenieria_caracteristicas.py
python 04_particionamiento.py
python 05_entrenamiento.py
python 06_evaluacion_test.py
python 07_exportacion_modelo.py
python 08_interpretabilidad.py
python 09_monitoreo.py --registro monitoreo/registro_sesion_informe.csv --salida monitoreo/salida_09_reproducida.txt
python 10_red_neuronal.py
python 11_modelo_mitigado.py
```

### 4. Inicio del Dashboard en Tiempo Real
Inicia el servidor backend FastAPI:
```bash
uvicorn dashboard_iot.app:app --host 0.0.0.0 --port 8000 --reload
```
Abre en tu navegador:
```text
http://localhost:8000
```
- Puedes probar la inferencia y el servicio cognitivo sin hardware físico abriendo `http://localhost:8000/api/simulate/fire` o haciendo clic en el botón **"Explicar ahora"** en el panel de IA del dashboard.

---

## ⚠️ Limitaciones Conocidas y Trabajo Futuro

1. **Dependencia de la Presión Atmosférica**: Como se demostró mediante validación Leave-One-Group-Out, los modelos entrenados con 21 variables sobreajustan a las variaciones barométricas del laboratorio original. Se recomienda el despliegue del modelo mitigado de 20 variables (`modelos/random_forest_sin_presion.joblib`) para ambientes industriales.
2. **Deriva de Sensores de Bajo Costo**: Los sensores basados en semiconductor de óxido metálico (MP503 / MQ) presentan derivas térmicas por precalentamiento (primeros 60–120 segundos). El sistema clasifica adecuadamente este estado como línea base en reposo cuando `raw_aire < 1500`.
3. **Trabajo Futuro**:
   - Incorporación de sensores ópticos calibrados de material particulado por láser (ej. Plantower PMS5003).
   - Entrenamiento continuo en línea mediante TinyML directamente sobre núcleos ESP32 con TensorFlow Lite for Microcontrollers.

---

## 👥 Autores y Créditos

Proyecto desarrollado para el curso de **Machine Learning (Ciclo 8)**:

- **Kong Carranza, Luis**
- **Maximiliano Villanueva, Jhon**
- **Rodriguez Zavaleta, Jheyson**
- **Silvestre Miguel, Alex Paul**

**Docente Guía**:
- **Ing. Heber Abanto**

**Institución**:
- **Universidad Privada Antenor Orrego (UPAO)** — Trujillo, Perú, 2026.
