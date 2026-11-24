# REPORTE CONSOLIDADO DE ENTREGA Y REVISIÓN TÉCNICA
**Curso:** Machine Learning - Ciclo 8 | **Institución:** Universidad Privada Antenor Orrego (UPAO), 2026

**Autores:** Kong Carranza, Luis; Maximiliano Villanueva, Jhon; Rodriguez Zavaleta, Jheyson; Silvestre Miguel, Alex Paul

**Docente:** Ing. Heber Abanto

---

## 1. Salida de Red Neuronal MLP (`resultados_evaluacion/red_neuronal/salida_10.txt`)

```text
=====================================================================================
TAREA 1: ENTRENAMIENTO Y EVALUACIÓN DE RED NEURONAL MLP (10_red_neuronal.py)
=====================================================================================

1. CARGA DE PARTICIONES ESTANDARIZADAS
-----------------------------------------------------------------
X_train_scaled: (43841, 21)
X_val_scaled  : (9394, 21)
X_test_scaled : (9395, 21)
Características (21): ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'pm_ratio', 'pm25_delta', 'tvoc_delta', 'eco2_delta', 'temp_delta', 'humidity_delta', 'pm25_mean_5', 'tvoc_mean_5', 'eco2_mean_5']

2. BÚSQUEDA DE HIPERPARÁMETROS (RandomizedSearchCV)
-----------------------------------------------------------------
Iniciando búsqueda aleatoria de hiperparámetros (10 iteraciones, 5 folds)...
Fitting 5 folds for each of 10 candidates, totalling 50 fits

Resultados del Entrenamiento:
  - Mejores hiperparámetros: {'learning_rate_init': 0.005, 'hidden_layer_sizes': (64, 32), 'alpha': 0.0001}
  - F1 Macro CV (5-fold)   : 0.9994
  - F1 Macro en Validación : 0.9993
  - Tiempo de entrenamiento: 198.35 s

3. EVALUACIÓN EN CONJUNTO DE TEST (Prueba Ciega)
-----------------------------------------------------------------
  - Accuracy              : 0.9995
  - Precision Macro       : 0.9993
  - Recall Macro          : 0.9994
  - F1 Macro              : 0.9993
  - Recall Clase Alarma (1): 0.9996
  - Falsos Negativos (FN) : 3
  - Falsos Positivos (FP) : 2
  - ROC-AUC               : 0.9999

Matriz de Confusión en TEST:
                Pred 0 (Normal)   Pred 1 (Alarma)
Real 0 (Normal)      2679              2
Real 1 (Alarma)      3                 6711

4. LATENCIA DE PREDICCIÓN POR OBSERVACIÓN
-----------------------------------------------------------------
Tiempo total para 1000 observaciones: 1.2297 ms
Tiempo medio de predicción por obs   : 0.0012 ms

5. VALIDACIÓN CRUZADA LEAVE-ONE-GROUP-OUT (LOGO) POR SESIONES
-----------------------------------------------------------------

--- LOGO Esquema: a) 21 Variables (Completas) ---
Sesión   N Total    Clase 0    Clase 1    F1 Macro     Recall Alarma    FN      FP      Observación         
------------------------------------------------------------------------------------------------------
0        24994      3178       21816      0.1269       0.1068           19485   2266    Biclase             
1        24994      3178       21816      0.6402       0.6776           7034    15      Biclase             
2        1154       33         1121       0.0278       0.0000           1121    0       Biclase             
3        5744       5742       2          0.4316       0.0000           2       1381    Biclase             
4        5744       5742       2          0.4556       0.0000           2       935     Biclase             
------------------------------------------------------------------------------------------------------
PROMEDIO                                   0.3364           0.1569     5528.8  919.4   Acc prom=0.4945

--- LOGO Esquema: b) 20 Variables (Sin Pressure[hPa]) ---
Sesión   N Total    Clase 0    Clase 1    F1 Macro     Recall Alarma    FN      FP      Observación         
------------------------------------------------------------------------------------------------------
0        24994      3178       21816      0.1411       0.1227           19140   2226    Biclase             
1        24994      3178       21816      0.6228       0.6540           7548    17      Biclase             
2        1154       33         1121       0.1454       0.1481           955     15      Biclase             
3        5744       5742       2          0.4410       0.0000           2       1211    Biclase             
4        5744       5742       2          0.4412       0.0000           2       1207    Biclase             
------------------------------------------------------------------------------------------------------
PROMEDIO                                   0.3583           0.1850     5529.4  935.2   Acc prom=0.5161

--- LOGO Esquema: c) 7 Variables del Prototipo ---
Sesión   N Total    Clase 0    Clase 1    F1 Macro     Recall Alarma    FN      FP      Observación         
------------------------------------------------------------------------------------------------------
0        24994      3178       21816      0.1651       0.1018           19596   1268    Biclase             
1        24994      3178       21816      0.6535       0.6951           6652    13      Biclase             
2        1154       33         1121       0.0632       0.0375           1079    2       Biclase             
3        5744       5742       2          0.3465       0.0000           2       2696    Biclase             
4        5744       5742       2          0.2732       0.0000           2       3583    Biclase             
------------------------------------------------------------------------------------------------------
PROMEDIO                                   0.3003           0.1669     5466.2 1512.4   Acc prom=0.3736

6. GUARDADO DE ARTEFACTOS Y GRÁFICOS
-----------------------------------------------------------------
  [OK] Modelo guardado en: D:\UPAO 2026\CICLO 8\ML\Proyecto\Data set\archive (5)\modelos\mlp.joblib
  [OK] Matriz de confusión guardada en: D:\UPAO 2026\CICLO 8\ML\Proyecto\Data set\archive (5)\resultados_evaluacion\red_neuronal\matriz_confusion_mlp.png
  [OK] Curva de pérdida guardada en: D:\UPAO 2026\CICLO 8\ML\Proyecto\Data set\archive (5)\resultados_evaluacion\red_neuronal\curva_perdida_mlp.png
  [OK] Salida completa guardada en: D:\UPAO 2026\CICLO 8\ML\Proyecto\Data set\archive (5)\resultados_evaluacion\red_neuronal\salida_10.txt

=====================================================================================
EJECUCIÓN DE 10_RED_NEURONAL.PY COMPLETADA EXITOSAMENTE
=====================================================================================
```

## 2. Salida de Modelos Mitigados Sin Presión (`resultados_evaluacion/mitigacion/salida_11.txt`)

```text
=====================================================================================
TAREA 2: MODELOS MITIGADOS SIN PRESSURE[hPa] (11_modelo_mitigado.py)
=====================================================================================

1. CARGA DE PARTICIONES Y ELIMINACIÓN DE Pressure[hPa]
-----------------------------------------------------------------
Variables seleccionadas (20): ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'pm_ratio', 'pm25_delta', 'tvoc_delta', 'eco2_delta', 'temp_delta', 'humidity_delta', 'pm25_mean_5', 'tvoc_mean_5', 'eco2_mean_5']

2. ENTRENAMIENTO: ÁRBOL DE DECISIÓN (Sin Pressure[hPa])
-----------------------------------------------------------------
Fitting 5 folds for each of 20 candidates, totalling 100 fits

Resultados Árbol de Decisión (Sin Presión):
  - Mejores hiperparámetros: {'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 20, 'criterion': 'entropy', 'class_weight': 'balanced'}
  - F1 Macro CV (5-fold)   : 0.9993
  - F1 Macro en Validación : 0.9995
  - Tiempo de entrenamiento: 28.53 s
  [OK] Modelo guardado en: D:\UPAO 2026\CICLO 8\ML\Proyecto\Data set\archive (5)\modelos\decision_tree_sin_presion.joblib

3. ENTRENAMIENTO: RANDOM FOREST (Sin Pressure[hPa])
-----------------------------------------------------------------
Fitting 5 folds for each of 20 candidates, totalling 100 fits

Resultados Random Forest (Sin Presión):
  - Mejores hiperparámetros: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': None, 'class_weight': None}
  - F1 Macro CV (5-fold)   : 0.9998
  - F1 Macro en Validación : 0.9997
  - Tiempo de entrenamiento: 871.87 s
  [OK] Modelo guardado en: D:\UPAO 2026\CICLO 8\ML\Proyecto\Data set\archive (5)\modelos\random_forest_sin_presion.joblib

4. EVALUACIÓN COMPARATIVA EN TEST (20 Variables, Sin Presión)
-----------------------------------------------------------------

--- Métricas en TEST: Árbol de Decisión (Sin Presión) ---
  - Accuracy              : 0.9996
  - Precision Macro       : 0.9995
  - Recall Macro          : 0.9995
  - F1 Macro              : 0.9995
  - Recall Clase Alarma (1): 0.9997
  - Falsos Negativos (FN) : 2
  - Falsos Positivos (FP) : 2
  - ROC-AUC               : 0.9995
  - Latencia media por obs: 0.0008 ms
  - Matriz de Confusión en TEST:
                Pred 0 (Normal)   Pred 1 (Alarma)
    Real 0 (Normal)  2679              2
    Real 1 (Alarma)  2                 6712

--- Métricas en TEST: Random Forest (Sin Presión) ---
  - Accuracy              : 0.9999
  - Precision Macro       : 0.9998
  - Recall Macro          : 0.9999
  - F1 Macro              : 0.9999
  - Recall Clase Alarma (1): 0.9999
  - Falsos Negativos (FN) : 1
  - Falsos Positivos (FP) : 0
  - ROC-AUC               : 1.0000
  - Latencia media por obs: 0.0250 ms
  - Matriz de Confusión en TEST:
                Pred 0 (Normal)   Pred 1 (Alarma)
    Real 0 (Normal)  2681              0
    Real 1 (Alarma)  1                 6713

5. TOP 8 DE IMPORTANCIA MDI (Feature Importances) - MODELOS SIN PRESIÓN
-----------------------------------------------------------------

Top 8 Importancia MDI - Árbol de Decisión (Sin Presión):
Ranking  Variable               Importancia    Acumulada   
--------------------------------------------------------
1        tvoc_mean_5                0.3500         0.3500
2        Humidity[%]                0.2480         0.5979
3        Raw H2                     0.1035         0.7014
4        Raw Ethanol                0.0793         0.7807
5        TVOC[ppb]                  0.0741         0.8548
6        NC0.5                      0.0524         0.9072
7        Temperature[C]             0.0430         0.9502
8        eCO2[ppm]                  0.0342         0.9845

Top 8 Importancia MDI - Random Forest (Sin Presión):
Ranking  Variable               Importancia    Acumulada   
--------------------------------------------------------
1        tvoc_mean_5                0.1822         0.1822
2        TVOC[ppb]                  0.1748         0.3570
3        Raw Ethanol                0.1588         0.5157
4        Humidity[%]                0.1371         0.6529
5        Raw H2                     0.0848         0.7376
6        Temperature[C]             0.0416         0.7792
7        NC0.5                      0.0404         0.8195
8        pm25_mean_5                0.0337         0.8533

6. VALIDACIÓN CRUZADA LEAVE-ONE-GROUP-OUT (LOGO) - MODELOS SIN PRESIÓN
-----------------------------------------------------------------

--- LOGO por Sesiones: Árbol de Decisión (Sin Pressure[hPa]) ---
Sesión   N Total    Clase 0    Clase 1    F1 Macro     Recall Alarma    FN      FP      Observación         
------------------------------------------------------------------------------------------------------
0        24994      3178       21816      0.2614       0.1636           18246   204     Biclase             
1        24994      3178       21816      0.2370       0.1261           19066   0       Biclase             
2        1154       33         1121       0.2654       0.3301           751     17      Biclase             
3        5744       5742       2          0.4103       1.0000           0       1766    Biclase             
4        5744       5742       2          0.3022       1.0000           0       3262    Biclase             
------------------------------------------------------------------------------------------------------
PROMEDIO                                   0.2952           0.5240     7612.6 1049.8   Acc prom=0.3916

--- LOGO por Sesiones: Random Forest (Sin Pressure[hPa]) ---
Sesión   N Total    Clase 0    Clase 1    F1 Macro     Recall Alarma    FN      FP      Observación         
------------------------------------------------------------------------------------------------------
0        24994      3178       21816      0.3864       0.3120           15009   3       Biclase             
1        24994      3178       21816      0.6594       0.7017           6508    0       Biclase             
2        1154       33         1121       0.0683       0.0419           1074    1       Biclase             
3        5744       5742       2          0.4834       0.0000           2       368     Biclase             
4        5744       5742       2          0.4943       0.0000           2       127     Biclase             
------------------------------------------------------------------------------------------------------
PROMEDIO                                   0.4184           0.2111     4519.0   99.8   Acc prom=0.6241

  [OK] Salida completa guardada en: D:\UPAO 2026\CICLO 8\ML\Proyecto\Data set\archive (5)\resultados_evaluacion\mitigacion\salida_11.txt

=====================================================================================
EJECUCIÓN DE 11_MODELO_MITIGADO.PY COMPLETADA EXITOSAMENTE
=====================================================================================
```

## 3. Evidencia de Pruebas de Seguridad (`monitoreo/prueba_seguridad.txt`)

```text
======================================================================
PRUEBA DE SEGURIDAD - AUTENTICACIÓN X-API-Key EN /api/predict
======================================================================

1. POST /api/predict SIN cabecera X-API-Key:
   Código HTTP recibido : 401
   Resultado esperado    : 401 Unauthorized
   Cuerpo de la respuesta: {"detail":"Unauthorized: API Key inválida o ausente en cabecera X-API-Key"}
   Validación            : EXITOSA (401)

2. POST /api/predict CON cabecera X-API-Key INVÁLIDA:
   Código HTTP recibido : 401
   Resultado esperado    : 401 Unauthorized
   Cuerpo de la respuesta: {"detail":"Unauthorized: API Key inválida o ausente en cabecera X-API-Key"}
   Validación            : EXITOSA (401)

3. POST /api/predict CON cabecera X-API-Key VÁLIDA:
   Código HTTP recibido : 200
   Resultado esperado    : 200 OK
   Cuerpo de la respuesta: {"status":"ok","fire_alarm":0,"alarm_active":false,"probability_fire":0.0,"latencia_ms":17.9634,"action":"NORMAL_MONITORING","message":"Aire limpio / Condiciones normales","timestamp":"2026-11-24 13:40:16","origin_ip":"127.0.0.1","device_id":"TEST_SECURITY","readings":{"temperature":25.0,"humidity":50.0,"tvoc":15.0,"eco2":410.0,"raw_h2":12500.0,"raw_ethanol":18700.0,"pressure":939.5,"pm1_0":0.05,"pm2_5":0.05,"pm25":null,"nc0_5":0.1,"nc1_0":0.02,"nc2_5":0.005,"raw_aire":1280.0,"device_id":"TEST_SECURITY"},"explicacion_ia":"Esperando inferencias del sistema..."}
   Validación            : EXITOSA (200)

4. GET /api/health (Endpoint público sin clave):
   Código HTTP recibido : 200
   Validación            : ACCESO PÚBLICO CORRECTO (200)
```

## 4. Evidencia de Pruebas del Servicio Cognitivo Gemini (`monitoreo/prueba_gemini.txt`)

```text
======================================================================
PRUEBA DE SERVICIO COGNITIVO GEMINI - /api/simulate/fire Y /api/explicar
======================================================================

1. Inferencia disparada con GET /api/simulate/fire:
   Código HTTP : 200
   Respuesta   : {"status":"ok","fire_alarm":1,"alarm_active":true,"probability_fire":1.0,"latencia_ms":1.3587,"action":"TRIGGER_BUZZER_AND_LED","message":"PELIGRO: Humo o combustión detectada!","timestamp":"2026-11-24 13:40:16","origin_ip":"127.0.0.1","device_id":"SIMULADOR_WEB","readings":{"temperature":15.2,"humidity":52.8,"tvoc":867.6493784830286,"eco2":520.0,"raw_h2":13340.0,"raw_ethanol":20160.0,"pressure":939.5,"pm1_0":2.15,"pm2_5":2.3014982236959205,"nc0_5":14.8,"nc1_0":2.31,"nc2_5":0.052,"device_id":"SIMULADOR_WEB"},"explicacion_ia":"El sistema ha detectado una concentración inusualmente elevada de gases volátiles (TVOC) en el ambiente, lo que indica un posible proceso de combustión temprana. Debido a estos valores, el nivel de riesgo actual es **crítico**, con un 100% de probabilidad de peligro de incendio según el modelo de análisis. Le recomendamos inspeccionar la zona de inmediato con precaución y evacuar el lugar preventivamente si percibe olor a quemado o cualquier presencia visible de humo."}

2. Consulta a GET /api/explicar:
   Código HTTP : 200
   Respuesta   : {"status":"ok","explicacion_ia":"El sistema ha detectado una elevación crítica en los niveles de gases volátiles en el aire, lo que indica una posible combustión temprana en el ambiente. Debido a este hallazgo, el nivel de riesgo actual es de peligro inminente de incendio con un diagnóstico de alerta máxima. Se recomienda inspeccionar el área de inmediato en busca de posibles fuentes de origen y evacuar el lugar preventivamente si percibe cualquier anomalía."}

3. Texto explicativo generado por Gemini:
   "El sistema ha detectado una elevación crítica en los niveles de gases volátiles en el aire, lo que indica una posible combustión temprana en el ambiente. Debido a este hallazgo, el nivel de riesgo actual es de peligro inminente de incendio con un diagnóstico de alerta máxima. Se recomienda inspeccionar el área de inmediato en busca de posibles fuentes de origen y evacuar el lugar preventivamente si percibe cualquier anomalía."
```

## 5. Salida de Monitoreo Continuo Reproducido (`monitoreo/salida_09_reproducida.txt`)

```text
=====================================================================================
SISTEMA DE MONITOREO CONTINUO DE INFERENCIA Y DRIFT (09_monitoreo.py)
=====================================================================================

1. RESUMEN GENERAL DE OPERACIÓN
-----------------------------------------------------------------
Número total de inferencias       : 307
Inicio de sesión                  : 2026-11-07 21:23:46
Fin de sesión                     : 2026-11-07 21:26:41
Duración de la sesión             : 2 min 55 s (175.0 s)
Frecuencia media de inferencia    : 1.75 lecturas/s
Eventos de alarma disparados      : 33 (10.75 %)
Latencia de inferencia media      : 0.8133 ms
Latencia Percentil 95 (P95)       : 1.0987 ms
Latencia máxima observada         : 4.7481 ms
Huecos > 5 s sin lecturas         : 0 evento(s)

Resumen de episodios de alarma : 1 episodio(s) detectado(s)
  - Episodio 1:
      Hora de inicio : 2026-11-07 21:26:19
      Hora de fin    : 2026-11-07 21:26:41
      Duración       : 22.0 s (33 lecturas consecutivas)
      TVOC máximo    : 793.10 ppb
      raw_aire máximo: 2395.0

2. ESTADO DE SALUD DE SENSORES FÍSICOS
-----------------------------------------------------------------
Temperatura fuera de rango (0 - 50 °C) : 0 muestra(s) [CORRECTO]
Humedad fuera de rango (20 - 95 %)     : 0 muestra(s) [CORRECTO]

Auditoría de congelamiento de señal (umbral > 30 lecturas idénticas):
  - temperature   : Estable / Normal                   (máx. repeticiones consecutivas: 7)
  - humidity      : Estable (resolución DHT11 1 %)     (máx. repeticiones consecutivas: 137)
  - tvoc          : Línea base (sensor en reposo)      (máx. repeticiones consecutivas: 271)
  - eco2          : Línea base (sensor en reposo)      (máx. repeticiones consecutivas: 271)
  - raw_aire      : Estable / Normal                   (máx. repeticiones consecutivas: 7)
  - pm2_5         : Línea base (sensor en reposo)      (máx. repeticiones consecutivas: 271)

3. EVALUACIÓN DE DATA DRIFT RESPECTO AL DATASET DE ENTRENAMIENTO
-------------------------------------------------------------------------------------
Variable       Media Train    Media Vivo     KS Stat    p-valor      PSI (10)   Estado         
-------------------------------------------------------------------------------------
Temperature    15.9704        29.8785        0.9509     0.0000e+00   8.2703     Drift          
Humidity       48.5395        61.8013        0.9792     0.0000e+00   8.2767     Drift          
TVOC           1942.0575      83.1762        0.8173     3.0892e-224  6.8120     Drift          
eCO2           670.0210       423.5199       0.3730     9.8132e-39   1.1276     Drift          
-------------------------------------------------------------------------------------
Criterio PSI: Estable (< 0.10) | Moderado (0.10 - 0.25) | Drift (> 0.25)

4. GENERACIÓN DE GRÁFICOS DIAGNÓSTICOS EN monitoreo/
-----------------------------------------------------------------
  [OK] Guardado: probabilidad_prediccion_tiempo.png
  [OK] Guardado: latencia_tiempo.png
  [OK] Guardado: histograma_comparativo_temperatura_tvoc.png
```

## 6. Git Diff (`dashboard_iot/app.py`, `dashboard_iot/templates/index.html`, `dashboard_iot/esp32_firmware/esp32_monitor.ino`)

```diff
diff --git a/dashboard_iot/app.py b/dashboard_iot/app.py
index 2b045f5..3220807 100644
--- a/dashboard_iot/app.py
+++ b/dashboard_iot/app.py
@@ -17,6 +17,8 @@ from fastapi.responses import HTMLResponse, JSONResponse
 from fastapi.middleware.cors import CORSMiddleware
 from pydantic import BaseModel, Field
 import uvicorn
+from dotenv import load_dotenv
+from google import genai
 
 # ====================================================================
 # 1. CONFIGURACIÓN Y RUTAS
@@ -25,6 +27,12 @@ BASE_DIR = Path(__file__).resolve().parent
 PROJECT_DIR = BASE_DIR.parent
 MODEL_PATH = PROJECT_DIR / "modelos" / "modelo_mejor_validacion.joblib"
 
+# Cargar variables de entorno desde ../.env de forma absoluta
+ENV_PATH = PROJECT_DIR / ".env"
+load_dotenv(dotenv_path=ENV_PATH)
+GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
+API_KEY_ESP32 = os.getenv("API_KEY_ESP32", "")
+
 if not MODEL_PATH.exists():
     MODEL_PATH = PROJECT_DIR / "modelos" / "decision_tree.joblib"
 
@@ -202,12 +210,15 @@ manager = ConnectionManager()
 # ====================================================================
 MONITOR_DIR = PROJECT_DIR / "monitoreo"
 REGISTRO_CSV = MONITOR_DIR / "registro_inferencias.csv"
+REGISTRO_GEMINI_CSV = MONITOR_DIR / "registro_gemini.csv"
 csv_lock = Lock()
 
 ultima_lectura_esp32: Optional[float] = None
 total_inferencias: int = 0
 total_alarmas: int = 0
 acumulador_latencias: List[float] = []
+ultima_prediccion_global: int = 0
+ultima_explicacion_ia: str = "Esperando inferencias del sistema..."
 
 COLUMNAS_MONITOREO = [
     "timestamp",
@@ -267,6 +278,78 @@ def registrar_inferencia_csv(
                 round(latencia_ms, 4)
             ])
 
+
+def registrar_gemini_csv(fecha_hora: str, latencia_ms: float, exito: str, texto_o_error: str):
+    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
+    with csv_lock:
+        es_nuevo = not REGISTRO_GEMINI_CSV.exists()
+        with open(REGISTRO_GEMINI_CSV, mode="a", newline="", encoding="utf-8") as f:
+            writer = csv.writer(f)
+            if es_nuevo:
+                writer.writerow(["fecha_hora", "latencia_ms", "exito", "texto_o_error"])
+            writer.writerow([fecha_hora, round(latencia_ms, 2), exito, texto_o_error])
+
+
+def consultar_gemini_explicacion(datos_inferencia: dict) -> str:
+    """
+    Solicita a Gemini una explicación concisa (2 a 3 frases) en español para un ciudadano.
+    Indica que el sistema es experimental y no reemplaza un detector de humo certificado.
+    """
+    global ultima_explicacion_ia
+    t0_gem = time.perf_counter()
+    ahora_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
+
+    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "" or GEMINI_API_KEY == "PEGAR_AQUI":
+        error_msg = "Clave GEMINI_API_KEY no configurada o vacía en .env"
+        registrar_gemini_csv(ahora_str, 0.0, "no", error_msg)
+        ultima_explicacion_ia = "Explicación no disponible: servicio cognitivo sin conexión"
+        return ultima_explicacion_ia
+
+    pred_val = datos_inferencia.get("prediccion", datos_inferencia.get("fire_alarm", 0))
+    prob_val = datos_inferencia.get("probabilidad", datos_inferencia.get("probability_fire", 0.0))
+
+    prompt = (
+        "Eres un asistente de seguridad ambiental para un sistema experimental de detección temprana de incendios con IoT y Machine Learning. "
+        "AVISO OBLIGATORIO: Este prototipo es académico experimental y no reemplaza a un detector de humo certificado por normas de seguridad. "
+        f"Datos del sensor en tiempo real: "
+        f"Temperatura: {datos_inferencia.get('temperature', 24.0)} °C, "
+        f"Humedad: {datos_inferencia.get('humidity', 50.0)} %, "
+        f"TVOC (gases volátiles): {datos_inferencia.get('tvoc', 10.0)} ppb, "
+        f"eCO2: {datos_inferencia.get('eco2', 400.0)} ppm, "
+        f"PM2.5: {datos_inferencia.get('pm2_5', 0.05)} ug/m3, "
+        f"Gas analógico MP503 (raw_aire): {datos_inferencia.get('raw_aire', 'N/D')}. "
+        f"Diagnóstico del modelo ML: {'PELIGRO DE INCENDIO / COMBUSTIÓN DETECTADA' if pred_val == 1 else 'CONDICIÓN NORMAL / AIRE LIMPIO'}, "
+        f"Probabilidad de alarma: {float(prob_val) * 100:.1f} %. "
+        "Redacta una explicación en español de exactamente 2 a 3 frases dirigida a un ciudadano común: "
+        "1) qué detectó el sensor en el aire, 2) nivel de riesgo actual, y 3) recomendación preventiva inmediata."
+    )
+
+    try:
+        client = genai.Client(api_key=GEMINI_API_KEY)
+        texto = None
+        # Probar modelos compatibles con la API
+        for modelo_nombre in ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.8-flash"]:
+            try:
+                resp = client.models.generate_content(model=modelo_nombre, contents=prompt)
+                if resp and resp.text:
+                    texto = resp.text.strip()
+                    break
+            except Exception:
+                continue
+
+        lat_gem_ms = (time.perf_counter() - t0_gem) * 1000.0
+        if texto:
+            registrar_gemini_csv(ahora_str, lat_gem_ms, "sí", texto)
+            ultima_explicacion_ia = texto
+            return texto
+        else:
+            raise RuntimeError("Respuesta vacía de los modelos Gemini disponibles")
+    except Exception as e:
+        lat_gem_ms = (time.perf_counter() - t0_gem) * 1000.0
+        registrar_gemini_csv(ahora_str, lat_gem_ms, "no", str(e))
+        ultima_explicacion_ia = "Explicación no disponible: servicio cognitivo sin conexión"
+        return ultima_explicacion_ia
+
 # Modelo de datos de entrada desde ESP32
 class LecturaSensor(BaseModel):
     temperature: float = Field(..., example=24.5)
@@ -324,19 +407,22 @@ def get_health():
         "porcentaje_alarmas": porcentaje_alarmas
     }
 
-@app.post("/api/predict")
-async def predict_sensor_data(lectura: LecturaSensor, request: Request):
+async def procesar_inferencia_completa(datos_dict: dict, ip_cliente: str, device_id: Optional[str] = "ESP32_UPAO"):
     """
-    Endpoint principal para recibir datos desde el ESP32 o simulador.
+    Función central de procesamiento de inferencia:
+    - Preprocesamiento y cálculo de deltas/medias móviles en buffer
+    - Medición de latencia conjunta de predict y predict_proba
+    - Registro en CSV
+    - Llamada automática a Gemini solo en transición 0 -> 1
+    - Emisión a WebSocket
     """
-    datos_dict = lectura.model_dump() if hasattr(lectura, "model_dump") else lectura.dict()
+    global ultima_prediccion_global, ultima_explicacion_ia
+
     df_features, features_calc = historial.procesar_lectura(datos_dict)
 
-    # Inferencia del modelo de ML
+    # Medición de latencia alrededor de predict y predict_proba juntos
+    t0_inf = time.perf_counter()
     prediccion = int(modelo.predict(df_features)[0])
-    
-    # Medición de latencia_ms con time.perf_counter() alrededor de predict_proba
-    t0 = time.perf_counter()
     probabilidad = 1.0 if prediccion == 1 else 0.0
     if hasattr(modelo, "predict_proba"):
         try:
@@ -344,12 +430,11 @@ async def predict_sensor_data(lectura: LecturaSensor, request: Request):
             probabilidad = float(prob_arr[1])
         except Exception:
             pass
-    latencia_ms = (time.perf_counter() - t0) * 1000.0
+    latencia_ms = (time.perf_counter() - t0_inf) * 1000.0
 
     hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
-    ip_cliente = request.client.host if request.client else "local"
 
-    # Registrar en monitoreo/registro_inferencias.csv
+    # Obtener valores crudos y de partículas
     raw_aire_val = datos_dict.get("raw_aire")
     if raw_aire_val is None:
         raw_aire_val = datos_dict.get("rawAire")
@@ -358,6 +443,7 @@ async def predict_sensor_data(lectura: LecturaSensor, request: Request):
     if pm25_val is None:
         pm25_val = datos_dict.get("pm25", 0.0)
 
+    # Registrar en monitoreo/registro_inferencias.csv
     registrar_inferencia_csv(
         timestamp=hora_actual,
         ip_origen=ip_cliente,
@@ -372,19 +458,35 @@ async def predict_sensor_data(lectura: LecturaSensor, request: Request):
         probabilidad=probabilidad,
         latencia_ms=latencia_ms
     )
-    
+
+    # Llamada automática a Gemini SOLO cuando la predicción cambie de 0 a 1
+    explicacion_nueva = None
+    if prediccion == 1 and ultima_prediccion_global == 0:
+        datos_para_explicar = {
+            **datos_dict,
+            "prediccion": prediccion,
+            "probabilidad": probabilidad
+        }
+        explicacion_nueva = await asyncio.to_thread(consultar_gemini_explicacion, datos_para_explicar)
+        ultima_explicacion_ia = explicacion_nueva
+    ultima_prediccion_global = prediccion
+
+    texto_ia = explicacion_nueva or ultima_explicacion_ia
+
     # Respuesta estructurada para el ESP32 y el Dashboard
     resultado = {
         "status": "ok",
         "fire_alarm": prediccion,
         "alarm_active": bool(prediccion == 1),
         "probability_fire": round(probabilidad, 4),
+        "latencia_ms": round(latencia_ms, 4),
         "action": "TRIGGER_BUZZER_AND_LED" if prediccion == 1 else "NORMAL_MONITORING",
         "message": "PELIGRO: Humo o combustión detectada!" if prediccion == 1 else "Aire limpio / Condiciones normales",
         "timestamp": hora_actual,
         "origin_ip": ip_cliente,
-        "device_id": lectura.device_id,
-        "readings": datos_dict
+        "device_id": device_id,
+        "readings": datos_dict,
+        "explicacion_ia": texto_ia
     }
 
     # Guardar en historial de memoria para recargas rápidas
@@ -393,15 +495,72 @@ async def predict_sensor_data(lectura: LecturaSensor, request: Request):
     # Transmitir a todos los clientes web conectados al Dashboard
     await manager.broadcast({
         "type": "NEW_READING",
-        "payload": resultado
+        "payload": resultado,
+        "explicacion_ia": texto_ia
     })
 
     return resultado
 
+
+@app.post("/api/predict")
+async def predict_sensor_data(lectura: LecturaSensor, request: Request):
+    """
+    Endpoint principal para recibir datos desde el ESP32.
+    Protegido con la cabecera X-API-Key comparada con API_KEY_ESP32 del .env.
+    """
+    api_key_header = request.headers.get("X-API-Key")
+    if not API_KEY_ESP32 or api_key_header != API_KEY_ESP32:
+        return JSONResponse(
+            status_code=401,
+            content={"detail": "Unauthorized: API Key inválida o ausente en cabecera X-API-Key"}
+        )
+
+    datos_dict = lectura.model_dump() if hasattr(lectura, "model_dump") else lectura.dict()
+    ip_cliente = request.client.host if request.client else "local"
+    return await procesar_inferencia_completa(datos_dict, ip_cliente, lectura.device_id)
+
+
+@app.get("/api/explicar")
+async def api_explicar():
+    """
+    Solicita a Gemini una explicación de 2 a 3 frases sobre la última inferencia registrada.
+    """
+    global ultima_explicacion_ia
+    if historial.historial_graficas:
+        ultima_inf = historial.historial_graficas[-1]
+        datos = ultima_inf.get("readings", {}).copy()
+        datos["prediccion"] = ultima_inf.get("fire_alarm", 0)
+        datos["probabilidad"] = ultima_inf.get("probability_fire", 0.0)
+    elif historial.ultima_lectura:
+        datos = historial.ultima_lectura.copy()
+        datos["prediccion"] = 0
+        datos["probabilidad"] = 0.0
+    else:
+        datos = {
+            "temperature": 24.0, "humidity": 50.0, "tvoc": 10.0, "eco2": 400.0,
+            "pm2_5": 0.05, "raw_aire": 1260, "prediccion": 0, "probabilidad": 0.0
+        }
+
+    texto = await asyncio.to_thread(consultar_gemini_explicacion, datos)
+    ultima_explicacion_ia = texto
+
+    # Difundir a clientes WebSocket conectados
+    await manager.broadcast({
+        "type": "EXPLICACION_IA",
+        "explicacion_ia": texto
+    })
+
+    return {
+        "status": "ok",
+        "explicacion_ia": texto
+    }
+
+
 @app.get("/api/history")
 def get_history():
     return list(historial.historial_graficas)
 
+
 # Datos de muestra del dataset original para simular con exactitud
 MUESTRAS_LIMPIAS = [
     {"temperature": 20.0, "humidity": 57.36, "tvoc": 0.0, "eco2": 400.0, "raw_h2": 12306.0, "raw_ethanol": 18520.0, "pressure": 939.735, "pm1_0": 0.0, "pm2_5": 0.0, "nc0_5": 0.0, "nc1_0": 0.0, "nc2_5": 0.0},
@@ -421,13 +580,12 @@ indice_fuego = 0
 @app.get("/api/simulate/{scenario}")
 async def simulate(scenario: str, request: Request):
     """
-    Permite probar el sistema directamente desde el navegador sin hardware conectado.
+    Permite probar el sistema directamente desde el navegador o pruebas sin cabecera de hardware.
     """
     global indice_clean, indice_fuego
     if scenario == "fire":
         base = MUESTRAS_FUEGO[indice_fuego % len(MUESTRAS_FUEGO)].copy()
         indice_fuego += 1
-        # Pequeña variación estocástica
         base["tvoc"] += np.random.uniform(-20, 20)
         base["pm2_5"] += np.random.uniform(-0.1, 0.2)
     else:
@@ -437,8 +595,8 @@ async def simulate(scenario: str, request: Request):
         base["humidity"] += np.random.uniform(-0.5, 0.5)
 
     base["device_id"] = "SIMULADOR_WEB"
-    lectura = LecturaSensor(**base)
-    return await predict_sensor_data(lectura, request)
+    ip_cliente = request.client.host if request.client else "local"
+    return await procesar_inferencia_completa(base, ip_cliente, "SIMULADOR_WEB")
 
 @app.websocket("/ws")
 async def websocket_endpoint(websocket: WebSocket):
@@ -450,7 +608,8 @@ async def websocket_endpoint(websocket: WebSocket):
             "local_ip": LOCAL_IP,
             "port": PORT,
             "endpoint": f"http://{LOCAL_IP}:{PORT}/api/predict",
-            "history": list(historial.historial_graficas)
+            "history": list(historial.historial_graficas),
+            "explicacion_ia": ultima_explicacion_ia
         })
         while True:
             # Mantener conexión viva
@@ -470,67 +629,21 @@ async def polling_esp32_ap():
     """
     Si la PC se conecta al Wi-Fi 'ML' del ESP32, esta tarea lee los sensores
     desde http://192.168.4.1/api/sensors automaticamente y manda la prediccion.
+    (Vía de consulta interna que no requiere cabecera de autenticación).
     """
     await asyncio.sleep(2)
     while True:
         try:
-            # Consultar en hilo separado para no bloquear el bucle de eventos
             def fetch_data():
                 return requests.get("http://192.168.4.1/api/sensors", timeout=0.8)
 
             resp = await asyncio.to_thread(fetch_data)
             if resp.status_code == 200:
                 data = resp.json()
-                lectura = LecturaSensor(**data)
-                datos_dict = lectura.model_dump()
-                df_features, features_calc = historial.procesar_lectura(datos_dict)
-
-                prediccion = int(modelo.predict(df_features)[0])
-                
-                t0 = time.perf_counter()
-                probabilidad = 1.0 if prediccion == 1 else 0.0
-                if hasattr(modelo, "predict_proba"):
-                    try:
-                        probabilidad = float(modelo.predict_proba(df_features)[0][1])
-                    except Exception:
-                        pass
-                latencia_ms = (time.perf_counter() - t0) * 1000.0
-
-                hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
-                raw_aire_val = data.get("raw_aire", data.get("rawAire"))
-
-                registrar_inferencia_csv(
-                    timestamp=hora_actual,
-                    ip_origen="192.168.4.1",
-                    temperature=float(data.get("temperature", 0.0)),
-                    humidity=float(data.get("humidity", 0.0)),
-                    raw_aire=float(raw_aire_val) if raw_aire_val is not None else None,
-                    tvoc=float(data.get("tvoc", 0.0)),
-                    pm2_5=float(data.get("pm2_5", data.get("pm25", 0.0))),
-                    eco2=float(data.get("eco2", 0.0)),
-                    tvoc_mean_5=float(features_calc.get("tvoc_mean_5", 0.0)),
-                    prediccion=prediccion,
-                    probabilidad=probabilidad,
-                    latencia_ms=latencia_ms
-                )
-
-                resultado = {
-                    "status": "ok",
-                    "fire_alarm": prediccion,
-                    "alarm_active": bool(prediccion == 1),
-                    "probability_fire": round(probabilidad, 4),
-                    "action": "TRIGGER_BUZZER_AND_LED" if prediccion == 1 else "NORMAL_MONITORING",
-                    "message": "PELIGRO: Humo o combustión detectada!" if prediccion == 1 else "Aire limpio / Condiciones normales",
-                    "timestamp": hora_actual,
-                    "origin_ip": "192.168.4.1",
-                    "device_id": data.get("device_id", "ESP32_AP_ML"),
-                    "readings": datos_dict
-                }
-
-                historial.historial_graficas.append(resultado)
-                await manager.broadcast({"type": "NEW_READING", "payload": resultado})
+                resultado = await procesar_inferencia_completa(data, "192.168.4.1", data.get("device_id", "ESP32_AP_ML"))
 
                 # Notificar al ESP32 para activar o silenciar buzzer/LEDs
+                prediccion = resultado.get("fire_alarm", 0)
                 def send_alarm():
                     requests.get(f"http://192.168.4.1/set_alarm?fire={prediccion}", timeout=0.6)
                 await asyncio.to_thread(send_alarm)
diff --git a/dashboard_iot/esp32_firmware/esp32_monitor.ino b/dashboard_iot/esp32_firmware/esp32_monitor.ino
index a22512b..0d6e156 100644
--- a/dashboard_iot/esp32_firmware/esp32_monitor.ino
+++ b/dashboard_iot/esp32_firmware/esp32_monitor.ino
@@ -32,7 +32,8 @@
 // 1. CONFIGURACIÓN DEL WI-FI EMITIDO POR EL ESP32
 // ========================================================================
 const char* ap_ssid     = "ML";
-const char* ap_password = "12345678";
+const char* ap_password = "SmokeGuard2026";
+const char* API_KEY     = "c45ae948e676f03d5c12044634f705de";
 
 WebServer server(80);
 
@@ -204,6 +205,7 @@ void loop() {
       HTTPClient http;
       http.begin("http://192.168.4.2:8000/api/predict");
       http.addHeader("Content-Type", "application/json");
+      http.addHeader("X-API-Key", API_KEY);
 
       String jsonPayload = "{";
       jsonPayload += "\"temperature\":" + String(temp, 2) + ",";
diff --git a/dashboard_iot/templates/index.html b/dashboard_iot/templates/index.html
index e9eee7b..e489f6d 100644
--- a/dashboard_iot/templates/index.html
+++ b/dashboard_iot/templates/index.html
@@ -478,8 +478,9 @@
         <h1>
           Monitor de Calidad de Aire & Detección de Fuego
           <span class="badge-course">UPAO - Ciclo 8</span>
+          <span class="badge-experimental" style="font-size:0.72rem;font-weight:600;padding:3px 10px;border-radius:6px;background:#fef3c7;color:#92400e;border:1px solid #fde68a;margin-left:8px;vertical-align:middle;">Prototipo académico experimental</span>
         </h1>
-        <p>Inferencia en Tiempo Real • Modelo: <strong>DecisionTreeClassifier</strong> (F1: 100%)</p>
+        <p>Inferencia en Tiempo Real • Modelo: <strong>DecisionTreeClassifier</strong></p>
       </div>
     </div>
 
@@ -502,7 +503,7 @@
         </div>
         <div class="strip-text-group">
           <h2 id="status-title">Condición Ambiental Normal</h2>
-          <p id="status-desc">Valores estables de material particulado y gases. Sin presencia de humo.</p>
+          <p id="status-desc">Valores estables de calidad de aire. Sin presencia de humo.</p>
         </div>
       </div>
 
@@ -585,9 +586,9 @@
           <span>Sensor Aire (AOUT)</span>
           <svg stroke="var(--safe-color)" viewBox="0 0 24 24"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/></svg>
         </div>
-        <div class="card-number" id="val-raw">1260</div>
+        <div class="card-number" id="val-raw">—</div>
         <div class="card-unit">ADC GPIO 34 (0-4095)</div>
-        <div class="card-track"><div id="bar-raw" class="card-bar" style="background:var(--safe-color);width:32%;"></div></div>
+        <div class="card-track"><div id="bar-raw" class="card-bar" style="background:var(--safe-color);width:0%;"></div></div>
       </div>
     </section>
 
@@ -622,6 +623,23 @@
       </div>
     </section>
 
+    <!-- PANEL EXPLICACIÓN IA (GEMINI) -->
+    <section class="table-card" style="margin-bottom: 24px; padding: 20px;">
+      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
+        <div class="table-title" style="margin:0; display:flex; align-items:center; gap:8px;">
+          <svg width="20" height="20" fill="none" stroke="#2563eb" stroke-width="2.2" viewBox="0 0 24 24"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 14h-2v-2h2zm0-4h-2V7h2z"/></svg>
+          Explicación IA (Gemini)
+        </div>
+        <button id="btn-explicar" onclick="solicitarExplicacionIA()" style="background:#2563eb; color:#fff; border:none; padding:7px 16px; border-radius:6px; font-weight:600; cursor:pointer; font-size:0.85rem;">Explicar ahora</button>
+      </div>
+      <div id="texto-explicacion-ia" style="font-size:0.95rem; line-height:1.5; color:#334155; background:#f8fafc; padding:14px; border-radius:8px; border:1px solid #e2e8f0; min-height:48px;">
+        Esperando eventos o solicita una explicación manual con el botón.
+      </div>
+      <div style="margin-top:8px; font-size:0.75rem; color:#64748b;">
+        * Nota: Prototipo experimental. Las interpretaciones de IA asisten pero no reemplazan sistemas de detección de incendios certificados.
+      </div>
+    </section>
+
     <!-- REGISTRO DE AUDITORÍA -->
     <section class="table-card">
       <div class="table-title">Registro Secuencial de Inferencia del Modelo</div>
@@ -819,6 +837,11 @@
           const msg = JSON.parse(event.data);
           if (msg.type === 'NEW_READING') {
             updateDashboard(msg.payload);
+            if (msg.payload && msg.payload.explicacion_ia) {
+              document.getElementById('texto-explicacion-ia').innerText = msg.payload.explicacion_ia;
+            }
+          } else if (msg.type === 'EXPLICACION_IA') {
+            document.getElementById('texto-explicacion-ia').innerText = msg.explicacion_ia;
           } else if (msg.type === 'INIT' && msg.history) {
             msg.history.forEach(item => updateDashboard(item));
           }
@@ -851,7 +874,7 @@
         iconBox.className = "strip-icon-box danger";
         iconBox.innerHTML = '<svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>';
         title.innerText = "Alerta: Detección de Humo o Combustión";
-        desc.innerText = "El modelo identificó una concentración crítica de material particulado y gases.";
+        desc.innerText = "El modelo identificó una concentración elevada de gases detectada por el sensor MP503.";
         prob.style.color = "var(--fire-color)";
         prob.innerText = `${(payload.probability_fire * 100).toFixed(1)}%`;
         act.className = "actuator-tag danger";
@@ -861,7 +884,7 @@
         iconBox.className = "strip-icon-box safe";
         iconBox.innerHTML = '<svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>';
         title.innerText = "Condición Ambiental Normal";
-        desc.innerText = "Valores estables de material particulado y gases. Sin presencia de humo.";
+        desc.innerText = "Valores estables de calidad de aire. Sin presencia de humo.";
         prob.style.color = "var(--safe-color)";
         prob.innerText = `${(payload.probability_fire * 100).toFixed(1)}%`;
         act.className = "actuator-tag safe";
@@ -876,7 +899,7 @@
       const eco2 = parseFloat(r.eco2 || 400);
       const temp = parseFloat(r.temperature || 30.8);
       const hum = parseFloat(r.humidity || 54.0);
-      const raw = parseInt(r.raw_aire || 1260);
+      const raw = (r.raw_aire !== undefined && r.raw_aire !== null && r.raw_aire !== "") ? parseInt(r.raw_aire) : null;
 
       document.getElementById('val-pm25').innerText = pm25.toFixed(2);
       document.getElementById('bar-pm25').style.width = Math.min((pm25 / 5.0) * 100, 100) + '%';
@@ -893,8 +916,8 @@
       document.getElementById('val-hum').innerText = hum.toFixed(1) + '%';
       document.getElementById('bar-hum').style.width = Math.min(hum, 100) + '%';
 
-      document.getElementById('val-raw').innerText = raw;
-      document.getElementById('bar-raw').style.width = Math.min((raw / 3500) * 100, 100) + '%';
+      document.getElementById('val-raw').innerText = (raw !== null && !isNaN(raw)) ? raw : '—';
+      document.getElementById('bar-raw').style.width = (raw !== null && !isNaN(raw)) ? Math.min((raw / 3500) * 100, 100) + '%' : '0%';
 
       // Actualizar Gráficas Chart.js en tiempo real
       if (timeLabels.length >= maxDataPoints) {
@@ -935,6 +958,23 @@
       }
     }
 
+    async function solicitarExplicacionIA() {
+      const btn = document.getElementById('btn-explicar');
+      const txt = document.getElementById('texto-explicacion-ia');
+      btn.disabled = true;
+      btn.innerText = 'Consultando...';
+      try {
+        const res = await fetch('/api/explicar');
+        const data = await res.json();
+        txt.innerText = data.explicacion || data.error || 'Sin respuesta';
+      } catch(e) {
+        txt.innerText = 'Error al consultar explicación IA: ' + e.message;
+      } finally {
+        btn.disabled = false;
+        btn.innerText = 'Explicar ahora';
+      }
+    }
+
     connectRealTimeWS();
   </script>
 </body>
```

## 7. Dependencias del Proyecto (`requirements.txt`)

```text
pandas==3.1.1
numpy==2.5.5
scikit-learn==1.9.1
matplotlib==3.11.2
joblib==1.6.0
scipy==1.18.1
fastapi==0.165.0
uvicorn==0.54.0
pydantic==2.14.4
requests==2.34.2
python-dotenv==1.2.4
google-genai==3.6.1
```

## 8. Lista de Archivos Creados y Modificados

### Archivos Creados:
- `10_red_neuronal.py`
- `11_modelo_mitigado.py`
- `modelos/mlp.joblib`
- `modelos/decision_tree_sin_presion.joblib`
- `modelos/random_forest_sin_presion.joblib`
- `resultados_evaluacion/red_neuronal/salida_10.txt`
- `resultados_evaluacion/red_neuronal/matriz_confusion_mlp.png`
- `resultados_evaluacion/red_neuronal/curva_perdida_mlp.png`
- `resultados_evaluacion/mitigacion/salida_11.txt`
- `resultados_evaluacion/interpretabilidad/salida_08.txt`
- `monitoreo/registro_sesion_informe.csv`
- `monitoreo/registro_gemini.csv`
- `monitoreo/salida_09_reproducida.txt`
- `monitoreo/prueba_seguridad.txt`
- `monitoreo/prueba_gemini.txt`
- `requirements.txt`
- `.env.example`
- `ENTREGA_REVISION.md`

### Archivos Modificados:
- `08_interpretabilidad.py`
- `09_monitoreo.py`
- `dashboard_iot/app.py`
- `dashboard_iot/templates/index.html`
- `dashboard_iot/esp32_firmware/esp32_monitor.ino`
- `.gitignore`
- `README.md`