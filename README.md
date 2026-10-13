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
- `data_procesada/`: Datasets limpios, transformados y particiones generadas.
- `modelos/`: Modelos entrenados y serializados en `.joblib`.
- `resultados_evaluacion/`: Métricas, matrices de confusión y gráficos comparativos.

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
