import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ====================================================================
# CONFIGURACIÓN Y RUTAS
# ====================================================================
BASE_DIR = Path(__file__).resolve().parent
DIR_PARTICIONES = BASE_DIR / "data_procesada" / "particiones"
DIR_MODELOS = BASE_DIR / "modelos"
DIR_RESULTADOS = BASE_DIR / "resultados_evaluacion" / "red_neuronal"

DIR_MODELOS.mkdir(parents=True, exist_ok=True)
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

RUTA_SALIDA_TXT = DIR_RESULTADOS / "salida_10.txt"


class LoggerTee:
    """Clase para duplicar la salida en consola y archivo .txt"""
    def __init__(self, ruta_archivo):
        self.terminal = sys.stdout
        self.archivo = open(ruta_archivo, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.archivo.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.archivo.flush()

    def close(self):
        self.archivo.close()


def main():
    logger = LoggerTee(RUTA_SALIDA_TXT)
    sys.stdout = logger

    print("=" * 85)
    print("TAREA 1: ENTRENAMIENTO Y EVALUACIÓN DE RED NEURONAL MLP (10_red_neuronal.py)")
    print("=" * 85)

    # 1. Carga de datos estandarizados
    print("\n1. CARGA DE PARTICIONES ESTANDARIZADAS")
    print("-" * 65)
    X_train_scaled = pd.read_csv(DIR_PARTICIONES / "X_train_scaled.csv")
    X_val_scaled = pd.read_csv(DIR_PARTICIONES / "X_val_scaled.csv")
    X_test_scaled = pd.read_csv(DIR_PARTICIONES / "X_test_scaled.csv")

    y_train = pd.read_csv(DIR_PARTICIONES / "y_train.csv")["Fire Alarm"]
    y_val = pd.read_csv(DIR_PARTICIONES / "y_val.csv")["Fire Alarm"]
    y_test = pd.read_csv(DIR_PARTICIONES / "y_test.csv")["Fire Alarm"]

    feature_names = list(X_train_scaled.columns)
    print(f"X_train_scaled: {X_train_scaled.shape}")
    print(f"X_val_scaled  : {X_val_scaled.shape}")
    print(f"X_test_scaled : {X_test_scaled.shape}")
    print(f"Características ({len(feature_names)}): {feature_names}")

    # 2. Configuración de MLPClassifier y RandomizedSearchCV
    print("\n2. BÚSQUEDA DE HIPERPARÁMETROS (RandomizedSearchCV)")
    print("-" * 65)
    mlp_base = MLPClassifier(
        activation="relu",
        solver="adam",
        early_stopping=True,
        max_iter=300,
        random_state=42
    )

    param_distributions = {
        "hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 5e-3]
    }

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=mlp_base,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="f1_macro",
        cv=cv_strat,
        random_state=42,
        n_jobs=1,
        verbose=1
    )

    print("Iniciando búsqueda aleatoria de hiperparámetros (10 iteraciones, 5 folds)...")
    t0_train = time.perf_counter()
    search.fit(X_train_scaled, y_train)
    t_train = time.perf_counter() - t0_train

    best_mlp = search.best_estimator_
    best_params = search.best_params_
    best_cv_f1 = search.best_score_

    # Evaluación en validación
    y_val_pred = best_mlp.predict(X_val_scaled)
    f1_val_macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

    print("\nResultados del Entrenamiento:")
    print(f"  - Mejores hiperparámetros: {best_params}")
    print(f"  - F1 Macro CV (5-fold)   : {best_cv_f1:.4f}")
    print(f"  - F1 Macro en Validación : {f1_val_macro:.4f}")
    print(f"  - Tiempo de entrenamiento: {t_train:.2f} s")

    # 3. Evaluación en conjunto de TEST
    print("\n3. EVALUACIÓN EN CONJUNTO DE TEST (Prueba Ciega)")
    print("-" * 65)
    y_test_pred = best_mlp.predict(X_test_scaled)
    y_test_proba = best_mlp.predict_proba(X_test_scaled)[:, 1]

    acc_test = accuracy_score(y_test, y_test_pred)
    prec_macro = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    rec_alarm = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    auc_test = roc_auc_score(y_test, y_test_proba)

    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"  - Accuracy              : {acc_test:.4f}")
    print(f"  - Precision Macro       : {prec_macro:.4f}")
    print(f"  - Recall Macro          : {rec_macro:.4f}")
    print(f"  - F1 Macro              : {f1_macro:.4f}")
    print(f"  - Recall Clase Alarma (1): {rec_alarm:.4f}")
    print(f"  - Falsos Negativos (FN) : {fn}")
    print(f"  - Falsos Positivos (FP) : {fp}")
    print(f"  - ROC-AUC               : {auc_test:.4f}")
    print("\nMatriz de Confusión en TEST:")
    print(f"                Pred 0 (Normal)   Pred 1 (Alarma)")
    print(f"Real 0 (Normal)      {tn:<17} {fp}")
    print(f"Real 1 (Alarma)      {fn:<17} {tp}")

    # 4. Medición de latencia media de predicción
    print("\n4. LATENCIA DE PREDICCIÓN POR OBSERVACIÓN")
    print("-" * 65)
    X_1000 = X_test_scaled.iloc[:1000]
    # Calentamiento
    _ = best_mlp.predict(X_1000[:50])
    t0_lat = time.perf_counter()
    _ = best_mlp.predict(X_1000)
    lat_total_ms = (time.perf_counter() - t0_lat) * 1000.0
    lat_por_obs_ms = lat_total_ms / 1000.0

    print(f"Tiempo total para 1000 observaciones: {lat_total_ms:.4f} ms")
    print(f"Tiempo medio de predicción por obs   : {lat_por_obs_ms:.4f} ms")

    # 5. Validación Leave-One-Group-Out (LOGO)
    print("\n5. VALIDACIÓN CRUZADA LEAVE-ONE-GROUP-OUT (LOGO) POR SESIONES")
    print("-" * 65)
    ruta_features = BASE_DIR / "data_procesada" / "smoke_detection_features.csv"
    df_raw = pd.read_csv(ruta_features)
    df_raw["session_id"] = (df_raw["CNT"].diff().fillna(0) < 0).cumsum()

    vars_sin_presion = [f for f in feature_names if f != "Pressure[hPa]"]
    vars_prototipo = [
        "Temperature[C]",
        "Humidity[%]",
        "TVOC[ppb]",
        "tvoc_delta",
        "tvoc_mean_5",
        "temp_delta",
        "humidity_delta"
    ]

    esquemas_logo = [
        ("a) 21 Variables (Completas)", feature_names),
        ("b) 20 Variables (Sin Pressure[hPa])", vars_sin_presion),
        ("c) 7 Variables del Prototipo", vars_prototipo)
    ]

    logo = LeaveOneGroupOut()
    groups = df_raw["session_id"]
    y_raw = df_raw["Fire Alarm"]

    for nombre_esq, columnas_esq in esquemas_logo:
        print(f"\n--- LOGO Esquema: {nombre_esq} ---")
        print(f"{'Sesión':<8} {'N Total':<10} {'Clase 0':<10} {'Clase 1':<10} {'F1 Macro':<12} {'Recall Alarma':<16} {'FN':<7} {'FP':<7} {'Observación':<20}")
        print("-" * 102)

        X_subset = df_raw[columnas_esq]
        res_sesiones = []

        for sess_id, (train_idx, test_idx) in enumerate(logo.split(X_subset, y_raw, groups)):
            X_tr, y_tr = X_subset.iloc[train_idx], y_raw.iloc[train_idx]
            X_te, y_te = X_subset.iloc[test_idx], y_raw.iloc[test_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    activation="relu",
                    solver="adam",
                    early_stopping=True,
                    max_iter=300,
                    random_state=42,
                    **best_params
                ))
            ])

            pipe.fit(X_tr, y_tr)
            y_pred_sess = pipe.predict(X_te)

            n_total = len(y_te)
            c0 = int((y_te == 0).sum())
            c1 = int((y_te == 1).sum())

            cm_sess = confusion_matrix(y_te, y_pred_sess, labels=[0, 1])
            tn_s, fp_s, fn_s, tp_s = cm_sess.ravel()

            clases_unicas = np.unique(y_te)
            if len(clases_unicas) == 1:
                acc_s = accuracy_score(y_te, y_pred_sess)
                f1_m = np.nan
                rec_al = np.nan
                obs = f"Clase única ({clases_unicas[0]}), Acc={acc_s:.4f}"
            else:
                acc_s = accuracy_score(y_te, y_pred_sess)
                f1_m = f1_score(y_te, y_pred_sess, average="macro", zero_division=0)
                rec_al = recall_score(y_te, y_pred_sess, pos_label=1, zero_division=0)
                obs = "Biclase"

            res_sesiones.append({
                "sess": sess_id,
                "total": n_total,
                "c0": c0,
                "c1": c1,
                "f1_macro": f1_m,
                "recall_alarm": rec_al,
                "acc": acc_s,
                "fn": fn_s,
                "fp": fp_s,
                "obs": obs
            })

            f1_str = f"{f1_m:.4f}" if not np.isnan(f1_m) else "N/A"
            rec_str = f"{rec_al:.4f}" if not np.isnan(rec_al) else "N/A"
            print(f"{sess_id:<8} {n_total:<10} {c0:<10} {c1:<10} {f1_str:<12} {rec_str:<16} {fn_s:<7} {fp_s:<7} {obs:<20}")

        print("-" * 102)
        f1_validos = [r["f1_macro"] for r in res_sesiones if not np.isnan(r["f1_macro"])]
        rec_validos = [r["recall_alarm"] for r in res_sesiones if not np.isnan(r["recall_alarm"])]
        prom_f1 = np.mean(f1_validos) if f1_validos else np.nan
        prom_rec = np.mean(rec_validos) if rec_validos else np.nan
        prom_fn = np.mean([r["fn"] for r in res_sesiones])
        prom_fp = np.mean([r["fp"] for r in res_sesiones])
        prom_acc = np.mean([r["acc"] for r in res_sesiones])

        print(f"{'PROMEDIO':<40} {prom_f1:>8.4f}     {prom_rec:>12.4f}     {prom_fn:>6.1f} {prom_fp:>6.1f}   Acc prom={prom_acc:.4f}")

    # 6. Guardado de modelo y gráficos
    print("\n6. GUARDADO DE ARTEFACTOS Y GRÁFICOS")
    print("-" * 65)

    # a) Modelo joblib
    ruta_modelo = DIR_MODELOS / "mlp.joblib"
    joblib.dump(best_mlp, ruta_modelo)
    print(f"  [OK] Modelo guardado en: {ruta_modelo}")

    # b) Matriz de confusión
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Alarma (1)"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusión - MLPClassifier\n(Conjunto de Test - {len(y_test)} muestras)")
    plt.tight_layout()
    ruta_png_cm = DIR_RESULTADOS / "matriz_confusion_mlp.png"
    plt.savefig(ruta_png_cm, dpi=150)
    plt.close()
    print(f"  [OK] Matriz de confusión guardada en: {ruta_png_cm}")

    # c) Curva de pérdida y validación
    plt.figure(figsize=(8, 5))
    plt.plot(best_mlp.loss_curve_, label="Pérdida de Entrenamiento (loss_curve_)", color="#1f77b4", lw=2)
    if hasattr(best_mlp, "validation_scores_") and best_mlp.validation_scores_ is not None and len(best_mlp.validation_scores_) > 0:
        plt.plot(best_mlp.validation_scores_, label="Score de Validación Interna (validation_scores_)", color="#ff7f0e", lw=2, linestyle="--")
    plt.xlabel("Épocas de Entrenamiento", fontsize=10)
    plt.ylabel("Pérdida / Score", fontsize=10)
    plt.title(f"Curva de Pérdida y Validación - MLPClassifier\n({best_params['hidden_layer_sizes']}, alpha={best_params['alpha']}, lr={best_params['learning_rate_init']})", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=9)
    plt.tight_layout()
    ruta_png_curva = DIR_RESULTADOS / "curva_perdida_mlp.png"
    plt.savefig(ruta_png_curva, dpi=150)
    plt.close()
    print(f"  [OK] Curva de pérdida guardada en: {ruta_png_curva}")

    print(f"  [OK] Salida completa guardada en: {RUTA_SALIDA_TXT}")
    print("\n" + "=" * 85)
    print("EJECUCIÓN DE 10_RED_NEURONAL.PY COMPLETADA EXITOSAMENTE")
    print("=" * 85)

    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
