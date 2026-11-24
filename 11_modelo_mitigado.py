import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ====================================================================
# CONFIGURACIÓN Y RUTAS
# ====================================================================
BASE_DIR = Path(__file__).resolve().parent
DIR_PARTICIONES = BASE_DIR / "data_procesada" / "particiones"
DIR_MODELOS = BASE_DIR / "modelos"
DIR_RESULTADOS = BASE_DIR / "resultados_evaluacion" / "mitigacion"

DIR_MODELOS.mkdir(parents=True, exist_ok=True)
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

RUTA_SALIDA_TXT = DIR_RESULTADOS / "salida_11.txt"


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


def evaluar_en_test(nombre, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    if hasattr(modelo, "predict_proba"):
        y_prob = modelo.predict_proba(X_test)[:, 1]
    else:
        y_prob = modelo.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    rec_alarm = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    auc_val = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Latencia por observación
    X_1000 = X_test.iloc[:1000]
    _ = modelo.predict(X_1000.iloc[:50])
    t0_lat = time.perf_counter()
    _ = modelo.predict(X_1000)
    lat_ms = ((time.perf_counter() - t0_lat) / 1000.0) * 1000.0

    print(f"\n--- Métricas en TEST: {nombre} ---")
    print(f"  - Accuracy              : {acc:.4f}")
    print(f"  - Precision Macro       : {prec_macro:.4f}")
    print(f"  - Recall Macro          : {rec_macro:.4f}")
    print(f"  - F1 Macro              : {f1_macro:.4f}")
    print(f"  - Recall Clase Alarma (1): {rec_alarm:.4f}")
    print(f"  - Falsos Negativos (FN) : {fn}")
    print(f"  - Falsos Positivos (FP) : {fp}")
    print(f"  - ROC-AUC               : {auc_val:.4f}")
    print(f"  - Latencia media por obs: {lat_ms:.4f} ms")
    print("  - Matriz de Confusión en TEST:")
    print(f"                Pred 0 (Normal)   Pred 1 (Alarma)")
    print(f"    Real 0 (Normal)  {tn:<17} {fp}")
    print(f"    Real 1 (Alarma)  {fn:<17} {tp}")

    return {
        "Accuracy": acc, "Precision_macro": prec_macro, "Recall_macro": rec_macro,
        "F1_macro": f1_macro, "Recall_alarm": rec_alarm, "FN": fn, "FP": fp, "AUC": auc_val, "Latencia_ms": lat_ms
    }


def ejecutar_logo(df_dataset, columnas_x, modelo, nombre_mod):
    print(f"\n--- LOGO por Sesiones: {nombre_mod} (Sin Pressure[hPa]) ---")
    print(f"{'Sesión':<8} {'N Total':<10} {'Clase 0':<10} {'Clase 1':<10} {'F1 Macro':<12} {'Recall Alarma':<16} {'FN':<7} {'FP':<7} {'Observación':<20}")
    print("-" * 102)

    X = df_dataset[columnas_x]
    y = df_dataset["Fire Alarm"]
    groups = df_dataset["session_id"]
    logo = LeaveOneGroupOut()

    res_filas = []

    for sess_id, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        modelo.fit(X_tr, y_tr)
        y_pred = modelo.predict(X_te)

        n_total = len(y_te)
        c0 = int((y_te == 0).sum())
        c1 = int((y_te == 1).sum())

        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        clases_unicas = np.unique(y_te)
        if len(clases_unicas) == 1:
            acc = accuracy_score(y_te, y_pred)
            f1_m = np.nan
            rec_alarm = np.nan
            obs = f"Clase única ({clases_unicas[0]}), Acc={acc:.4f}"
        else:
            acc = accuracy_score(y_te, y_pred)
            f1_m = f1_score(y_te, y_pred, average="macro", zero_division=0)
            rec_alarm = recall_score(y_te, y_pred, pos_label=1, zero_division=0)
            obs = "Biclase"

        res_filas.append({
            "sess": sess_id, "total": n_total, "c0": c0, "c1": c1,
            "f1_macro": f1_m, "recall_alarm": rec_alarm, "acc": acc, "fn": fn, "fp": fp, "obs": obs
        })

        f1_str = f"{f1_m:.4f}" if not np.isnan(f1_m) else "N/A"
        rec_str = f"{rec_alarm:.4f}" if not np.isnan(rec_alarm) else "N/A"
        print(f"{sess_id:<8} {n_total:<10} {c0:<10} {c1:<10} {f1_str:<12} {rec_str:<16} {fn:<7} {fp:<7} {obs:<20}")

    print("-" * 102)
    f1_validos = [r["f1_macro"] for r in res_filas if not np.isnan(r["f1_macro"])]
    rec_validos = [r["recall_alarm"] for r in res_filas if not np.isnan(r["recall_alarm"])]
    prom_f1 = np.mean(f1_validos) if f1_validos else np.nan
    prom_rec = np.mean(rec_validos) if rec_validos else np.nan
    prom_fn = np.mean([r["fn"] for r in res_filas])
    prom_fp = np.mean([r["fp"] for r in res_filas])
    prom_acc = np.mean([r["acc"] for r in res_filas])

    print(f"{'PROMEDIO':<40} {prom_f1:>8.4f}     {prom_rec:>12.4f}     {prom_fn:>6.1f} {prom_fp:>6.1f}   Acc prom={prom_acc:.4f}")


def main():
    logger = LoggerTee(RUTA_SALIDA_TXT)
    sys.stdout = logger

    print("=" * 85)
    print("TAREA 2: MODELOS MITIGADOS SIN PRESSURE[hPa] (11_modelo_mitigado.py)")
    print("=" * 85)

    # 1. Carga de datos no escalados
    print("\n1. CARGA DE PARTICIONES Y ELIMINACIÓN DE Pressure[hPa]")
    print("-" * 65)
    X_train = pd.read_csv(DIR_PARTICIONES / "X_train.csv")
    X_val = pd.read_csv(DIR_PARTICIONES / "X_val.csv")
    X_test = pd.read_csv(DIR_PARTICIONES / "X_test.csv")

    y_train = pd.read_csv(DIR_PARTICIONES / "y_train.csv")["Fire Alarm"]
    y_val = pd.read_csv(DIR_PARTICIONES / "y_val.csv")["Fire Alarm"]
    y_test = pd.read_csv(DIR_PARTICIONES / "y_test.csv")["Fire Alarm"]

    vars_sin_presion = [c for c in X_train.columns if c != "Pressure[hPa]"]
    print(f"Variables seleccionadas ({len(vars_sin_presion)}): {vars_sin_presion}")

    X_train_np = X_train[vars_sin_presion]
    X_val_np = X_val[vars_sin_presion]
    X_test_np = X_test[vars_sin_presion]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Árbol de Decisión sin Presión
    print("\n2. ENTRENAMIENTO: ÁRBOL DE DECISIÓN (Sin Pressure[hPa])")
    print("-" * 65)
    modelo_dt = DecisionTreeClassifier(random_state=42)
    parametros_dt = {
        "criterion": ["gini", "entropy"],
        "max_depth": [5, 10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": [None, "balanced"]
    }

    busqueda_dt = RandomizedSearchCV(
        estimator=modelo_dt,
        param_distributions=parametros_dt,
        n_iter=20,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=1
    )

    t0_dt = time.perf_counter()
    busqueda_dt.fit(X_train_np, y_train)
    t_dt = time.perf_counter() - t0_dt

    mejor_dt = busqueda_dt.best_estimator_
    pred_dt_val = mejor_dt.predict(X_val_np)
    f1_dt_val = f1_score(y_val, pred_dt_val, average="macro", zero_division=0)

    print("\nResultados Árbol de Decisión (Sin Presión):")
    print(f"  - Mejores hiperparámetros: {busqueda_dt.best_params_}")
    print(f"  - F1 Macro CV (5-fold)   : {busqueda_dt.best_score_:.4f}")
    print(f"  - F1 Macro en Validación : {f1_dt_val:.4f}")
    print(f"  - Tiempo de entrenamiento: {t_dt:.2f} s")

    # Guardar modelo Árbol sin presión
    ruta_dt_joblib = DIR_MODELOS / "decision_tree_sin_presion.joblib"
    joblib.dump(mejor_dt, ruta_dt_joblib)
    print(f"  [OK] Modelo guardado en: {ruta_dt_joblib}")

    # 3. Random Forest sin Presión
    print("\n3. ENTRENAMIENTO: RANDOM FOREST (Sin Pressure[hPa])")
    print("-" * 65)
    modelo_rf = RandomForestClassifier(random_state=42, n_jobs=1)
    parametros_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"]
    }

    busqueda_rf = RandomizedSearchCV(
        estimator=modelo_rf,
        param_distributions=parametros_rf,
        n_iter=20,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=1
    )

    t0_rf = time.perf_counter()
    busqueda_rf.fit(X_train_np, y_train)
    t_rf = time.perf_counter() - t0_rf

    mejor_rf = busqueda_rf.best_estimator_
    pred_rf_val = mejor_rf.predict(X_val_np)
    f1_rf_val = f1_score(y_val, pred_rf_val, average="macro", zero_division=0)

    print("\nResultados Random Forest (Sin Presión):")
    print(f"  - Mejores hiperparámetros: {busqueda_rf.best_params_}")
    print(f"  - F1 Macro CV (5-fold)   : {busqueda_rf.best_score_:.4f}")
    print(f"  - F1 Macro en Validación : {f1_rf_val:.4f}")
    print(f"  - Tiempo de entrenamiento: {t_rf:.2f} s")

    # Guardar modelo Random Forest sin presión
    ruta_rf_joblib = DIR_MODELOS / "random_forest_sin_presion.joblib"
    joblib.dump(mejor_rf, ruta_rf_joblib)
    print(f"  [OK] Modelo guardado en: {ruta_rf_joblib}")

    # 4. Evaluación en TEST
    print("\n4. EVALUACIÓN COMPARATIVA EN TEST (20 Variables, Sin Presión)")
    print("-" * 65)
    evaluar_en_test("Árbol de Decisión (Sin Presión)", mejor_dt, X_test_np, y_test)
    evaluar_en_test("Random Forest (Sin Presión)", mejor_rf, X_test_np, y_test)

    # 5. Top 8 de Importancia MDI
    print("\n5. TOP 8 DE IMPORTANCIA MDI (Feature Importances) - MODELOS SIN PRESIÓN")
    print("-" * 65)

    df_mdi_dt = pd.DataFrame({
        "Variable": vars_sin_presion,
        "Importancia": mejor_dt.feature_importances_
    }).sort_values(by="Importancia", ascending=False).reset_index(drop=True)
    df_mdi_dt["Acumulada"] = df_mdi_dt["Importancia"].cumsum()

    print("\nTop 8 Importancia MDI - Árbol de Decisión (Sin Presión):")
    print(f"{'Ranking':<8} {'Variable':<22} {'Importancia':<14} {'Acumulada':<12}")
    print("-" * 56)
    for i, row in df_mdi_dt.head(8).iterrows():
        print(f"{i+1:<8} {row['Variable']:<22} {row['Importancia']:>10.4f}     {row['Acumulada']:>10.4f}")

    df_mdi_rf = pd.DataFrame({
        "Variable": vars_sin_presion,
        "Importancia": mejor_rf.feature_importances_
    }).sort_values(by="Importancia", ascending=False).reset_index(drop=True)
    df_mdi_rf["Acumulada"] = df_mdi_rf["Importancia"].cumsum()

    print("\nTop 8 Importancia MDI - Random Forest (Sin Presión):")
    print(f"{'Ranking':<8} {'Variable':<22} {'Importancia':<14} {'Acumulada':<12}")
    print("-" * 56)
    for i, row in df_mdi_rf.head(8).iterrows():
        print(f"{i+1:<8} {row['Variable']:<22} {row['Importancia']:>10.4f}     {row['Acumulada']:>10.4f}")

    # 6. Validación Leave-One-Group-Out (LOGO)
    print("\n6. VALIDACIÓN CRUZADA LEAVE-ONE-GROUP-OUT (LOGO) - MODELOS SIN PRESIÓN")
    print("-" * 65)
    ruta_features = BASE_DIR / "data_procesada" / "smoke_detection_features.csv"
    df_raw = pd.read_csv(ruta_features)
    df_raw["session_id"] = (df_raw["CNT"].diff().fillna(0) < 0).cumsum()

    # Reinstanciar con mejores hiperparámetros
    dt_logo = DecisionTreeClassifier(random_state=42, **busqueda_dt.best_params_)
    rf_logo = RandomForestClassifier(random_state=42, n_jobs=1, **busqueda_rf.best_params_)

    ejecutar_logo(df_raw, vars_sin_presion, dt_logo, "Árbol de Decisión")
    ejecutar_logo(df_raw, vars_sin_presion, rf_logo, "Random Forest")

    print(f"\n  [OK] Salida completa guardada en: {RUTA_SALIDA_TXT}")
    print("\n" + "=" * 85)
    print("EJECUCIÓN DE 11_MODELO_MITIGADO.PY COMPLETADA EXITOSAMENTE")
    print("=" * 85)

    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()
