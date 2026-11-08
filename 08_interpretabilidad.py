import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# ====================================================================
# CONFIGURACIÓN Y CONSTANTES
# ====================================================================
BASE_DIR = Path(__file__).resolve().parent
DIR_DATOS = BASE_DIR / "data_procesada" / "particiones"
DIR_MODELOS = BASE_DIR / "modelos"
DIR_RESULTADOS = BASE_DIR / "resultados_evaluacion" / "interpretabilidad"
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

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

VARS_PROTOTIPO = [
    "Temperature[C]",
    "Humidity[%]",
    "TVOC[ppb]",
    "tvoc_delta",
    "tvoc_mean_5",
    "temp_delta",
    "humidity_delta"
]


def ejecutar_interpretabilidad():
    # ----------------------------------------------------------------
    # 0. CARGA DE MODELOS Y PARTICIONES TEST
    # ----------------------------------------------------------------
    dt = joblib.load(DIR_MODELOS / "decision_tree.joblib")
    rf = joblib.load(DIR_MODELOS / "random_forest.joblib")
    svm = joblib.load(DIR_MODELOS / "svm.joblib")

    X_test = pd.read_csv(DIR_DATOS / "X_test.csv")
    X_test_scaled = pd.read_csv(DIR_DATOS / "X_test_scaled.csv")
    y_test = pd.read_csv(DIR_DATOS / "y_test.csv")["Fire Alarm"]

    # ================================================================
    # 1. IMPORTANCIA MDI DE RANDOM FOREST Y DEL ÁRBOL DE DECISIÓN
    # ================================================================
    print("=" * 85)
    print("1. IMPORTANCIA MDI DE RANDOM FOREST Y DEL ÁRBOL DE DECISIÓN (feature_importances_)")
    print("=" * 85)

    df_mdi = pd.DataFrame({
        "Variable": FEATURE_NAMES,
        "MDI_DT": dt.feature_importances_,
        "MDI_RF": rf.feature_importances_
    })

    # Árbol de Decisión
    df_mdi_dt = df_mdi.sort_values(by="MDI_DT", ascending=False).reset_index(drop=True)
    top5_dt_sum = df_mdi_dt.loc[:4, "MDI_DT"].sum()

    print("\n--- Árbol de Decisión (DecisionTreeClassifier) ---")
    print(f"{'Ranking':<8} {'Variable':<22} {'Importancia MDI':<18}")
    print("-" * 50)
    for idx, row in df_mdi_dt.iterrows():
        print(f"{idx+1:<8} {row['Variable']:<22} {row['MDI_DT']:.4f}")
    print("-" * 50)
    print(f"Suma acumulada Top 5 variables: {top5_dt_sum:.4f}")

    # Random Forest
    df_mdi_rf = df_mdi.sort_values(by="MDI_RF", ascending=False).reset_index(drop=True)
    top5_rf_sum = df_mdi_rf.loc[:4, "MDI_RF"].sum()

    print("\n--- Random Forest (RandomForestClassifier) ---")
    print(f"{'Ranking':<8} {'Variable':<22} {'Importancia MDI':<18}")
    print("-" * 50)
    for idx, row in df_mdi_rf.iterrows():
        print(f"{idx+1:<8} {row['Variable']:<22} {row['MDI_RF']:.4f}")
    print("-" * 50)
    print(f"Suma acumulada Top 5 variables: {top5_rf_sum:.4f}")

    # Gráfico MDI - Árbol de Decisión
    plt.figure(figsize=(9, 6.5))
    df_plot_dt = df_mdi_dt.sort_values(by="MDI_DT", ascending=True)
    plt.barh(df_plot_dt["Variable"], df_plot_dt["MDI_DT"], color="#1f77b4", edgecolor="#0f4c81")
    plt.title("Importancia MDI de Características - Árbol de Decisión", fontsize=12, pad=12)
    plt.xlabel("Importancia MDI (Impureza)", fontsize=10)
    plt.ylabel("Variable", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    ruta_png_dt = DIR_RESULTADOS / "mdi_arbol_decision.png"
    plt.savefig(ruta_png_dt, dpi=150)
    plt.close()

    # Gráfico MDI - Random Forest
    plt.figure(figsize=(9, 6.5))
    df_plot_rf = df_mdi_rf.sort_values(by="MDI_RF", ascending=True)
    plt.barh(df_plot_rf["Variable"], df_plot_rf["MDI_RF"], color="#2ca02c", edgecolor="#1b631b")
    plt.title("Importancia MDI de Características - Random Forest", fontsize=12, pad=12)
    plt.xlabel("Importancia MDI Promedio", fontsize=10)
    plt.ylabel("Variable", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    ruta_png_rf = DIR_RESULTADOS / "mdi_random_forest.png"
    plt.savefig(ruta_png_rf, dpi=150)
    plt.close()

    # ================================================================
    # 2. IMPORTANCIA POR PERMUTACIÓN SOBRE TEST
    # ================================================================
    print("\n" + "=" * 85)
    print("2. IMPORTANCIA POR PERMUTACIÓN SOBRE TEST (permutation_importance, scoring='f1_macro')")
    print("=" * 85)

    perm_dt = permutation_importance(
        dt, X_test, y_test, scoring="f1_macro", n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_rf = permutation_importance(
        rf, X_test, y_test, scoring="f1_macro", n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_svm = permutation_importance(
        svm, X_test_scaled, y_test, scoring="f1_macro", n_repeats=10, random_state=42, n_jobs=-1
    )

    df_perm = pd.DataFrame({
        "Variable": FEATURE_NAMES,
        "DT_mean": perm_dt.importances_mean,
        "DT_std": perm_dt.importances_std,
        "RF_mean": perm_rf.importances_mean,
        "RF_std": perm_rf.importances_std,
        "SVM_mean": perm_svm.importances_mean,
        "SVM_std": perm_svm.importances_std,
    })

    print("\n--- Árbol de Decisión (Permutación en X_test) ---")
    df_p_dt = df_perm.sort_values(by="DT_mean", ascending=False).reset_index(drop=True)
    print(f"{'Ranking':<8} {'Variable':<22} {'Media':<12} {'Desv. Estándar':<15}")
    print("-" * 57)
    for idx, row in df_p_dt.iterrows():
        print(f"{idx+1:<8} {row['Variable']:<22} {row['DT_mean']:>8.4f}     {row['DT_std']:>8.4f}")

    print("\n--- Random Forest (Permutación en X_test) ---")
    df_p_rf = df_perm.sort_values(by="RF_mean", ascending=False).reset_index(drop=True)
    print(f"{'Ranking':<8} {'Variable':<22} {'Media':<12} {'Desv. Estándar':<15}")
    print("-" * 57)
    for idx, row in df_p_rf.iterrows():
        print(f"{idx+1:<8} {row['Variable']:<22} {row['RF_mean']:>8.4f}     {row['RF_std']:>8.4f}")

    print("\n--- Support Vector Machine (Permutación en X_test_scaled) ---")
    df_p_svm = df_perm.sort_values(by="SVM_mean", ascending=False).reset_index(drop=True)
    print(f"{'Ranking':<8} {'Variable':<22} {'Media':<12} {'Desv. Estándar':<15}")
    print("-" * 57)
    for idx, row in df_p_svm.iterrows():
        print(f"{idx+1:<8} {row['Variable']:<22} {row['SVM_mean']:>8.4f}     {row['SVM_std']:>8.4f}")

    # Top 10 variables principales según media promedio entre modelos
    df_perm["Promedio_Global"] = (df_perm["DT_mean"] + df_perm["RF_mean"] + df_perm["SVM_mean"]) / 3.0
    top10_vars = df_perm.sort_values(by="Promedio_Global", ascending=False).head(10).reset_index(drop=True)

    # Gráfico comparativo de las 10 variables principales
    plt.figure(figsize=(11, 7))
    y_pos = np.arange(len(top10_vars))
    height = 0.25

    plt.barh(y_pos + height, top10_vars["DT_mean"], height=height, label="Árbol de Decisión", color="#1f77b4", edgecolor="#0f4c81")
    plt.barh(y_pos, top10_vars["RF_mean"], height=height, label="Random Forest", color="#2ca02c", edgecolor="#1b631b")
    plt.barh(y_pos - height, top10_vars["SVM_mean"], height=height, label="SVM", color="#ff7f0e", edgecolor="#b35504")

    plt.yticks(y_pos, top10_vars["Variable"], fontsize=10)
    plt.gca().invert_yaxis()
    plt.xlabel("Caída de F1-Score Macro tras Permutación", fontsize=10)
    plt.title("Comparación de Importancia por Permutación (Top 10 Variables Principales)", fontsize=12, pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    ruta_png_perm = DIR_RESULTADOS / "permutacion_comparativo_top10.png"
    plt.savefig(ruta_png_perm, dpi=150)
    plt.close()

    # ================================================================
    # 3. REGLAS DEL ÁRBOL DE DECISIÓN
    # ================================================================
    print("\n" + "=" * 85)
    print("3. REGLAS DEL ÁRBOL DE DECISIÓN")
    print("=" * 85)

    reglas_texto = export_text(dt, feature_names=FEATURE_NAMES, decimals=3, show_weights=True)
    print(reglas_texto)

    n_hojas = dt.get_n_leaves()
    prof_real = dt.get_depth()
    tree_features = dt.tree_.feature
    indices_usados = sorted(list(set(tree_features[tree_features != -2])))
    vars_en_nodos = [FEATURE_NAMES[i] for i in indices_usados]

    print(f"Número de hojas en el árbol: {n_hojas}")
    print(f"Profundidad real del árbol  : {prof_real}")
    print(f"Variables que aparecen en algún nodo ({len(vars_en_nodos)}):")
    for v in vars_en_nodos:
        print(f"  - {v}")

    # ================================================================
    # 4. VALIDACIÓN POR SESIÓN (LEAVE-ONE-GROUP-OUT)
    # ================================================================
    print("\n" + "=" * 85)
    print("4. VALIDACIÓN POR SESIÓN (LEAVE-ONE-GROUP-OUT) - TODAS LAS VARIABLES (21 CARACTERÍSTICAS)")
    print("=" * 85)

    ruta_features = BASE_DIR / "data_procesada" / "smoke_detection_features.csv"
    df_raw = pd.read_csv(ruta_features)
    df_raw["session_id"] = (df_raw["CNT"].diff().fillna(0) < 0).cumsum()

    def evaluar_logo(df_dataset, columnas_x, titulo):
        X = df_dataset[columnas_x]
        y = df_dataset["Fire Alarm"]
        groups = df_dataset["session_id"]
        logo = LeaveOneGroupOut()

        modelos = {
            "Árbol de Decisión": DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, min_samples_split=5, min_samples_leaf=2,
                max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
            )
        }

        for nom_mod, model in modelos.items():
            print(f"\n--- {titulo} | Modelo: {nom_mod} ---")
            print(f"{'Sesión':<8} {'N Total':<10} {'Clase 0':<10} {'Clase 1':<10} {'F1 Macro':<12} {'Recall Alarma':<16} {'FN':<7} {'FP':<7} {'Observación':<20}")
            print("-" * 102)

            res_filas = []

            for sess_id, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)

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
                    "sess": sess_id,
                    "total": n_total,
                    "c0": c0,
                    "c1": c1,
                    "f1_macro": f1_m,
                    "recall_alarm": rec_alarm,
                    "acc": acc,
                    "fn": fn,
                    "fp": fp,
                    "obs": obs
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

    evaluar_logo(df_raw, FEATURE_NAMES, "Todas las Variables (21)")

    # ================================================================
    # 5. REPETICIONES DE VALIDACIÓN POR SESIÓN
    # ================================================================
    print("\n" + "=" * 85)
    print("5. VALIDACIÓN POR SESIÓN VARIANDO VARIABLES PREDICTORAS")
    print("=" * 85)

    vars_sin_presion = [f for f in FEATURE_NAMES if f != "Pressure[hPa]"]
    print("\n5.1. VALIDACIÓN POR SESIÓN QUITANDO Pressure[hPa] (20 características)")
    evaluar_logo(df_raw, vars_sin_presion, "Sin Pressure[hPa] (20)")

    print("\n5.2. VALIDACIÓN POR SESIÓN SOLO CON VARIABLES DEL PROTOTIPO (7 características)")
    print(f"Variables utilizadas: {', '.join(VARS_PROTOTIPO)}")
    evaluar_logo(df_raw, VARS_PROTOTIPO, "Solo Variables Prototipo (7)")

    # ================================================================
    # 6. SIMULACIÓN DEL PROTOTIPO FÍSICO
    # ================================================================
    print("\n" + "=" * 85)
    print("6. SIMULACIÓN DEL PROTOTIPO FÍSICO (CALIBRACIÓN ESP32)")
    print("=" * 85)

    raw_values = list(range(1000, 3101, 25))
    sim_rows = []

    for rawAire in raw_values:
        factorHumo = np.clip((rawAire - 1500) / 1600.0, 0.0, 1.0)
        tvoc = 10.0 + factorHumo * 1400.0
        pm25 = 0.05 + factorHumo * 4.2
        pm10 = 0.04 + factorHumo * 3.8
        eco2 = 400.0 + factorHumo * 450.0
        raw_h2 = 12400.0 + factorHumo * 800.0
        raw_ethanol = 18600.0 + factorHumo * 1200.0
        pressure = 939.5
        nc0_5 = 0.1 + factorHumo * 25.0
        nc1_0 = 0.02 + factorHumo * 4.0
        nc2_5 = 0.005 + factorHumo * 0.1
        temperature = 24.0
        humidity = 50.0

        pm_ratio = pm25 / (pm10 + 1e-6)
        pm25_delta = 0.0
        tvoc_delta = 0.0
        eco2_delta = 0.0
        temp_delta = 0.0
        humidity_delta = 0.0
        pm25_mean_5 = pm25
        tvoc_mean_5 = tvoc
        eco2_mean_5 = eco2

        row_feat = [
            temperature, humidity, tvoc, eco2, raw_h2, raw_ethanol, pressure,
            pm10, pm25, nc0_5, nc1_0, nc2_5, pm_ratio, pm25_delta, tvoc_delta,
            eco2_delta, temp_delta, humidity_delta, pm25_mean_5, tvoc_mean_5, eco2_mean_5
        ]
        sim_rows.append(row_feat)

    df_sim = pd.DataFrame(sim_rows, columns=FEATURE_NAMES)
    pred_dt_sim = dt.predict(df_sim)
    pred_rf_sim = rf.predict(df_sim)

    # Imprimir tabla de predicciones en la simulación
    print("\n--- Predicciones de DT y RF ante Variación de Señal rawAire (1000 a 3100) ---")
    print(f"{'rawAire':<9} {'factorHumo':<12} {'TVOC[ppb]':<12} {'PM2.5':<10} {'Pred DT':<10} {'Pred RF':<10}")
    print("-" * 65)
    for raw_val, f_humo, tv, pm, p_dt, p_rf in zip(
        raw_values,
        [(v - 1500) / 1600.0 for v in raw_values],
        df_sim["TVOC[ppb]"],
        df_sim["PM2.5"],
        pred_dt_sim,
        pred_rf_sim
    ):
        f_clip = np.clip(f_humo, 0.0, 1.0)
        # Mostrar pasos representativos y la zona de transición completa
        if raw_val % 100 == 0 or (1650 <= raw_val <= 1800):
            print(f"{raw_val:<9} {f_clip:<12.4f} {tv:<12.4f} {pm:<10.4f} {p_dt:<10} {p_rf:<10}")

    min_raw_dt = None
    min_raw_rf = None

    for raw_val, p_dt, p_rf in zip(raw_values, pred_dt_sim, pred_rf_sim):
        if p_dt == 1 and min_raw_dt is None:
            min_raw_dt = raw_val
        if p_rf == 1 and min_raw_rf is None:
            min_raw_rf = raw_val

    print("\n--- Umbrales Mínimos de Activación ---")
    print(f"Valor mínimo de rawAire que activa la alarma en Árbol de Decisión: {min_raw_dt}")
    print(f"Valor mínimo de rawAire que activa la alarma en Random Forest   : {min_raw_rf}")

    # Detalle de la ruta de decisión del Árbol
    idx_min_dt = raw_values.index(min_raw_dt)
    sample_min = df_sim.iloc[[idx_min_dt]]
    node_indicator = dt.decision_path(sample_min)
    leaf_id = dt.apply(sample_min)[0]

    tree = dt.tree_
    node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    print(f"\nRuta de decisión en el Árbol (decision_path) para rawAire = {min_raw_dt}:")
    for step, node_id in enumerate(node_indices, 1):
        if node_id == leaf_id:
            pred_class = dt.classes_[np.argmax(tree.value[node_id])]
            pesos = tree.value[node_id][0]
            print(f"  Paso {step}: [Nodo {node_id}] Nodo Hoja -> Predicción: Fire Alarm = {pred_class} (pesos entrenamiento: [Normal={pesos[0]:.0f}, Fuego={pesos[1]:.0f}])")
        else:
            feat_idx = tree.feature[node_id]
            feat_name = FEATURE_NAMES[feat_idx]
            val_sample = sample_min[feat_name].values[0]
            thresh = tree.threshold[node_id]
            if val_sample <= thresh:
                decision = f"<= {thresh:.4f} (rama izquierda)"
            else:
                decision = f"> {thresh:.4f} (rama derecha)"
            print(f"  Paso {step}: [Nodo {node_id}] {feat_name} ({val_sample:.4f}) {decision}")


if __name__ == "__main__":
    ejecutar_interpretabilidad()
