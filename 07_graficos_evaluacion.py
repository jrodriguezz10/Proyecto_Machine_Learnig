import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from sklearn.tree import plot_tree


# =========================================================
# 1. RUTAS
# =========================================================

base = Path(__file__).parent

carpeta_datos = (
    base
    / "data_procesada"
    / "particiones"
)

carpeta_modelos = (
    base
    / "modelos"
)

carpeta_graficos = (
    base
    / "resultados_evaluacion"
    / "graficos"
)

carpeta_graficos.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. CARGA DE DATOS
# =========================================================

X_test = pd.read_csv(
    carpeta_datos / "X_test.csv"
)

X_test_scaled = pd.read_csv(
    carpeta_datos / "X_test_scaled.csv"
)

y_test = pd.read_csv(
    carpeta_datos / "y_test.csv"
)["Fire Alarm"]


# =========================================================
# 3. CARGA DE MODELOS
# =========================================================

decision_tree = joblib.load(
    carpeta_modelos / "decision_tree.joblib"
)

random_forest = joblib.load(
    carpeta_modelos / "random_forest.joblib"
)

svm = joblib.load(
    carpeta_modelos / "svm.joblib"
)


# =========================================================
# 4. MATRIZ DE CONFUSIÓN - DECISION TREE
# =========================================================

plt.figure(figsize=(6, 5))

ConfusionMatrixDisplay.from_estimator(
    decision_tree,
    X_test,
    y_test,
    display_labels=[
        "Sin alarma",
        "Alarma"
    ],
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Matriz de Confusión - Decision Tree"
)

plt.tight_layout()

plt.savefig(
    carpeta_graficos
    / "matriz_confusion_decision_tree.png",
    dpi=300
)

plt.close()


# =========================================================
# 5. MATRIZ DE CONFUSIÓN - RANDOM FOREST
# =========================================================

plt.figure(figsize=(6, 5))

ConfusionMatrixDisplay.from_estimator(
    random_forest,
    X_test,
    y_test,
    display_labels=[
        "Sin alarma",
        "Alarma"
    ],
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Matriz de Confusión - Random Forest"
)

plt.tight_layout()

plt.savefig(
    carpeta_graficos
    / "matriz_confusion_random_forest.png",
    dpi=300
)

plt.close()


# =========================================================
# 6. MATRIZ DE CONFUSIÓN - SVM
# =========================================================

plt.figure(figsize=(6, 5))

ConfusionMatrixDisplay.from_estimator(
    svm,
    X_test_scaled,
    y_test,
    display_labels=[
        "Sin alarma",
        "Alarma"
    ],
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Matriz de Confusión - SVM"
)

plt.tight_layout()

plt.savefig(
    carpeta_graficos
    / "matriz_confusion_svm.png",
    dpi=300
)

plt.close()


# =========================================================
# 7. CURVAS ROC COMPARATIVAS
# =========================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

RocCurveDisplay.from_estimator(
    decision_tree,
    X_test,
    y_test,
    name="Decision Tree",
    ax=ax
)

RocCurveDisplay.from_estimator(
    random_forest,
    X_test,
    y_test,
    name="Random Forest",
    ax=ax
)

RocCurveDisplay.from_estimator(
    svm,
    X_test_scaled,
    y_test,
    name="SVM",
    ax=ax
)

ax.set_title(
    "Curvas ROC de los Modelos"
)

plt.tight_layout()

plt.savefig(
    carpeta_graficos
    / "curvas_roc_modelos.png",
    dpi=300
)

plt.close()


# =========================================================
# 8. VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN
# =========================================================

plt.figure(
    figsize=(24, 12)
)

plot_tree(
    decision_tree,
    feature_names=X_test.columns,
    class_names=[
        "Sin alarma",
        "Alarma"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Árbol de Decisión Entrenado"
)

plt.tight_layout()

plt.savefig(
    carpeta_graficos
    / "decision_tree_completo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 9. IMPORTANCIA DE CARACTERÍSTICAS
# =========================================================

importancias = pd.Series(
    decision_tree.feature_importances_,
    index=X_test.columns
).sort_values(
    ascending=True
)

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    importancias.index,
    importancias.values
)

plt.xlabel(
    "Importancia"
)

plt.ylabel(
    "Característica"
)

plt.title(
    "Importancia de Características - Decision Tree"
)

plt.tight_layout()

plt.savefig(
    carpeta_graficos
    / "importancia_caracteristicas_decision_tree.png",
    dpi=300
)

plt.close()


# =========================================================
# 10. RESULTADO
# =========================================================

print("=" * 70)
print("GRÁFICOS GENERADOS CORRECTAMENTE")
print("=" * 70)

print("\nArchivos generados:")

print(
    "- matriz_confusion_decision_tree.png"
)

print(
    "- matriz_confusion_random_forest.png"
)

print(
    "- matriz_confusion_svm.png"
)

print(
    "- curvas_roc_modelos.png"
)

print(
    "- decision_tree_completo.png"
)

print(
    "- importancia_caracteristicas_decision_tree.png"
)

print("\nGuardados en:")
print(carpeta_graficos)