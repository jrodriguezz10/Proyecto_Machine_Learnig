import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


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

carpeta_resultados = (
    base
    / "resultados_evaluacion"
)

carpeta_resultados.mkdir(exist_ok=True)


# =========================================================
# 2. CARGA DE DATOS TEST
# =========================================================

X_train = pd.read_csv(
    carpeta_datos / "X_train.csv"
)

y_train = pd.read_csv(
    carpeta_datos / "y_train.csv"
)["Fire Alarm"]

X_test = pd.read_csv(
    carpeta_datos / "X_test.csv"
)

X_test_scaled = pd.read_csv(
    carpeta_datos / "X_test_scaled.csv"
)

y_test = pd.read_csv(
    carpeta_datos / "y_test.csv"
)["Fire Alarm"]


print("=" * 75)
print("1. CONJUNTO DE PRUEBA")
print("=" * 75)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nDistribución de clases:")
print(y_test.value_counts().sort_index())


# =========================================================
# 3. CARGA DE MODELOS ENTRENADOS
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
# 4. BENCHMARK - DUMMY CLASSIFIER
# =========================================================

dummy = DummyClassifier(
    strategy="most_frequent"
)

dummy.fit(
    X_train,
    y_train
)


# =========================================================
# 5. PREDICCIONES
# =========================================================

pred_dummy = dummy.predict(X_test)

pred_dt = decision_tree.predict(X_test)

pred_rf = random_forest.predict(X_test)

pred_svm = svm.predict(X_test_scaled)


# =========================================================
# 6. SCORES PARA ROC-AUC
# =========================================================

score_dummy = dummy.predict_proba(X_test)[:, 1]

score_dt = decision_tree.predict_proba(X_test)[:, 1]

score_rf = random_forest.predict_proba(X_test)[:, 1]

# SVM fue entrenado sin probability=True.
# Para AUC se utiliza decision_function.
score_svm = svm.decision_function(
    X_test_scaled
)


# =========================================================
# 7. FUNCIÓN DE EVALUACIÓN
# =========================================================

def evaluar_modelo(
    nombre,
    y_real,
    y_pred,
    y_score
):
    return {
        "Modelo": nombre,

        "Accuracy": accuracy_score(
            y_real,
            y_pred
        ),

        "Precision_macro": precision_score(
            y_real,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Recall_macro": recall_score(
            y_real,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "F1_macro": f1_score(
            y_real,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_real,
            y_score
        )
    }


# =========================================================
# 8. TABLA DE RESULTADOS
# =========================================================

resultados = pd.DataFrame([
    evaluar_modelo(
        "Dummy Classifier",
        y_test,
        pred_dummy,
        score_dummy
    ),

    evaluar_modelo(
        "Decision Tree",
        y_test,
        pred_dt,
        score_dt
    ),

    evaluar_modelo(
        "Random Forest",
        y_test,
        pred_rf,
        score_rf
    ),

    evaluar_modelo(
        "SVM",
        y_test,
        pred_svm,
        score_svm
    )
])


print("\n" + "=" * 75)
print("2. RESULTADOS FINALES SOBRE TEST")
print("=" * 75)

print(
    resultados
    .round(4)
    .to_string(index=False)
)


# =========================================================
# 9. MATRICES DE CONFUSIÓN
# =========================================================

modelos_pred = {
    "Decision Tree": pred_dt,
    "Random Forest": pred_rf,
    "SVM": pred_svm
}

print("\n" + "=" * 75)
print("3. MATRICES DE CONFUSIÓN")
print("=" * 75)

for nombre, pred in modelos_pred.items():

    matriz = confusion_matrix(
        y_test,
        pred
    )

    print(f"\n{nombre}")
    print(matriz)


# =========================================================
# 10. REPORTES DE CLASIFICACIÓN
# =========================================================

print("\n" + "=" * 75)
print("4. REPORTES DE CLASIFICACIÓN")
print("=" * 75)

for nombre, pred in modelos_pred.items():

    print("\n" + nombre)
    print("-" * 50)

    print(
        classification_report(
            y_test,
            pred,
            digits=4
        )
    )


# =========================================================
# 11. CONFIRMACIÓN DEL MODELO SELECCIONADO
# =========================================================

print("\n" + "=" * 75)
print("5. MODELO SELECCIONADO EN VALIDACIÓN")
print("=" * 75)

fila_dt = resultados[
    resultados["Modelo"] == "Decision Tree"
].iloc[0]

print("Modelo: Decision Tree")
print(
    "Accuracy:",
    round(fila_dt["Accuracy"], 4)
)
print(
    "Precision Macro:",
    round(fila_dt["Precision_macro"], 4)
)
print(
    "Recall Macro:",
    round(fila_dt["Recall_macro"], 4)
)
print(
    "F1 Macro:",
    round(fila_dt["F1_macro"], 4)
)
print(
    "ROC-AUC:",
    round(fila_dt["ROC_AUC"], 4)
)


# =========================================================
# 12. GUARDAR RESULTADOS
# =========================================================

resultados.to_csv(
    carpeta_resultados
    / "metricas_test.csv",
    index=False
)

for nombre, pred in modelos_pred.items():

    matriz = confusion_matrix(
        y_test,
        pred
    )

    nombre_archivo = (
        nombre.lower()
        .replace(" ", "_")
        + "_confusion_matrix.csv"
    )

    pd.DataFrame(
        matriz,
        index=["Real_0", "Real_1"],
        columns=["Pred_0", "Pred_1"]
    ).to_csv(
        carpeta_resultados
        / nombre_archivo
    )


print("\n" + "=" * 75)
print("6. EVALUACIÓN FINALIZADA")
print("=" * 75)

print(
    "Resultados guardados en:"
)

print(carpeta_resultados)

print(
    "\nEl conjunto TEST fue utilizado "
    "únicamente en esta etapa final."
)