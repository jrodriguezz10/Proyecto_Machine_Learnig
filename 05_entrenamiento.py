import pandas as pd
import joblib
import time
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# 1. RUTAS DEL PROYECTO
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

carpeta_modelos.mkdir(exist_ok=True)


# =========================================================
# 2. CARGA DE DATOS
# =========================================================

# Datos originales para Árbol de Decisión y Random Forest
X_train = pd.read_csv(carpeta_datos / "X_train.csv")
X_val = pd.read_csv(carpeta_datos / "X_val.csv")

# Datos estandarizados para SVM
X_train_scaled = pd.read_csv(
    carpeta_datos / "X_train_scaled.csv"
)

X_val_scaled = pd.read_csv(
    carpeta_datos / "X_val_scaled.csv"
)

# Etiquetas
y_train = pd.read_csv(
    carpeta_datos / "y_train.csv"
)["Fire Alarm"]

y_val = pd.read_csv(
    carpeta_datos / "y_val.csv"
)["Fire Alarm"]


print("=" * 72)
print("1. DATOS PARA ENTRENAMIENTO")
print("=" * 72)

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)

print("\nDistribución de clases en entrenamiento:")
print(y_train.value_counts().sort_index())


# =========================================================
# 3. VALIDACIÓN CRUZADA
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n" + "=" * 72)
print("2. CONFIGURACIÓN DE VALIDACIÓN CRUZADA")
print("=" * 72)

print("Método: StratifiedKFold")
print("Número de folds: 5")
print("Shuffle: True")
print("random_state: 42")
print("Métrica de optimización: F1 Macro")


# =========================================================
# 4. ÁRBOL DE DECISIÓN
# =========================================================

print("\n" + "=" * 72)
print("3. ENTRENAMIENTO - ÁRBOL DE DECISIÓN")
print("=" * 72)

modelo_dt = DecisionTreeClassifier(
    random_state=42
)

parametros_dt = {
    "criterion": [
        "gini",
        "entropy"
    ],
    "max_depth": [
        5,
        10,
        20,
        30,
        None
    ],
    "min_samples_split": [
        2,
        5,
        10
    ],
    "min_samples_leaf": [
        1,
        2,
        5
    ],
    "class_weight": [
        None,
        "balanced"
    ]
}

busqueda_dt = RandomizedSearchCV(
    estimator=modelo_dt,
    param_distributions=parametros_dt,
    n_iter=20,
    scoring="f1_macro",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

inicio = time.perf_counter()

busqueda_dt.fit(
    X_train,
    y_train
)

tiempo_dt = time.perf_counter() - inicio

mejor_dt = busqueda_dt.best_estimator_

pred_dt = mejor_dt.predict(X_val)

print("\nMejores hiperparámetros:")
print(busqueda_dt.best_params_)

print(
    "Mejor F1 Macro CV:",
    round(busqueda_dt.best_score_, 4)
)

print(
    "F1 Macro validación:",
    round(
        f1_score(
            y_val,
            pred_dt,
            average="macro"
        ),
        4
    )
)

print(
    "Tiempo de entrenamiento:",
    round(tiempo_dt, 2),
    "segundos"
)


# =========================================================
# 5. RANDOM FOREST
# =========================================================

print("\n" + "=" * 72)
print("4. ENTRENAMIENTO - RANDOM FOREST")
print("=" * 72)

modelo_rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

parametros_rf = {
    "n_estimators": [
        100,
        200,
        300
    ],
    "max_depth": [
        10,
        20,
        30,
        None
    ],
    "min_samples_split": [
        2,
        5,
        10
    ],
    "min_samples_leaf": [
        1,
        2,
        4
    ],
    "max_features": [
        "sqrt",
        "log2"
    ],
    "class_weight": [
        None,
        "balanced"
    ]
}

busqueda_rf = RandomizedSearchCV(
    estimator=modelo_rf,
    param_distributions=parametros_rf,
    n_iter=20,
    scoring="f1_macro",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

inicio = time.perf_counter()

busqueda_rf.fit(
    X_train,
    y_train
)

tiempo_rf = time.perf_counter() - inicio

mejor_rf = busqueda_rf.best_estimator_

pred_rf = mejor_rf.predict(X_val)

print("\nMejores hiperparámetros:")
print(busqueda_rf.best_params_)

print(
    "Mejor F1 Macro CV:",
    round(busqueda_rf.best_score_, 4)
)

print(
    "F1 Macro validación:",
    round(
        f1_score(
            y_val,
            pred_rf,
            average="macro"
        ),
        4
    )
)

print(
    "Tiempo de entrenamiento:",
    round(tiempo_rf, 2),
    "segundos"
)


# =========================================================
# 6. SVM
# =========================================================

print("\n" + "=" * 72)
print("5. ENTRENAMIENTO - SVM")
print("=" * 72)

modelo_svm = SVC(
    kernel="rbf",
    cache_size=1024
)

parametros_svm = {
    "C": [
        0.1,
        0.5,
        1,
        5,
        10
    ],
    "gamma": [
        "scale",
        "auto",
        0.01,
        0.1
    ],
    "class_weight": [
        None,
        "balanced"
    ]
}

# Para SVM usamos una búsqueda más reducida
# debido a su mayor costo computacional.
busqueda_svm = RandomizedSearchCV(
    estimator=modelo_svm,
    param_distributions=parametros_svm,
    n_iter=10,
    scoring="f1_macro",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

inicio = time.perf_counter()

busqueda_svm.fit(
    X_train_scaled,
    y_train
)

tiempo_svm = time.perf_counter() - inicio

mejor_svm = busqueda_svm.best_estimator_

pred_svm = mejor_svm.predict(
    X_val_scaled
)

print("\nMejores hiperparámetros:")
print(busqueda_svm.best_params_)

print(
    "Mejor F1 Macro CV:",
    round(busqueda_svm.best_score_, 4)
)

print(
    "F1 Macro validación:",
    round(
        f1_score(
            y_val,
            pred_svm,
            average="macro"
        ),
        4
    )
)

print(
    "Tiempo de entrenamiento:",
    round(tiempo_svm, 2),
    "segundos"
)


# =========================================================
# 7. MÉTRICAS DE VALIDACIÓN
# =========================================================

def calcular_metricas(
    nombre,
    y_real,
    y_pred,
    cv_score,
    tiempo
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
        "F1_CV": cv_score,
        "Tiempo_segundos": tiempo
    }


resultados = pd.DataFrame([
    calcular_metricas(
        "Decision Tree",
        y_val,
        pred_dt,
        busqueda_dt.best_score_,
        tiempo_dt
    ),
    calcular_metricas(
        "Random Forest",
        y_val,
        pred_rf,
        busqueda_rf.best_score_,
        tiempo_rf
    ),
    calcular_metricas(
        "SVM",
        y_val,
        pred_svm,
        busqueda_svm.best_score_,
        tiempo_svm
    )
])


print("\n" + "=" * 72)
print("6. COMPARACIÓN SOBRE VALIDACIÓN")
print("=" * 72)

print(
    resultados.round(4)
    .to_string(index=False)
)


# =========================================================
# 8. MODELO CON MEJOR F1 DE VALIDACIÓN
# =========================================================

indice_mejor = resultados[
    "F1_macro"
].idxmax()

nombre_mejor = resultados.loc[
    indice_mejor,
    "Modelo"
]

print("\n" + "=" * 72)
print("7. MODELO CON MEJOR RESULTADO DE VALIDACIÓN")
print("=" * 72)

print(
    "Modelo:",
    nombre_mejor
)

print(
    "F1 Macro:",
    round(
        resultados.loc[
            indice_mejor,
            "F1_macro"
        ],
        4
    )
)


# =========================================================
# 9. GUARDAR MODELOS
# =========================================================

joblib.dump(
    mejor_dt,
    carpeta_modelos / "decision_tree.joblib"
)

joblib.dump(
    mejor_rf,
    carpeta_modelos / "random_forest.joblib"
)

joblib.dump(
    mejor_svm,
    carpeta_modelos / "svm.joblib"
)

resultados.to_csv(
    carpeta_modelos
    / "resultados_validacion.csv",
    index=False
)


if nombre_mejor == "Decision Tree":
    mejor_modelo = mejor_dt

elif nombre_mejor == "Random Forest":
    mejor_modelo = mejor_rf

else:
    mejor_modelo = mejor_svm


joblib.dump(
    mejor_modelo,
    carpeta_modelos
    / "modelo_mejor_validacion.joblib"
)


# =========================================================
# 10. RESUMEN FINAL
# =========================================================

print("\n" + "=" * 72)
print("8. ENTRENAMIENTO FINALIZADO")
print("=" * 72)

print("Modelos entrenados:")
print("- Decision Tree")
print("- Random Forest")
print("- SVM")

print("\nTécnica de optimización:")
print("- RandomizedSearchCV")

print("\nValidación cruzada:")
print("- Estratificada")

print("\nMétrica principal:")
print("- F1 Macro")

print("\nIMPORTANTE:")
print(
    "El conjunto TEST NO fue utilizado "
    "durante esta etapa."
)

print("\nModelos guardados en:")
print(carpeta_modelos)