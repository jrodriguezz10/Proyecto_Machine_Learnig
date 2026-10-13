import pandas as pd
from pathlib import Path

# =========================================================
# 1. RUTAS
# =========================================================

base = Path(__file__).parent

carpeta = (
    base
    / "data_procesada"
    / "particiones"
)

# =========================================================
# 2. CARGA DE ETIQUETAS
# =========================================================

y_train = pd.read_csv(carpeta / "y_train.csv")
y_val = pd.read_csv(carpeta / "y_val.csv")
y_test = pd.read_csv(carpeta / "y_test.csv")

# =========================================================
# 3. TAMAÑO DE LOS CONJUNTOS
# =========================================================

print("=" * 65)
print("1. TAMAÑO DE LAS PARTICIONES")
print("=" * 65)

print("Entrenamiento:", len(y_train))
print("Validación   :", len(y_val))
print("Prueba       :", len(y_test))

print("\nTotal:", len(y_train) + len(y_val) + len(y_test))


# =========================================================
# 4. DISTRIBUCIÓN DE CLASES
# =========================================================

print("\n" + "=" * 65)
print("2. DISTRIBUCIÓN DE CLASES")
print("=" * 65)

for nombre, datos in [
    ("TRAIN", y_train),
    ("VALIDATION", y_val),
    ("TEST", y_test)
]:
    serie = datos["Fire Alarm"]

    conteo = serie.value_counts().sort_index()
    porcentaje = (
        serie.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print(f"\n{nombre}")
    print("-" * 30)

    for clase in [0, 1]:
        print(
            f"Clase {clase}: "
            f"{conteo.get(clase, 0)} registros "
            f"({porcentaje.get(clase, 0)} %)"
        )


# =========================================================
# 5. RESUMEN FINAL
# =========================================================

print("\n" + "=" * 65)
print("3. RESUMEN DE PARTICIÓN")
print("=" * 65)

print("Train      : 70 %")
print("Validation : 15 %")
print("Test       : 15 %")

print("\nMétodo:")
print("- División estratificada")
print("- random_state = 42")
print("- Problema de clasificación binaria")

print("\nEl conjunto TEST permanecerá reservado")
print("para la evaluación final del modelo.")