# TP4 · Entrenamiento y serialización del modelo de riesgo de ACV
#
# Este script entrena UNA vez el modelo final del TP3 (regresión logística con pesos de clase),
# los otros dos modelos de comparación, el clustering (K-medias, k = 3) y las tablas que usa la app,
# y lo guarda todo en un único archivo: modelo_acv.joblib
#
# La app (app_tp3.py) carga ese archivo y NO reentrena.
#
# Uso:   python entrenar_modelo.py        (el CSV tiene que estar en la misma carpeta)
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

CARPETA = Path(__file__).resolve().parent
CSV = CARPETA / "healthcare-dataset-stroke-data.csv"
SALIDA = CARPETA / "modelo_acv.joblib"

NUMERICAS = ["age", "avg_glucose_level", "bmi"]
BINARIAS = ["hypertension", "heart_disease"]
CATEGORICAS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
COLUMNAS = NUMERICAS + BINARIAS + CATEGORICAS
SEMILLA = 42


def preparar():
    """Preprocesamiento: mediana para faltantes, estandarizar números, One-Hot para texto."""
    return make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERICAS),
        ("passthrough", BINARIAS),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    )


def construir():
    """Entrena todo y devuelve un diccionario con lo que necesita la app."""
    d = pd.read_csv(CSV).drop(columns="id")
    d = d[d["gender"] != "Other"]                      # un solo registro: no alcanza para aprender
    X, y = d[COLUMNAS], d["stroke"]
    X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEMILLA)

    modelos = {
        "Regresión logística": make_pipeline(preparar(), LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000)),
        "Árbol de decisión": make_pipeline(preparar(), DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=SEMILLA)),
        "Bosque aleatorio": make_pipeline(preparar(), RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced_subsample", random_state=SEMILLA, n_jobs=-1)),
    }
    for m in modelos.values():
        m.fit(X_ent, y_ent)
    p_pru = {n: m.predict_proba(X_pru)[:, 1] for n, m in modelos.items()}
    # Paciente "promedio" (en el espacio ya transformado): sirve para explicar cuánto aporta cada variable
    x_medio = np.asarray(modelos["Regresión logística"][0].transform(X_ent)).mean(axis=0)

    # Umbral que marca al 14 % de mayor riesgo, elegido SOLO con el entrenamiento (validación cruzada)
    p_cv = cross_val_predict(modelos["Regresión logística"], X_ent, y_ent,
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA), method="predict_proba")[:, 1]
    u14 = float(np.quantile(p_cv, 0.86))

    # Importancia por permutación del bosque, sobre la prueba, medida en AUC
    pi = permutation_importance(modelos["Bosque aleatorio"], X_pru, y_pru, scoring="roc_auc", n_repeats=5, random_state=SEMILLA, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_pru.columns).sort_values()
    imp_sd = pd.Series(pi.importances_std, index=X_pru.columns)[imp.index]

    # Clustering: K-medias con k = 3 sin mirar stroke; los grupos se ordenan por edad media
    pre_cl = preparar().fit(X_ent)
    Z = np.asarray(pre_cl.transform(X_ent))
    km = KMeans(n_clusters=3, n_init=10, random_state=SEMILLA).fit(Z)
    orden = pd.Series(X_ent["age"].values).groupby(km.labels_).mean().sort_values().index.tolist()
    remapa = {int(viejo): nuevo for nuevo, viejo in enumerate(orden)}
    g_ent = np.array([remapa[int(g)] for g in km.labels_])
    tasa_g = pd.Series(y_ent.values).groupby(g_ent).mean().to_dict()
    tam_g = pd.Series(g_ent).value_counts().sort_index().to_dict()
    Z2 = PCA(2, random_state=0).fit_transform(Z)

    return {
        "versiones": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "columnas": COLUMNAS,
        "modelos": modelos,
        "p_pru": p_pru,
        "y_pru": y_pru.values,
        "u14": u14,
        "x_medio": x_medio,
        "imp": imp,
        "imp_sd": imp_sd,
        "corr": d[["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "stroke"]].corr(),
        "rangos": {c: (float(d[c].min()), float(d[c].max())) for c in NUMERICAS},
        "n_pacientes": int(len(d)),
        "n_acv": int(d["stroke"].sum()),
        "cluster": {"pre": pre_cl, "km": km, "remapa": remapa, "tasa": tasa_g, "tam": tam_g, "Z2": Z2, "g": g_ent, "y": y_ent.values},
    }


if __name__ == "__main__":
    paquete = construir()
    joblib.dump(paquete, SALIDA, compress=3)
    print(f"Modelo guardado en {SALIDA.name} ({SALIDA.stat().st_size / 1e6:.1f} MB)")
    print("Umbral del 14 %:", round(paquete["u14"], 3), "· versiones:", paquete["versiones"])
