# TP3 · Clasificación · aplicación Streamlit
# Dice, para un paciente nuevo, qué riesgo de ACV estima cada modelo y a qué grupo de pacientes se parece (clustering).
# Para abrirla:  streamlit run app_tp3.py      (el CSV tiene que estar en la misma carpeta)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score

st.set_page_config(page_title="TP3 · Riesgo de ACV", layout="wide")
NUMERICAS = ["age", "avg_glucose_level", "bmi"]
BINARIAS = ["hypertension", "heart_disease"]
CATEGORICAS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
COLUMNAS = NUMERICAS + BINARIAS + CATEGORICAS
NOMBRES_G = {0: "jóvenes sin factores", 1: "adultos con glucosa normal", 2: "adultos con glucosa alta"}


@st.cache_resource
def entrenar():
    d = pd.read_csv("healthcare-dataset-stroke-data.csv").drop(columns="id")
    d = d[d["gender"] != "Other"]
    X, y = d[COLUMNAS], d["stroke"]
    X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    def prep():
        return make_column_transformer(
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERICAS),
            ("passthrough", BINARIAS),
            (OneHotEncoder(handle_unknown="ignore"), CATEGORICAS))

    modelos = {
        "Regresión logística": make_pipeline(prep(), LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000)),
        "Árbol de decisión": make_pipeline(prep(), DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=42)),
        "Bosque aleatorio": make_pipeline(prep(), RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    }
    for m in modelos.values():
        m.fit(X_ent, y_ent)
    p_pru = {n: m.predict_proba(X_pru)[:, 1] for n, m in modelos.items()}

    # clustering: k-medias con k = 3 sobre las mismas variables, sin mirar stroke; los grupos se ordenan por edad media
    pre_cl = prep().fit(X_ent)
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(np.asarray(pre_cl.transform(X_ent)))
    orden = pd.Series(X_ent["age"].values).groupby(km.labels_).mean().sort_values().index.tolist()
    remapa = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
    g_ent = np.array([remapa[g] for g in km.labels_])
    tasa_g = pd.Series(y_ent.values).groupby(g_ent).mean().to_dict()

    # --- extras para las pestañas ---
    ex = {"datos": d}
    # umbral del 14 %: se elige con validación cruzada sobre el entrenamiento
    p_cv = cross_val_predict(modelos["Regresión logística"], X_ent, y_ent, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    ex["u14"] = float(np.quantile(p_cv, 0.86))
    # importancia por permutación del bosque, sobre prueba, medida en AUC
    pi = permutation_importance(modelos["Bosque aleatorio"], X_pru, y_pru, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
    ex["imp"] = pd.Series(pi.importances_mean, index=X_pru.columns).sort_values()
    ex["imp_sd"] = pd.Series(pi.importances_std, index=X_pru.columns)[ex["imp"].index]
    # clustering en 2D (PCA) y tasas por grupo
    Z = np.asarray(pre_cl.transform(X_ent)); pca = PCA(2, random_state=0).fit(Z)
    ex["pca"] = pca; ex["Z2"] = pca.transform(Z); ex["g"] = g_ent; ex["y_ent"] = y_ent.values
    ex["nuevos"] = pd.DataFrame([
        ("A · mujer joven, sin antecedentes", "Female", 28, 0, 0, "No", "Private", "Urban", 85.0, 22.0, "never smoked"),
        ("B · varón de 45, sin antecedentes", "Male", 45, 0, 0, "Yes", "Private", "Urban", 100.0, 27.0, "formerly smoked"),
        ("C · mujer de 62, hipertensa", "Female", 62, 1, 0, "Yes", "Self-employed", "Rural", 130.0, 30.0, "never smoked"),
        ("D · varón de 72, hipertenso y cardiópata", "Male", 72, 1, 1, "Yes", "Private", "Urban", 210.0, 31.0, "formerly smoked"),
        ("E · mujer de 80, cardiópata", "Female", 80, 0, 1, "Yes", "Self-employed", "Rural", 105.0, 25.0, "never smoked"),
        ("F · varón de 58, hipertenso, glucosa alta", "Male", 58, 1, 0, "Yes", "Private", "Urban", 230.0, 35.0, "smokes"),
        ("G · niña de 8 años", "Female", 8, 0, 0, "No", "children", "Urban", 80.0, 17.0, "Unknown"),
        ("H · varón de 67, sin otros factores", "Male", 67, 0, 0, "Yes", "Private", "Rural", 95.0, 24.0, "never smoked")],
        columns=["perfil", "gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]).set_index("perfil")[COLUMNAS]
    return modelos, X_pru, y_pru, p_pru, pre_cl, km, remapa, tasa_g, ex


modelos, X_pru, y_pru, p_pru, pre_cl, km, remapa, tasa_g, ex = entrenar()
st.title("¿Qué riesgo de ACV estima el modelo para este paciente?")
st.caption("Clasificación (TP3). Los modelos usan class_weight balanceado por el desbalance (solo ~5 % de ACV): las probabilidades están infladas a propósito, vale el orden y el veredicto según el umbral. Es una demostración, no un diagnóstico.")

with st.sidebar:
    st.header("Paciente")
    edad = st.slider("Edad (años)", 0, 82, 62)
    glucosa = st.slider("Glucosa promedio (mg/dL)", 55, 272, 130)
    bmi = st.slider("IMC", 10.0, 60.0, 28.0, step=0.5)
    hta = st.checkbox("Hipertensión", value=True)
    card = st.checkbox("Cardiopatía")
    genero = st.selectbox("Género", ["Female", "Male"])
    casado = st.selectbox("Alguna vez casado/a", ["Yes", "No"])
    trabajo = st.selectbox("Tipo de trabajo", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    zona = st.selectbox("Zona de residencia", ["Urban", "Rural"])
    fuma = st.selectbox("Tabaquismo", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    st.header("Decisión")
    umbral = st.slider("Umbral: arriba de esta probabilidad, decir «ACV»", 0.05, 0.95, 0.50, step=0.05)
    elegido = st.selectbox("Modelo para el gráfico", list(modelos), index=0)

paciente = pd.DataFrame([{"age": edad, "avg_glucose_level": glucosa, "bmi": bmi, "hypertension": int(hta), "heart_disease": int(card),
                          "gender": genero, "ever_married": casado, "work_type": trabajo, "Residence_type": zona, "smoking_status": fuma}])[COLUMNAS]
prob = {n: float(m.predict_proba(paciente)[0, 1]) for n, m in modelos.items()}
percentil = {n: float((p_pru[n] < prob[n]).mean() * 100) for n in modelos}
grupo = remapa[int(km.predict(np.asarray(pre_cl.transform(paciente)))[0])]

tab_p, tab_c, tab_g, tab_i, tab_n, tab_k = st.tabs(["Paciente", "Correlación", "Curva de ganancia", "Importancia", "Pacientes nuevos (A a H)", "Clustering"])
with tab_p:
    st.subheader("Qué responde cada modelo")
    cols = st.columns(3)
    for c, n in zip(cols, modelos):
        veredicto = "ACV" if prob[n] >= umbral else "sin ACV"
        c.metric(n, veredicto, f"prob. {prob[n]:.2f} · percentil {percentil[n]:.0f} %", delta_color="off")
    st.caption("El percentil dice a cuántos pacientes de prueba supera este paciente en riesgo. Si los modelos no coinciden, es un caso de la zona gris.")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        nombres = list(modelos); vals = [prob[n] for n in nombres]
        ax.bar(nombres, vals, color=["#0055a5", "#f0a800", "#2e8b57"]); ax.axhline(umbral, color="#c0392b", ls="--", label=f"umbral {umbral:.2f}")
        for i, v in enumerate(vals): ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
        ax.set_ylim(0, 1.05); ax.set_ylabel("probabilidad de ACV"); ax.legend(); ax.set_title("Probabilidad de cada modelo"); ax.tick_params(axis="x", labelsize=9)
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6.2, 3.8))
        ax2.hist(p_pru[elegido][y_pru.values == 0], bins=30, alpha=0.6, color="#9ca3af", label="sin ACV (prueba)", density=True)
        ax2.hist(p_pru[elegido][y_pru.values == 1], bins=30, alpha=0.7, color="#c0392b", label="con ACV (prueba)", density=True)
        ax2.axvline(prob[elegido], color="black", lw=2, label="este paciente"); ax2.axvline(umbral, color="#c0392b", ls="--", lw=1)
        ax2.set_xlabel(f"probabilidad según {elegido}"); ax2.set_ylabel("densidad"); ax2.legend(fontsize=8); ax2.set_title("Dónde cae entre los pacientes de prueba")
        st.pyplot(fig2)

    st.subheader("¿A qué grupo se parece? (clustering, sin mirar el ACV)")
    st.write(f"K-medias con 3 grupos lo ubica en el **grupo {grupo}: {NOMBRES_G[grupo]}**, donde el {tasa_g[grupo] * 100:.1f} % de los pacientes de entrenamiento tuvo un ACV "
             f"(grupos: jóvenes {tasa_g[0] * 100:.1f} %, adultos con glucosa normal {tasa_g[1] * 100:.1f} %, adultos con glucosa alta {tasa_g[2] * 100:.1f} %).")

    st.subheader(f"Cómo rinde cada modelo en los pacientes de prueba (umbral {umbral:.2f})")
    filas = []
    for n in modelos:
        pred = (p_pru[n] >= umbral).astype(int)
        filas.append({"modelo": n, "AUC": roc_auc_score(y_pru, p_pru[n]), "recall (ACV detectados)": recall_score(y_pru, pred),
                      "precisión": precision_score(y_pru, pred, zero_division=0), "pacientes marcados como ACV": int(pred.sum())})
    st.dataframe(pd.DataFrame(filas).set_index("modelo").round(3))
    st.caption("AUC: qué tan bien ordena por riesgo (0,5 = azar, 1 = perfecto). Recall: de los ACV reales, cuántos detecta. Precisión: de los que marca, cuántos son ACV. Subir el umbral baja el recall y sube la precisión. Los valores pueden diferir un poco de los del notebook porque acá los hiperparámetros están fijados.")


with tab_c:
    st.subheader("Correlación: ninguna variable numérica se relaciona fuerte con el ACV")
    dd = ex["datos"]; cols_c = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "stroke"]
    cm = dd[cols_c].corr()
    fig, a = plt.subplots(figsize=(7.5, 5.2))
    a.imshow(cm.values, cmap="Blues", vmin=0, vmax=1)
    a.set_xticks(range(6)); a.set_xticklabels(cols_c, rotation=40, ha="right"); a.set_yticks(range(6)); a.set_yticklabels(cols_c)
    for i in range(6):
        for j in range(6):
            a.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if cm.values[i, j] > 0.5 else "black")
    plt.tight_layout(); st.pyplot(fig)
    st.write("La fila de **stroke**: la edad es la que más se relaciona (%.2f), después hipertensión, cardiopatía y glucosa (alrededor de 0,13 a 0,14). Son valores bajos: por eso el problema es difícil y solo el ~5 %% de los pacientes tiene ACV." % cm.loc["stroke", "age"])

with tab_g:
    st.subheader("Curva de ganancia: ¿cuántos ACV encontramos llamando al X % de mayor riesgo?")
    fig, a = plt.subplots(figsize=(8.5, 5))
    for n, c in zip(modelos, ["#0055a5", "#f0a800", "#2e8b57"]):
        orden = np.argsort(-p_pru[n]); acum = np.cumsum(y_pru.values[orden]) / y_pru.sum()
        a.plot(np.arange(1, len(acum) + 1) / len(acum) * 100, acum * 100, label=n, color=c, lw=2)
    a.plot([0, 100], [0, 100], color="gray", ls="--", label="al azar")
    a.axvline(14, color="#c0392b", ls=":"); a.set_xlabel("% de pacientes llamados (de mayor a menor riesgo)"); a.set_ylabel("% de ACV encontrados"); a.legend(); a.grid(alpha=0.3)
    st.pyplot(fig)
    pp = p_pru["Regresión logística"]; o = np.argsort(-pp); k = int(round(0.14 * len(pp)))
    st.write(f"Llamando al **14 %** de los pacientes con más riesgo, la regresión logística encuentra el **{y_pru.values[o][:k].sum() / y_pru.sum() * 100:.0f} %** de los ACV (al azar encontraría el 14 %). Es la forma más fácil de ver para qué sirve el modelo.")

with tab_i:
    st.subheader("Importancia por permutación (bosque aleatorio, pacientes de prueba)")
    fig, a = plt.subplots(figsize=(8, 4.6))
    a.barh(ex["imp"].index, ex["imp"].values, xerr=ex["imp_sd"].values, color="#0055a5")
    a.set_xlabel("cuánto baja el AUC al mezclar la variable"); a.axvline(0, color="black", lw=0.8)
    plt.tight_layout(); st.pyplot(fig)
    st.write("Se mezcla al azar una variable a la vez y se mide cuánto empeora el modelo. **La edad domina**; glucosa e IMC aportan menos, y varias variables casi no cambian nada.")

with tab_n:
    st.subheader("Pacientes inventados que el modelo nunca vio")
    nu = ex["nuevos"]; U = ex["u14"]
    pr = pd.DataFrame({n: modelos[n].predict_proba(nu)[:, 1] for n in modelos}, index=nu.index)
    fig, a = plt.subplots(figsize=(11, 4.8)); w = 0.26; pos = np.arange(len(pr))
    for i, (n, c) in enumerate(zip(pr.columns, ["#0055a5", "#f0a800", "#2e8b57"])):
        a.bar(pos + (i - 1) * w, pr[n], w, label=n, color=c)
    a.axhline(0.5, color="#c0392b", ls="--", label="umbral 0,5"); a.axhline(U, color="#7f8c8d", ls=":", lw=2, label=f"umbral del 14 % ({U:.2f})")
    a.set_xticks(pos); a.set_xticklabels([p.split(" · ")[0] for p in pr.index]); a.set_ylim(0, 1.05); a.set_ylabel("probabilidad de ACV"); a.legend(fontsize=8, ncol=3)
    plt.tight_layout(); st.pyplot(fig)
    tabla = pr.round(2).copy(); tabla["logística: veredicto (14 %)"] = np.where(pr["Regresión logística"] >= U, "ACV", "sin ACV")
    st.dataframe(tabla)
    st.caption("Las probabilidades están infladas a propósito por el balanceo de clases: vale el orden y el veredicto según el umbral. Los pacientes son inventados, no hay un resultado real con el cual comparar.")

with tab_k:
    st.subheader("Clustering (k-medias, k = 3): grupos sin mirar el ACV")
    Z2, g, yy = ex["Z2"], ex["g"], ex["y_ent"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    for gi, c in zip(range(3), ["#2e8b57", "#0055a5", "#f0a800"]):
        a1.scatter(Z2[g == gi, 0], Z2[g == gi, 1], s=6, alpha=0.4, color=c, label=f"{gi}: {NOMBRES_G[gi]}")
    a1.legend(fontsize=8, markerscale=3); a1.set_title("Coloreado por grupo")
    a2.scatter(Z2[yy == 0, 0], Z2[yy == 0, 1], s=6, alpha=0.25, color="#9ca3af", label="sin ACV")
    a2.scatter(Z2[yy == 1, 0], Z2[yy == 1, 1], s=14, alpha=0.9, color="#c0392b", label="con ACV"); a2.legend(fontsize=8, markerscale=2); a2.set_title("Coloreado por ACV real")
    for a in (a1, a2): a.set_xlabel("componente 1"); a.set_ylabel("componente 2")
    plt.tight_layout(); st.pyplot(fig)
    st.write(f"Tasa de ACV por grupo: jóvenes **{tasa_g[0] * 100:.1f} %**, adultos con glucosa normal **{tasa_g[1] * 100:.1f} %**, adultos con glucosa alta **{tasa_g[2] * 100:.1f} %**. "
             "Los grupos **no coinciden** con ACV sí/no (los puntos rojos están repartidos), pero sirven como perfiles descriptivos de riesgo. La estructura es débil: es un resumen, no un clasificador.")
