# TP2 · Regresión · aplicación Streamlit
# Estima el IMC (bmi) de un paciente a partir de las demás columnas, con cuatro modelos.
# Para abrirla:  streamlit run app_tp2.py      (el CSV tiene que estar en la misma carpeta)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="TP2 · Estimar el IMC", layout="wide")
NUMERICAS = ["age", "hypertension", "heart_disease", "avg_glucose_level", "stroke"]
CATEGORICAS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


@st.cache_resource
def entrenar():
    d = pd.read_csv("healthcare-dataset-stroke-data.csv").drop(columns="id")
    d = d[d["gender"] != "Other"].dropna(subset=["bmi"])          # igual que el notebook: sin la fila "Other" ni los pacientes sin bmi
    X, y = d[NUMERICAS + CATEGORICAS], d["bmi"]
    X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.20, random_state=42)

    def preparado(modelo, grado=1):
        num = make_pipeline(SimpleImputer(strategy="median"), PolynomialFeatures(grado, include_bias=False), StandardScaler())
        return make_pipeline(make_column_transformer((num, NUMERICAS), (OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)), modelo)

    modelos = {
        "Línea de base (la media)": preparado(DummyRegressor()),
        "Regresión lineal": preparado(LinearRegression()),
        "Lasso (grado 2)": preparado(LassoCV(cv=5, random_state=42, max_iter=20000), grado=2),
        "Bosque aleatorio": preparado(RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1)),
    }
    filas = []
    for nombre, m in modelos.items():
        m.fit(X_ent, y_ent)
        p = m.predict(X_pru)
        filas.append({"modelo": nombre, "RMSE (prueba)": mean_squared_error(y_pru, p) ** 0.5, "R² (prueba)": r2_score(y_pru, p)})
    # modelos extra, solo para la pestaña de entrenamiento contra prueba (no entran en el selector de pacientes)
    extra = {"KNN (k = 5)": preparado(KNeighborsRegressor(n_neighbors=5)), "Árbol sin poda": preparado(DecisionTreeRegressor(random_state=42))}
    todos = {**modelos, **extra}
    comp = []
    for nombre, m in todos.items():
        if nombre in extra: m.fit(X_ent, y_ent)
        comp.append({"modelo": nombre, "entrenamiento": mean_squared_error(y_ent, m.predict(X_ent)) ** 0.5, "prueba": mean_squared_error(y_pru, m.predict(X_pru)) ** 0.5})
    return modelos, pd.DataFrame(filas).set_index("modelo"), X_pru, y_pru, d, pd.DataFrame(comp).set_index("modelo")


@st.cache_resource
def preliminar():
    """Prueba preliminar: ¿qué variable se puede predecir mejor, el IMC o la glucosa?"""
    d0 = pd.read_csv("healthcare-dataset-stroke-data.csv").drop(columns="id")
    d0 = d0[d0["gender"] != "Other"]
    res = {}
    for objetivo in ["bmi", "avg_glucose_level"]:
        dd = d0.dropna(subset=[objetivo])
        X, y = dd.drop(columns=objetivo), dd[objetivo]
        num = [c for c in X.columns if c not in CATEGORICAS]
        Xe, Xp, ye, yp = train_test_split(X, y, test_size=0.20, random_state=42)
        for nombre, m in [("Regresión lineal", LinearRegression()),
                          ("Bosque aleatorio", RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1))]:
            pre = make_column_transformer((make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                                          (OneHotEncoder(handle_unknown="ignore"), CATEGORICAS))
            res[(objetivo, nombre)] = r2_score(yp, make_pipeline(pre, m).fit(Xe, ye).predict(Xp))
    return res


modelos, tabla, X_pru, y_pru, datos, comp = entrenar()
st.title("¿Cuál es el IMC de este paciente?")
st.caption("Regresión (TP2): el modelo estima el IMC a partir de las demás columnas. Se entrena con el 80 % de los pacientes y se mide con el 20 % que no vio.")

with st.sidebar:
    st.header("Paciente")
    edad = st.slider("Edad (años)", 0, 82, 45)
    glucosa = st.slider("Glucosa promedio (mg/dL)", 55, 272, 100)
    hta = st.checkbox("Hipertensión")
    card = st.checkbox("Cardiopatía")
    acv = st.checkbox("Tuvo un ACV")
    genero = st.selectbox("Género", ["Female", "Male"])
    casado = st.selectbox("Alguna vez casado/a", ["Yes", "No"])
    trabajo = st.selectbox("Tipo de trabajo", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    zona = st.selectbox("Zona de residencia", ["Urban", "Rural"])
    fuma = st.selectbox("Tabaquismo", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    elegido = st.selectbox("Modelo a mostrar", list(modelos)[1:], index=2)

paciente = pd.DataFrame([{"age": edad, "hypertension": int(hta), "heart_disease": int(card), "avg_glucose_level": glucosa, "stroke": int(acv),
                          "gender": genero, "ever_married": casado, "work_type": trabajo, "Residence_type": zona, "smoking_status": fuma}])[NUMERICAS + CATEGORICAS]
pred = {n: float(m.predict(paciente)[0]) for n, m in modelos.items()}
rmse = tabla.loc[elegido, "RMSE (prueba)"]

tab_p, tab_c, tab_pre, tab_pr, tab_et, tab_n = st.tabs(["Paciente", "Correlación", "Prueba preliminar", "Predicho vs real", "Entrenamiento vs prueba", "Pacientes nuevos (A a H)"])
with tab_p:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.subheader("Estimación")
        st.metric(f"IMC estimado ({elegido})", f"{pred[elegido]:.1f}", help="Valor que predice el modelo elegido")
        st.write(f"Con el error típico de prueba de este modelo (RMSE = {rmse:.1f} puntos), el IMC estaría **entre {pred[elegido] - rmse:.0f} y {pred[elegido] + rmse:.0f}**.")
        st.info("El rango es amplio: con estas variables el modelo explica menos de un tercio de la variación del IMC (R² ≈ 0,25 a 0,28).")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        nombres = list(pred)
        colores = ["#9ca3af", "#0055a5", "#f0a800", "#2e8b57"]
        ax.barh(nombres, [pred[n] for n in nombres], color=colores); ax.invert_yaxis()
        ax.axvspan(18.5, 25, color="#2e8b57", alpha=0.08); ax.axvspan(30, 45, color="#c0392b", alpha=0.06)
        ax.set_xlim(10, 45); ax.set_xlabel("IMC estimado (franja verde: normal · roja: obesidad)")
        for i, n in enumerate(nombres): ax.text(pred[n] + 0.3, i, f"{pred[n]:.1f}", va="center", fontsize=9)
        ax.set_title("Qué estima cada modelo para este paciente")
        st.pyplot(fig)
    with c2:
        st.subheader(f"Predicho contra real · {elegido}")
        p_pru = modelos[elegido].predict(X_pru)
        fig2, ax2 = plt.subplots(figsize=(6.5, 5))
        ax2.scatter(y_pru, p_pru, s=8, alpha=0.35, color="#0055a5")
        lim = [10, 60]; ax2.plot(lim, lim, color="#f0a800", lw=2, label="predicción perfecta")
        ax2.scatter([pred[elegido]], [pred[elegido]], marker="*", s=300, color="#c0392b", edgecolor="white", zorder=5, label="este paciente (estimado)")
        ax2.set_xlim(lim); ax2.set_ylim(lim); ax2.set_xlabel("IMC real (pacientes de prueba)"); ax2.set_ylabel("IMC predicho"); ax2.legend(fontsize=8)
        st.pyplot(fig2)
        st.caption("Si el modelo fuera perfecto, los puntos estarían sobre la línea amarilla. La nube plana muestra que predice poco.")

    st.subheader("Error de cada modelo en los pacientes de prueba")
    st.dataframe(tabla.round(3))
    st.caption("RMSE: cuántos puntos de IMC se equivoca el modelo en promedio (menor es mejor). R²: qué parte de la variación explica (0 = nada, 1 = todo). Los valores pueden diferir un poco de los del notebook porque acá los hiperparámetros están fijados.")


with tab_c:
    st.subheader("Correlación: ninguna variable explica bien el IMC")
    cols = ["bmi", "age", "hypertension", "heart_disease", "avg_glucose_level", "stroke"]
    corr = datos[cols].corr()
    fig3, (a3, b3) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
    a3.hist(datos["bmi"], bins=40, color="#0055a5", alpha=0.85); a3.axvline(datos["bmi"].mean(), color="#c0392b", ls="--", label=f"media {datos['bmi'].mean():.1f}")
    a3.set_xlabel("bmi"); a3.set_ylabel("pacientes"); a3.legend(); a3.set_title("El target: distribución del IMC")
    im = b3.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
    b3.set_xticks(range(len(cols))); b3.set_xticklabels(cols, rotation=40, ha="right", fontsize=9); b3.set_yticks(range(len(cols))); b3.set_yticklabels(cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            b3.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if corr.values[i, j] > 0.5 else "black")
    b3.set_title("Correlación entre variables numéricas")
    plt.tight_layout(); st.pyplot(fig3)
    st.write("La fila del **bmi**: la edad es la que más se relaciona (%.2f); glucosa e hipertensión, mucho menos (%.2f y %.2f). Todas son bajas, lo que anticipa que el IMC va a ser difícil de predecir." % (corr.loc["bmi", "age"], corr.loc["bmi", "avg_glucose_level"], corr.loc["bmi", "hypertension"]))

with tab_pr:
    st.subheader("Predicho contra real: si el modelo fuera perfecto, los puntos estarían sobre la diagonal")
    fig4, ejes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    for ax, n in zip(ejes, list(modelos)[1:]):
        p = modelos[n].predict(X_pru)
        ax.scatter(y_pru, p, s=7, alpha=0.3, color="#0055a5"); ax.plot([10, 60], [10, 60], color="#f0a800", lw=2)
        ax.set_title(f"{n} · R² {tabla.loc[n, 'R² (prueba)']:.2f}"); ax.set_xlabel("IMC real"); ax.set_xlim(10, 60); ax.set_ylim(10, 60)
    ejes[0].set_ylabel("IMC predicho"); plt.tight_layout(); st.pyplot(fig4)
    st.write("Se ven dos grupos: los **niños** (predicción cerca de 21) y los **adultos** (entre 24 y 36). El modelo distingue bien a un niño de un adulto, pero casi no distingue a un adulto de otro: de ahí el R² bajo.")

with tab_et:
    st.subheader("Un error bajo en entrenamiento no alcanza: hay que mirar la prueba")
    fig5, ax5 = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(comp)); w = 0.38
    ax5.bar(x - w / 2, comp["entrenamiento"], w, color="#9aa3af", label="entrenamiento")
    bb = ax5.bar(x + w / 2, comp["prueba"], w, color="#0055a5", label="prueba")
    for r in bb: ax5.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.12, f"{r.get_height():.1f}", ha="center", fontsize=10)
    ax5.set_xticks(x); ax5.set_xticklabels([n.replace(" (", "\n(") for n in comp.index], fontsize=9); ax5.set_ylabel("error (RMSE, puntos de IMC)"); ax5.legend(); ax5.set_ylim(0, 10.5)
    plt.tight_layout(); st.pyplot(fig5)
    st.write("El **árbol sin poda** tiene error 0 en entrenamiento y mucho mayor en prueba: memorizó los datos y no generaliza (sobreajuste). Los demás quedan entre 6,8 y 7,3 en prueba.")

with tab_n:
    st.subheader("Pacientes inventados que el modelo nunca vio")
    NUEVOS = pd.DataFrame([
        ("A · mujer joven, sin antecedentes", "Female", 28, 0, 0, "No", "Private", "Urban", 85.0, "never smoked", 0),
        ("B · varón de 45, sin antecedentes", "Male", 45, 0, 0, "Yes", "Private", "Urban", 100.0, "formerly smoked", 0),
        ("C · mujer de 62, hipertensa", "Female", 62, 1, 0, "Yes", "Self-employed", "Rural", 130.0, "never smoked", 0),
        ("D · varón de 72, hipertenso y cardiópata", "Male", 72, 1, 1, "Yes", "Private", "Urban", 210.0, "formerly smoked", 1),
        ("E · mujer de 80, cardiópata", "Female", 80, 0, 1, "Yes", "Self-employed", "Rural", 105.0, "never smoked", 0),
        ("F · varón de 58, hipertenso, glucosa alta", "Male", 58, 1, 0, "Yes", "Private", "Urban", 230.0, "smokes", 0),
        ("G · niña de 8 años", "Female", 8, 0, 0, "No", "children", "Urban", 80.0, "Unknown", 0),
        ("H · varón de 67, sin otros factores", "Male", 67, 0, 0, "Yes", "Private", "Rural", 95.0, "never smoked", 0)],
        columns=["perfil", "gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "smoking_status", "stroke"]).set_index("perfil")[NUMERICAS + CATEGORICAS]
    est = pd.DataFrame({n: modelos[n].predict(NUEVOS) for n in list(modelos)[1:]}, index=NUEVOS.index)
    fig6, ax6 = plt.subplots(figsize=(11, 4.6)); ancho = 0.26; pos = np.arange(len(est))
    for i, (n, c) in enumerate(zip(est.columns, ["#0055a5", "#f0a800", "#2e8b57"])):
        ax6.bar(pos + (i - 1) * ancho, est[n], ancho, label=n, color=c)
    ax6.axhline(datos["bmi"].mean(), color="#c0392b", ls="--", label="media del dataset")
    ax6.axhspan(18.5, 25, color="#2e8b57", alpha=0.08); ax6.axhspan(30, 45, color="#c0392b", alpha=0.06)
    ax6.set_xticks(pos); ax6.set_xticklabels([p.split(" · ")[0] for p in est.index]); ax6.set_ylim(10, 45); ax6.set_ylabel("IMC estimado"); ax6.legend(fontsize=8, ncol=2)
    ax6.set_title("IMC estimado por cada modelo (franja verde: normal · roja: obesidad)")
    plt.tight_layout(); st.pyplot(fig6)
    st.dataframe(est.round(1))
    st.caption("Las letras son los perfiles de la tabla. Los modelos coinciden entre sí, pero el error típico es de unos 7 puntos de IMC: cada barra es una estimación aproximada. Los pacientes son inventados, no tienen un IMC real con el cual comparar.")

with tab_pre:
    st.subheader("Prueba preliminar: ¿por qué elegimos el IMC como variable a estimar?")
    rp = preliminar()
    fig7, a7 = plt.subplots(figsize=(8, 4.6)); w = 0.35; xs = np.arange(2)
    for i, (n, c) in enumerate([("Regresión lineal", "#0055a5"), ("Bosque aleatorio", "#2e8b57")]):
        v = [rp[("bmi", n)], rp[("avg_glucose_level", n)]]
        bb = a7.bar(xs + (i - 0.5) * w, v, w, label=n, color=c)
        for r in bb: a7.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008, f"{r.get_height():.2f}".replace(".", ","), ha="center", fontsize=11)
    a7.set_xticks(xs); a7.set_xticklabels(["IMC (bmi)", "Glucosa"], fontsize=12); a7.set_ylabel("R² en prueba"); a7.set_ylim(0, max(rp.values()) * 1.25); a7.legend()
    plt.tight_layout(); st.pyplot(fig7)
    st.write("Antes de elegir, probamos predecir dos variables continuas con una regresión lineal y un bosque. El **IMC** dio un R² de %s a %s y la **glucosa** solo %s a %s: las demás columnas casi no explican la glucosa. Por eso elegimos el IMC."
             % tuple(f"{x:.2f}".replace(".", ",") for x in (rp[("bmi", "Regresión lineal")], rp[("bmi", "Bosque aleatorio")], rp[("avg_glucose_level", "Bosque aleatorio")], rp[("avg_glucose_level", "Regresión lineal")])))
    st.caption("R² = qué parte de la variación explica el modelo (0 = lo mismo que adivinar el promedio, 1 = perfecto). Es una prueba rápida, previa a los resultados finales del TP2.")
