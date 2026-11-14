# TP4 · Aplicación web: riesgo de ACV
# Carga el modelo ya entrenado (modelo_acv.joblib) y estima el riesgo de ACV de un paciente.
# Para abrirla en tu compu:   streamlit run app_tp3.py
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import entrenar_modelo as em   # constantes y, solo si el archivo guardado no se puede leer, el reentrenamiento

st.set_page_config(page_title="Riesgo de ACV · TP4", layout="wide")
CARPETA = Path(__file__).resolve().parent
NOMBRES_G = {0: "jóvenes sin factores", 1: "adultos con glucosa normal", 2: "adultos con glucosa alta"}
ETIQUETAS = {"age": "Edad", "avg_glucose_level": "Glucosa", "bmi": "IMC", "hypertension": "Hipertensión", "heart_disease": "Cardiopatía",
             "gender": "Género", "ever_married": "Estado civil", "work_type": "Tipo de trabajo", "Residence_type": "Residencia", "smoking_status": "Tabaquismo"}
MODELO = "Regresión logística"


@st.cache_resource
def cargar():
    """Carga el modelo guardado. Solo si no se puede leer (por ejemplo, otra versión de scikit-learn) lo reentrena."""
    try:
        paquete = joblib.load(CARPETA / "modelo_acv.joblib")
        prueba = pd.DataFrame([{"age": 50, "avg_glucose_level": 100.0, "bmi": 27.0, "hypertension": 0, "heart_disease": 0, "gender": "Female",
                                "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban", "smoking_status": "never smoked"}])[em.COLUMNAS]
        paquete["modelos"][MODELO].predict_proba(prueba)          # comprobación rápida de que el modelo funciona
        return paquete, "archivo"
    except Exception:
        return em.construir(), "reentrenado"


paquete, origen = cargar()
modelos, p_pru, y_pru, u14 = paquete["modelos"], paquete["p_pru"], paquete["y_pru"], paquete["u14"]
cl = paquete["cluster"]

# ---------------------------------------------------------------- encabezado
st.title("¿Qué riesgo de ACV estima el modelo para un paciente?")
st.markdown(
    "**Qué problema resuelve.** Una obra social o un centro de salud no puede llamar a todos sus pacientes para un control preventivo. "
    "Esta herramienta estima, con los datos de un paciente, **qué riesgo tiene de sufrir un ACV (accidente cerebrovascular)** y ayuda a decidir a quién llamar primero.  \n"
    f"**Qué modelo usa.** Una **regresión logística** entrenada con {paquete['n_pacientes']} pacientes del dataset *Stroke Prediction* "
    f"(solo {paquete['n_acv']} tuvieron un ACV, el 5 %). Devuelve una probabilidad y un nivel de riesgo.  \n"
    "**Importante:** sirve para *priorizar*, no para diagnosticar. No reemplaza la evaluación de un profesional de la salud."
)

# ---------------------------------------------------------------- entradas (barra lateral) y validación
with st.sidebar:
    st.header("Datos del paciente")
    edad = st.slider("Edad (años)", 0, 82, 62)
    glucosa = st.slider("Glucosa promedio (mg/dL)", 55, 272, 130)
    bmi = st.slider("IMC", 10.0, 60.0, 28.0, step=0.5)
    hta = st.checkbox("Hipertensión", value=True)
    card = st.checkbox("Cardiopatía")
    genero = st.selectbox("Género", ["Female", "Male"], format_func=lambda x: {"Female": "Mujer", "Male": "Varón"}[x])
    casado = st.selectbox("Alguna vez casado/a", ["Yes", "No"], format_func=lambda x: {"Yes": "Sí", "No": "No"}[x])
    trabajo = st.selectbox("Tipo de trabajo", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                           format_func=lambda x: {"Private": "Privado", "Self-employed": "Cuenta propia", "Govt_job": "Estatal", "children": "Menor (children)", "Never_worked": "Nunca trabajó"}[x])
    zona = st.selectbox("Zona de residencia", ["Urban", "Rural"], format_func=lambda x: {"Urban": "Urbana", "Rural": "Rural"}[x])
    fuma = st.selectbox("Tabaquismo", ["never smoked", "formerly smoked", "smokes", "Unknown"],
                        format_func=lambda x: {"never smoked": "Nunca fumó", "formerly smoked": "Fumó antes", "smokes": "Fuma", "Unknown": "Sin dato"}[x])
    st.caption("Modelo cargado desde modelo_acv.joblib, sin reentrenar." if origen == "archivo"
               else "Aviso: no se pudo leer modelo_acv.joblib (falta el archivo o es de otra versión de scikit-learn); el modelo se reentrenó al abrir la app.")


def validar(edad, trabajo, casado, fuma):
    """Devuelve (errores, avisos). Un error impide calcular el resultado; un aviso solo advierte."""
    errores, avisos = [], []
    if edad >= 18 and trabajo == "children":
        errores.append("«Menor (children)» solo corresponde a menores de edad. Cambiá el tipo de trabajo o la edad.")
    if edad < 7 and trabajo in ("Private", "Self-employed", "Govt_job", "Never_worked"):
        avisos.append("En los datos no hay pacientes menores de 7 años con ese tipo de trabajo: el resultado puede ser poco confiable.")
    if edad > 23 and trabajo == "Never_worked":
        avisos.append("En los datos, quienes «nunca trabajaron» tienen hasta 23 años: el resultado puede ser poco confiable.")
    if edad < 18 and casado == "Yes":
        avisos.append("En los datos nadie menor de 18 años estuvo casado/a: el resultado puede ser poco confiable.")
    if edad < 10 and fuma in ("smokes", "formerly smoked"):
        avisos.append("En los datos nadie menor de 10 años fuma ni fumó: el resultado puede ser poco confiable.")
    return errores, avisos


errores, avisos = validar(edad, trabajo, casado, fuma)
paciente = pd.DataFrame([{"age": edad, "avg_glucose_level": glucosa, "bmi": bmi, "hypertension": int(hta), "heart_disease": int(card),
                          "gender": genero, "ever_married": casado, "work_type": trabajo, "Residence_type": zona, "smoking_status": fuma}])[em.COLUMNAS]

tab_p, tab_c, tab_g, tab_i, tab_n, tab_k, tab_a = st.tabs(
    ["Paciente", "Correlación", "Curva de ganancia", "Importancia", "Pacientes nuevos (A a H)", "Clustering", "Acerca del modelo"])

# ---------------------------------------------------------------- pestaña 1: paciente
with tab_p:
    for e in errores:
        st.error(e)
    for a in avisos:
        st.warning(a)
    if errores:
        st.info("Corregí los datos de la barra lateral para ver el resultado.")
    else:
        prob = {n: float(m.predict_proba(paciente)[0, 1]) for n, m in modelos.items()}
        p = prob[MODELO]
        percentil = float((p_pru[MODELO] < p).mean() * 100)
        if p >= u14:
            nivel, caja, texto = "ALTO", st.error, "Está entre el 14 % de pacientes de mayor riesgo. Conviene priorizarlo para un control preventivo."
        elif p >= 0.5:
            nivel, caja, texto = "MEDIO", st.warning, "Riesgo intermedio: por encima del corte básico, pero fuera del 14 % de mayor riesgo."
        else:
            nivel, caja, texto = "BAJO", st.success, "Riesgo por debajo del corte básico (0,5)."
        caja(f"### Riesgo {nivel}\n{texto}  \nSupera en riesgo al **{percentil:.0f} %** de los pacientes de prueba (probabilidad del modelo: {p:.2f}).")
        st.caption("Cortes: **bajo** por debajo de 0,5; **medio** de 0,5 hasta {:.2f}; **alto** desde {:.2f} (el 14 % de mayor riesgo, elegido con el entrenamiento). "
                   "Las probabilidades están infladas a propósito por el balanceo de clases (solo ~5 % de ACV): lo que vale es el orden entre pacientes y el nivel.".format(u14, u14))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Por qué da este resultado")
            modelo = modelos[MODELO]
            nombres = modelo[0].get_feature_names_out()
            x = np.asarray(modelo[0].transform(paciente))[0]
            contrib = modelo[-1].coef_[0] * (x - paquete["x_medio"])
            por_var = {}
            for n, c in zip(nombres, contrib):
                base = n.split("__", 1)[1]
                for col in em.CATEGORICAS:
                    if base.startswith(col + "_"):
                        base = col
                por_var[base] = por_var.get(base, 0.0) + float(c)
            serie = pd.Series(por_var).rename(index=ETIQUETAS)
            serie = serie.reindex(serie.abs().sort_values().index)
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            ax.barh(serie.index, serie.values, color=["#c0392b" if v > 0 else "#0055a5" for v in serie.values])
            ax.axvline(0, color="black", lw=0.8); ax.set_xlabel("← baja el riesgo · sube el riesgo →")
            ax.set_title("Aporte de cada dato, comparado con un paciente promedio")
            plt.tight_layout(); st.pyplot(fig)
            sube = serie[serie > 0.05].sort_values(ascending=False).index.tolist()
            st.caption("Lo que más sube el riesgo en este paciente: " + (", ".join(sube[:3]) if sube else "nada en particular (queda cerca del promedio)") + ".")
        with c2:
            st.subheader("Dónde cae entre los pacientes de prueba")
            fig2, ax2 = plt.subplots(figsize=(6.2, 4.2))
            ax2.hist(p_pru[MODELO][y_pru == 0], bins=30, alpha=0.6, color="#9ca3af", label="sin ACV", density=True)
            ax2.hist(p_pru[MODELO][y_pru == 1], bins=30, alpha=0.7, color="#c0392b", label="con ACV", density=True)
            ax2.axvline(p, color="black", lw=2, label="este paciente"); ax2.axvline(u14, color="#7f8c8d", ls=":", lw=1.5, label="corte de riesgo alto")
            ax2.set_xlabel("probabilidad de ACV según el modelo"); ax2.set_ylabel("densidad"); ax2.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig2)
            st.caption("Cada barra es un grupo de pacientes reales que el modelo no vio al entrenar. Cuanto más a la derecha cae el paciente, más riesgo.")

        grupo = cl["remapa"][int(cl["km"].predict(np.asarray(cl["pre"].transform(paciente)))[0])]
        st.subheader("¿A qué grupo de pacientes se parece?")
        t = cl["tasa"]
        st.write(f"Se parece al grupo **{grupo}: {NOMBRES_G[grupo]}**, donde el {t[grupo] * 100:.1f} % de los pacientes tuvo un ACV "
                 f"(jóvenes sin factores {t[0] * 100:.1f} %, adultos con glucosa normal {t[1] * 100:.1f} %, adultos con glucosa alta {t[2] * 100:.1f} %).")

        with st.expander("Ver qué dicen los otros modelos entrenados"):
            cols = st.columns(3)
            for c, n in zip(cols, modelos):
                c.metric(n, f"prob. {prob[n]:.2f}", "riesgo alto" if prob[n] >= u14 else ("riesgo medio" if prob[n] >= 0.5 else "riesgo bajo"), delta_color="off")
            st.caption("Los tres modelos se entrenaron igual; se despliega la regresión logística porque es la más simple, la que mejor ordena (AUC 0,84) y la más fácil de explicar.")

# ---------------------------------------------------------------- pestaña 2: correlación
with tab_c:
    st.subheader("Correlación: ninguna variable numérica se relaciona fuerte con el ACV")
    cm = paquete["corr"]; nombres_c = list(cm.columns)
    fig, a = plt.subplots(figsize=(7.5, 5.2))
    a.imshow(cm.values, cmap="Blues", vmin=0, vmax=1)
    a.set_xticks(range(6)); a.set_xticklabels(nombres_c, rotation=40, ha="right"); a.set_yticks(range(6)); a.set_yticklabels(nombres_c)
    for i in range(6):
        for j in range(6):
            a.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if cm.values[i, j] > 0.5 else "black")
    plt.tight_layout(); st.pyplot(fig)
    st.write("En la fila de **stroke**, la edad es la que más se relaciona (%.2f); hipertensión, cardiopatía y glucosa quedan alrededor de 0,13 a 0,14. "
             "Son valores bajos: por eso el problema es difícil y solo ~5 %% de los pacientes tiene ACV." % cm.loc["stroke", "age"])

# ---------------------------------------------------------------- pestaña 3: curva de ganancia
with tab_g:
    st.subheader("Curva de ganancia: ¿cuántos ACV encontramos llamando al X % de mayor riesgo?")
    fig, a = plt.subplots(figsize=(8.5, 5))
    for n, c in zip(modelos, ["#0055a5", "#f0a800", "#2e8b57"]):
        orden = np.argsort(-p_pru[n]); acum = np.cumsum(y_pru[orden]) / y_pru.sum()
        a.plot(np.arange(1, len(acum) + 1) / len(acum) * 100, acum * 100, label=n, color=c, lw=2)
    a.plot([0, 100], [0, 100], color="gray", ls="--", label="al azar")
    a.axvline(14, color="#c0392b", ls=":"); a.set_xlabel("% de pacientes llamados (de mayor a menor riesgo)"); a.set_ylabel("% de ACV encontrados"); a.legend(); a.grid(alpha=0.3)
    st.pyplot(fig)
    pp = p_pru[MODELO]; o = np.argsort(-pp); k = int(round(0.14 * len(pp)))
    st.write(f"Llamando al **14 %** de los pacientes con más riesgo, la regresión logística encuentra el **{y_pru[o][:k].sum() / y_pru.sum() * 100:.0f} %** de los ACV "
             "(al azar encontraría el 14 %). Es la forma más fácil de ver para qué sirve el modelo.")

# ---------------------------------------------------------------- pestaña 4: importancia
with tab_i:
    st.subheader("Importancia por permutación (bosque aleatorio, pacientes de prueba)")
    fig, a = plt.subplots(figsize=(8, 4.6))
    a.barh([ETIQUETAS[i] for i in paquete["imp"].index], paquete["imp"].values, xerr=paquete["imp_sd"].values, color="#0055a5")
    a.set_xlabel("cuánto baja el AUC al mezclar la variable"); a.axvline(0, color="black", lw=0.8)
    plt.tight_layout(); st.pyplot(fig)
    st.write("Se mezcla al azar una variable a la vez y se mide cuánto empeora el modelo. **La edad domina**; glucosa e IMC aportan menos, y varias variables casi no cambian nada.")

# ---------------------------------------------------------------- pestaña 5: pacientes nuevos
with tab_n:
    st.subheader("Pacientes inventados que el modelo nunca vio")
    nu = pd.DataFrame([
        ("A · mujer joven, sin antecedentes", "Female", 28, 0, 0, "No", "Private", "Urban", 85.0, 22.0, "never smoked"),
        ("B · varón de 45, sin antecedentes", "Male", 45, 0, 0, "Yes", "Private", "Urban", 100.0, 27.0, "formerly smoked"),
        ("C · mujer de 62, hipertensa", "Female", 62, 1, 0, "Yes", "Self-employed", "Rural", 130.0, 30.0, "never smoked"),
        ("D · varón de 72, hipertenso y cardiópata", "Male", 72, 1, 1, "Yes", "Private", "Urban", 210.0, 31.0, "formerly smoked"),
        ("E · mujer de 80, cardiópata", "Female", 80, 0, 1, "Yes", "Self-employed", "Rural", 105.0, 25.0, "never smoked"),
        ("F · varón de 58, hipertenso, glucosa alta", "Male", 58, 1, 0, "Yes", "Private", "Urban", 230.0, 35.0, "smokes"),
        ("G · niña de 8 años", "Female", 8, 0, 0, "No", "children", "Urban", 80.0, 17.0, "Unknown"),
        ("H · varón de 67, sin otros factores", "Male", 67, 0, 0, "Yes", "Private", "Rural", 95.0, 24.0, "never smoked")],
        columns=["perfil", "gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]).set_index("perfil")[em.COLUMNAS]
    pr = pd.DataFrame({n: modelos[n].predict_proba(nu)[:, 1] for n in modelos}, index=nu.index)
    fig, a = plt.subplots(figsize=(11, 4.8)); w = 0.26; pos = np.arange(len(pr))
    for i, (n, c) in enumerate(zip(pr.columns, ["#0055a5", "#f0a800", "#2e8b57"])):
        a.bar(pos + (i - 1) * w, pr[n], w, label=n, color=c)
    a.axhline(0.5, color="#c0392b", ls="--", label="umbral 0,5"); a.axhline(u14, color="#7f8c8d", ls=":", lw=2, label=f"umbral del 14 % ({u14:.2f})")
    a.set_xticks(pos); a.set_xticklabels([q.split(" · ")[0] for q in pr.index]); a.set_ylim(0, 1.05); a.set_ylabel("probabilidad de ACV"); a.legend(fontsize=8, ncol=3)
    plt.tight_layout(); st.pyplot(fig)
    tabla = pr.round(2).copy(); tabla["regresión logística: nivel"] = np.where(pr[MODELO] >= u14, "alto", np.where(pr[MODELO] >= 0.5, "medio", "bajo"))
    st.dataframe(tabla)
    st.caption("Los pacientes son inventados: no hay un resultado real con el cual compararlos. Las probabilidades están infladas a propósito por el balanceo de clases.")

# ---------------------------------------------------------------- pestaña 6: clustering
with tab_k:
    st.subheader("Clustering (K-medias, k = 3): grupos armados sin mirar el ACV")
    Z2, g, yy = cl["Z2"], cl["g"], cl["y"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    for gi, c in zip(range(3), ["#2e8b57", "#0055a5", "#f0a800"]):
        a1.scatter(Z2[g == gi, 0], Z2[g == gi, 1], s=6, alpha=0.4, color=c, label=f"{gi}: {NOMBRES_G[gi]}")
    a1.legend(fontsize=8, markerscale=3); a1.set_title("Coloreado por grupo")
    a2.scatter(Z2[yy == 0, 0], Z2[yy == 0, 1], s=6, alpha=0.25, color="#9ca3af", label="sin ACV")
    a2.scatter(Z2[yy == 1, 0], Z2[yy == 1, 1], s=14, alpha=0.9, color="#c0392b", label="con ACV"); a2.legend(fontsize=8, markerscale=2); a2.set_title("Coloreado por ACV real")
    for ax_ in (a1, a2): ax_.set_xlabel("componente 1"); ax_.set_ylabel("componente 2")
    plt.tight_layout(); st.pyplot(fig)
    t = cl["tasa"]
    st.write(f"Tasa de ACV por grupo: jóvenes **{t[0] * 100:.1f} %**, adultos con glucosa normal **{t[1] * 100:.1f} %**, adultos con glucosa alta **{t[2] * 100:.1f} %**. "
             "Los grupos **no coinciden** con ACV sí/no (los puntos rojos están repartidos), pero sirven como perfiles descriptivos de riesgo. "
             "El grupo más grande tiene más puntos rojos simplemente porque tiene más pacientes: lo que mide el riesgo es el porcentaje dentro de cada grupo.")

# ---------------------------------------------------------------- pestaña 7: acerca del modelo
with tab_a:
    st.subheader("Acerca del modelo")
    st.markdown(
        "**Modelo desplegado:** regresión logística con pesos de clase (`class_weight=\"balanced\"`), dentro de un *Pipeline* que completa faltantes con la mediana, "
        "estandariza los números y pasa el texto a columnas de 0 y 1 (One-Hot).  \n"
        "**Por qué este:** es el mejor modelo individual del TP3 (AUC ≈ 0,84), casi no se sobreajusta, es liviano y se puede explicar. "
        "Los ensambles no mejoraron lo suficiente como para justificar su complejidad.  \n"
        "**Datos:** 5109 pacientes (80 % entrenamiento, 20 % prueba, misma proporción de ACV en ambos). Todos los resultados de esta pestaña son sobre el 20 % de prueba.")
    filas = []
    for n in modelos:
        for etiqueta, umbral in (("0,5", 0.5), (f"{u14:.2f} (14 %)".replace(".", ","), u14)):
            pred = (p_pru[n] >= umbral).astype(int)
            filas.append({"modelo": n, "umbral": etiqueta, "AUC": roc_auc_score(y_pru, p_pru[n]), "recall (ACV detectados)": recall_score(y_pru, pred),
                          "precisión": precision_score(y_pru, pred, zero_division=0), "pacientes marcados": f"{pred.mean() * 100:.0f} %"})
    st.dataframe(pd.DataFrame(filas).set_index(["modelo", "umbral"]).round(3))
    st.caption("AUC: qué tan bien ordena por riesgo (0,5 = azar, 1 = perfecto). Recall: de los ACV reales, cuántos detecta. Precisión: de los que marca, cuántos son ACV. "
               "Los valores pueden diferir un poco de los del notebook porque acá los hiperparámetros están fijados.")
    st.markdown(
        "**Limitaciones**\n"
        "- Con solo 249 casos de ACV (50 en la prueba), los resultados tienen mucho margen de error.\n"
        "- La precisión es baja: muchas alarmas son falsas. Es una herramienta para priorizar controles, no para diagnosticar.\n"
        "- El dataset es público y no fue validado clínicamente; el modelo puede rendir distinto en otras poblaciones.\n"
        "- Pide datos que un paciente puede no conocer (glucosa promedio, IMC).")
    v = paquete["versiones"]
    st.caption(f"Modelo guardado con Python {v['python']}, scikit-learn {v['scikit-learn']}, pandas {v['pandas']} y numpy {v['numpy']}. "
               "Para reentrenarlo: `python entrenar_modelo.py`.")
