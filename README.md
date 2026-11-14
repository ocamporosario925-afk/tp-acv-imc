# Riesgo de ACV · TP4 (Aprendizaje Automático e IA)

Aplicación web que estima, a partir de los datos de un paciente, **qué riesgo tiene de sufrir un ACV (accidente cerebrovascular)**.
Sirve para **priorizar** a quién llamar primero para un control preventivo. **No es una herramienta de diagnóstico.**

- **Autoras:** Alonso Emilia y Ocampo Rosario (Licenciatura en Bioinformática, UNRaf)
- **URL pública de la app:** _(completar al publicar en Streamlit Community Cloud)_

## Qué hace la app
- Pide los datos del paciente en la barra lateral (con validación de rangos y de combinaciones poco habituales).
- Devuelve un **nivel de riesgo (bajo, medio o alto)**, la probabilidad del modelo y el percentil entre los pacientes de prueba.
- Explica **por qué** da ese resultado (aporte de cada dato respecto de un paciente promedio).
- Muestra contexto: dónde cae el paciente entre los pacientes de prueba, a qué grupo de pacientes se parece (clustering), curva de ganancia, importancia de variables y pacientes de ejemplo.
- La pestaña **Acerca del modelo** resume el rendimiento y las limitaciones.

## Modelo desplegado
Regresión logística con pesos de clase (`class_weight="balanced"`), dentro de un `Pipeline` de scikit-learn que completa faltantes con la mediana, estandariza los números y pasa el texto a columnas de 0 y 1 (One-Hot). Es el mejor modelo individual del TP3 (AUC ≈ 0,84 en la prueba) y el más simple de explicar.
El modelo ya entrenado está guardado en `modelo_acv.joblib` y **la app lo carga sin reentrenar**.

## Archivos
| Archivo | Para qué sirve |
| --- | --- |
| `app_tp3.py` | La aplicación (Streamlit) |
| `modelo_acv.joblib` | Modelo entrenado, preprocesamiento y tablas que usa la app |
| `entrenar_modelo.py` | Entrena y guarda el modelo (`python entrenar_modelo.py`) |
| `healthcare-dataset-stroke-data.csv` | Dataset *Stroke Prediction* (Kaggle) usado en todo el proyecto |
| `requirements.txt` | Librerías necesarias |

## Cómo ejecutarla en tu computadora
1. Instalar las librerías: `pip install -r requirements.txt`
2. Abrir una terminal en esta carpeta y escribir: `streamlit run app_tp3.py`
3. Se abre sola en el navegador (por lo general en `http://localhost:8501`).

Para volver a entrenar el modelo: `python entrenar_modelo.py` (genera de nuevo `modelo_acv.joblib`).

Si el archivo `modelo_acv.joblib` no se puede leer (por ejemplo, porque se guardó con otra versión de scikit-learn), la app lo avisa y reentrena el modelo al abrirse, así que siempre funciona.

## Versiones con las que se guardó el modelo
Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3.

## Datos y limitaciones
- 5109 pacientes; solo 249 tuvieron un ACV (≈ 5 %). La partición es 80 % entrenamiento y 20 % prueba, con la misma proporción de ACV en ambas.
- Con pocos casos positivos (50 en la prueba) los resultados tienen mucho margen de error.
- La precisión es baja: muchas alarmas son falsas. Las probabilidades están infladas a propósito por el balanceo de clases; lo que vale es el orden entre pacientes y el nivel de riesgo.
- El dataset es público y no fue validado clínicamente; el modelo puede rendir distinto en otras poblaciones.
