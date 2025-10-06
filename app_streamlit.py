# app.py (Streamlit)
import streamlit as st
import pandas as pd
from core import load_artifacts, predict_df

st.set_page_config(page_title="Predicción de diabetes", page_icon="🩺", layout="centered")
@st.cache_resource
def get_artifacts():
    bundle, model, policy = load_artifacts()
    return bundle, model, policy

bundle, model, policy = get_artifacts()
FEATURES = bundle["feature_cols"]
DEFAULT_THR = float(policy["threshold"])


st.title("🩺 Predicción de diabetes (SVM + SMOTE)")
st.caption("Preprocesado: 0→NaN, imputación mediana, capping IQR. Modelo: SVM-RBF + SMOTE. Umbral por CV (F2*).")

with st.sidebar:
    st.header("Configuración")
    use_default = st.toggle("Usar umbral guardado (F2*)", value=True)
    if use_default:
        thr = DEFAULT_THR
    else:
        thr = st.slider("Umbral de decisión", 0.01, 0.99, DEFAULT_THR, 0.01)
    st.write(f"**Umbral activo:** {thr:.3f}")

# ---- Formulario de un paciente ----
st.subheader("Predicción individual")
c1, c2, c3, c4 = st.columns(4)
Pregnancies   = c1.number_input("Pregnancies", 0.0, 20.0, 1.0, step=1.0)
Glucose       = c2.number_input("Glucose", 0.0, 300.0, 120.0)
BloodPressure = c3.number_input("BloodPressure", 0.0, 200.0, 70.0)
SkinThickness = c4.number_input("SkinThickness", 0.0, 100.0, 20.0)
Insulin       = c1.number_input("Insulin", 0.0, 900.0, 80.0)
BMI           = c2.number_input("BMI", 0.0, 80.0, 30.0)
DPF           = c3.number_input("DiabetesPedigreeFunction", 0.0, 5.0, 0.5)
Age           = c4.number_input("Age", 0.0, 120.0, 35.0)

if st.button("Predecir"):
    row = pd.DataFrame([{
        "Pregnancies": Pregnancies, "Glucose": Glucose, "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness, "Insulin": Insulin, "BMI": BMI,
        "DiabetesPedigreeFunction": DPF, "Age": Age
    }])
    res = predict_df(row, bundle, model, thr)
    st.metric("Probabilidad", f"{float(res.loc[0, 'prob_diabetes']):.3f}")
    st.metric("Predicción", "Diabético" if int(res.loc[0, 'prediccion'])==1 else "No diabético")

# ---- CSV por lotes ----
st.subheader("Predicción por lotes (CSV)")
st.caption("El CSV debe contener exactamente estas columnas: " + ", ".join(FEATURES))
up = st.file_uploader("Sube un CSV", type=["csv"])
if up is not None:
    try:
        df = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Faltan columnas: {missing}")
        else:
            res = predict_df(df[FEATURES], bundle, model, thr)
            st.dataframe(res)
            st.download_button(
                "Descargar resultados CSV",
                data=res.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_diabetes.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error leyendo el CSV: {e}")
