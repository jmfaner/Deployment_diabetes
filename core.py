# core.py
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

# Rutas a artefactos (ajusta si están en otra carpeta)
PREPROC_PATH = "preproc_med_cap.pkl"
MODEL_PATH   = "svm_smote_pipeline_vfinal.pkl"
POLICY_PATH  = "decision_policy_vfinal.pkl"

def load_artifacts() -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """
    Carga bundle de preprocesado, modelo y política (umbral).
    Devuelve (bundle, model, policy_dict).
    """
    bundle = joblib.load(PREPROC_PATH)   # dict: cols_cero, feature_cols, imp_med, low_med, up_med
    model  = joblib.load(MODEL_PATH)     # pipeline SVM+SMOTE (sklearn)
    policy = joblib.load(POLICY_PATH)    # {"threshold": ..., "policy": "..."}
    return bundle, model, policy

def preprocess(df_raw: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """
    0->NaN en columnas con ceros inválidos, imputación mediana y capping IQR
    según el bundle aprendido en TRAIN.
    """
    X = df_raw[bundle["feature_cols"]].copy()
    X[bundle["cols_cero"]] = X[bundle["cols_cero"]].replace(0, np.nan)
    X_imp = pd.DataFrame(
        bundle["imp_med"].transform(X),
        columns=bundle["feature_cols"],
        index=X.index,
    )
    X_cap = X_imp.clip(lower=bundle["low_med"], upper=bundle["up_med"], axis=1)
    return X_cap

def predict_df(df_in: pd.DataFrame, bundle: Dict[str, Any], model, threshold: float) -> pd.DataFrame:
    """
    Aplica preprocess + predict_proba + corte por umbral. Devuelve df con columnas:
    'prob_diabetes' y 'prediccion'.
    """
    X_cap = preprocess(df_in, bundle)
    proba = model.predict_proba(X_cap)[:, 1]
    pred  = (proba >= threshold).astype(int)
    out = df_in.copy()
    out["prob_diabetes"] = np.round(proba, 3)
    out["prediccion"]    = pred
    return out

