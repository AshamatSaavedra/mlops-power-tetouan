import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import List
import joblib
import os


# -------------------------------------------------------------------------
# 1. Selección de columnas numéricas relevantes
# -------------------------------------------------------------------------
def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    exclude = ["DateTime"]
    numeric_cols = [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]
    return numeric_cols


# -------------------------------------------------------------------------
# 2. Entrenar el scaler y guardarlo
# -------------------------------------------------------------------------
def fit_scaler(df: pd.DataFrame, cols: List[str], model_path="models/scaler.pkl") -> RobustScaler:
    """
    Entrena un RobustScaler únicamente con las columnas especificadas.
    Guarda el scaler en disco para uso posterior.
    """
    scaler = RobustScaler()
    scaler.fit(df[cols])

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(scaler, model_path)
    print(f"[preprocessing] Scaler guardado en: {model_path}")

    return scaler


# -------------------------------------------------------------------------
# 3. Aplicar un scaler existente
# -------------------------------------------------------------------------
def apply_scaler(df: pd.DataFrame, scaler: RobustScaler, cols: List[str]) -> pd.DataFrame:
    """
    Aplica el scaler a las columnas especificadas.
    Retorna un nuevo DataFrame.
    """
    df_scaled = df.copy()
    df_scaled[cols] = scaler.transform(df[cols])

    return df_scaled


# -------------------------------------------------------------------------
# 4. Pipeline completo de preprocesamiento
# -------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, scaler_path="models/scaler.pkl") -> pd.DataFrame:
    """
    Pipeline principal:
    - Identifica columnas numéricas
    - Entrena y guarda scaler
    - Aplica el escalado
    """
    print("[preprocessing] Iniciando preprocesamiento...")

    cols = get_numeric_columns(df)
    print(f"[preprocessing] Columnas numéricas detectadas: {cols}")

    scaler = fit_scaler(df, cols, model_path=scaler_path)

    df_scaled = apply_scaler(df, scaler, cols)

    print("[preprocessing] Preprocesamiento finalizado.")
    return df_scaled