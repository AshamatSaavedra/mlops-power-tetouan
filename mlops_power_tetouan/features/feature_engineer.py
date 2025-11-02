import os
import joblib
import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import RobustScaler
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Encapsula la lógica de ingeniería de características:
    - Detección de columnas numéricas
    - Entrenamiento del scaler
    - Carga del scaler
    - Aplicación del escalado
    - Pipeline completo de preprocesamiento
    """

    def __init__(self, scaler_path: str = "models/scaler.pkl"):
        self.scaler_path = scaler_path
        self.scaler: Optional[RobustScaler] = None

    # -------------------------------------------------------------------------
    # 1) DETECCIÓN DE COLUMNAS
    # -------------------------------------------------------------------------
    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> List[str]:
        """Retorna las columnas numéricas válidas para escalado."""
        numeric_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col != "DateTime"
        ]
        logger.info(f"[FeatureEngineer] Columnas numéricas detectadas: {numeric_cols}")
        return numeric_cols

    # -------------------------------------------------------------------------
    # 2) ENTRENAR EL SCALER
    # -------------------------------------------------------------------------
    def fit_scaler(self, df: pd.DataFrame, cols: List[str]) -> RobustScaler:
        """Entrena un RobustScaler con las columnas indicadas y lo guarda en disco."""
        scaler = RobustScaler()
        scaler.fit(df[cols])

        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(scaler, self.scaler_path)

        logger.info(f"[FeatureEngineer] Scaler entrenado y guardado en {self.scaler_path}")
        self.scaler = scaler
        return scaler

    # -------------------------------------------------------------------------
    # 3) CARGAR SCALER
    # -------------------------------------------------------------------------
    def load_scaler(self) -> RobustScaler:
        """Carga el scaler desde disco."""
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"[FeatureEngineer] No se encontró scaler en {self.scaler_path}")

        self.scaler = joblib.load(self.scaler_path)
        logger.info(f"[FeatureEngineer] Scaler cargado desde {self.scaler_path}")
        return self.scaler

    # -------------------------------------------------------------------------
    # 4) APLICAR SCALER
    # -------------------------------------------------------------------------
    def apply_scaler(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Aplica un scaler previamente entrenado."""
        if self.scaler is None:
            raise ValueError("[FeatureEngineer] No hay scaler cargado ni entrenado.")

        df_scaled = df.copy()
        df_scaled[cols] = self.scaler.transform(df[cols])
        return df_scaled

    # -------------------------------------------------------------------------
    # 5) PIPELINE COMPLETO
    # -------------------------------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline principal para preprocesamiento:
        - Detecta columnas numéricas
        - Entrena scaler
        - Aplica escalado
        """
        logger.info("[FeatureEngineer] Iniciando preprocesamiento...")

        cols = self.get_numeric_columns(df)
        self.fit_scaler(df, cols)
        df_scaled = self.apply_scaler(df, cols)

        logger.info("[FeatureEngineer] Preprocesamiento finalizado.")
        return df_scaled
