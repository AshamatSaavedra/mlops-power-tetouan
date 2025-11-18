# mlops_power_tetouan/features/feature_engineering.py

import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Genera variables temporales + interacciones.
    """

    def __init__(self, datetime_col: str = "DateTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Aseguramos DateTime
        if self.datetime_col in df.columns:
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], errors="coerce")

            df = df.sort_values(self.datetime_col).reset_index(drop=True)

            # Time-based features
            df["hour"] = df[self.datetime_col].dt.hour
            df["day_of_week"] = df[self.datetime_col].dt.dayofweek
            df["month"] = df[self.datetime_col].dt.month
            df["day_of_year"] = df[self.datetime_col].dt.dayofyear
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

            # Cyclic encoding
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Interaction features
        if {"Temperature", "Humidity"}.issubset(df.columns):
            df["temp_x_hum"] = df["Temperature"] * df["Humidity"]

        if {"general diffuse flows", "diffuse flows"}.issubset(df.columns):
            df["radiation_total"] = (
                df["general diffuse flows"].fillna(0)
                + df["diffuse flows"].fillna(0)
            )

        return df


# -------------------------------------------------------------------
# FeaturePipeline — versión corregida (sin columnas duplicadas)
# -------------------------------------------------------------------
class FeaturePipeline:
    """
    Pipeline:
      FeatureGenerator -> Imputación -> Escalado
    """

    def __init__(
        self,
        pipeline_path: str = "models/feature_pipeline.pkl",
        scaler_path: str = "models/scaler.pkl"
    ):
        self.pipeline_path = pipeline_path
        self.scaler_path = scaler_path
        self.pipeline: Optional[Pipeline] = None

    def build_pipeline(self, df: pd.DataFrame, target_cols: List[str]):

        feature_gen = FeatureGenerator()

        # Pre-generamos features para saber columnas disponibles
        df_tmp = feature_gen.transform(df)

        # ---- DEFINICIÓN FIJA DE FEATURES (14 EXACTAS) ----
        numeric_cols = [
            "Temperature", "Humidity", "Wind Speed",
            "general diffuse flows", "diffuse flows",
            "hour", "hour_sin", "hour_cos",
            "temp_x_hum", "radiation_total"
        ]

        passthrough_cols = [
            "day_of_week", "day_of_year", "is_weekend", "month"
        ]

        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        preproc = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numeric_cols),
                ("pass", "passthrough", passthrough_cols),
            ],
            remainder="drop"
        )

        # Construcción del pipeline final
        self.pipeline = Pipeline(steps=[
            ("feature_gen", feature_gen),   # añade columnas necesarias
            ("preprocess", preproc),        # solo usa nuestras listas fijas
        ])

        return self.pipeline

    def fit_transform(self, df: pd.DataFrame, target_cols: List[str]):
        pipeline = self.build_pipeline(df, target_cols)
        X = pipeline.fit_transform(df)

        os.makedirs(os.path.dirname(self.pipeline_path), exist_ok=True)
        joblib.dump(pipeline, self.pipeline_path)

        # Guardar scaler
        scaler = pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"]
        joblib.dump(scaler, self.scaler_path)

        # Recuperar nombres reales del CT
        final_cols = pipeline.named_steps["preprocess"].get_feature_names_out()
        final_cols = [c.replace("num__", "").replace("pass__", "") for c in final_cols]

        df_out = pd.DataFrame(X, columns=final_cols)

        # Agregar targets
        df_out = pd.concat(
            [df_out.reset_index(drop=True), df[target_cols].reset_index(drop=True)],
            axis=1
        )

        return df_out

    def transform(self, df: pd.DataFrame):
        if self.pipeline is None:
            if os.path.exists(self.pipeline_path):
                self.pipeline = joblib.load(self.pipeline_path)
            else:
                raise ValueError("Pipeline no entrenado.")

        X = self.pipeline.transform(df)

        final_cols = self.pipeline.named_steps["preprocess"].get_feature_names_out()
        final_cols = [c.replace("num__", "").replace("pass__", "") for c in final_cols]

        return pd.DataFrame(X, columns=final_cols)
