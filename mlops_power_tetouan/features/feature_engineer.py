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


# -------------------------------------------------------------------
# 1) FeatureGenerator (sin lags, sin rolling)
# -------------------------------------------------------------------
class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Crea variables temporales + interacciones.
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

            # --- time features ---
            df["hour"] = df[self.datetime_col].dt.hour
            df["day_of_week"] = df[self.datetime_col].dt.dayofweek
            df["month"] = df[self.datetime_col].dt.month
            df["day_of_year"] = df[self.datetime_col].dt.dayofyear
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

            # Representación cíclica
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # --- interaction features ---
        if {"Temperature", "Humidity"}.issubset(df.columns):
            df["temp_x_hum"] = df["Temperature"] * df["Humidity"]

        if {"general diffuse flows", "diffuse flows"}.issubset(df.columns):
            df["radiation_total"] = \
                df["general diffuse flows"].fillna(0) + df["diffuse flows"].fillna(0)

        return df


# -------------------------------------------------------------------
# 2) FeaturePipeline (pipeline sklearn con ColumnTransformer)
# -------------------------------------------------------------------
class FeaturePipeline:
    """
    Construye un pipeline sklearn:
      FeatureGenerator -> Imputación -> Escalado
    Guarda también scaler y pipeline completo.
    """

    def __init__(
        self,
        pipeline_path: str = "models/feature_pipeline.pkl",
        scaler_path: str = "models/scaler.pkl"
    ):
        self.pipeline_path = pipeline_path
        self.scaler_path = scaler_path
        self.pipeline: Optional[Pipeline] = None

    def _get_numeric_features(self, df: pd.DataFrame, exclude: List[str]):
        return [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude
        ]

    def build_pipeline(self, df: pd.DataFrame, target_cols: List[str]):
        feature_gen = FeatureGenerator()

        # Previo: transform para conocer columnas generadas
        df_tmp = feature_gen.transform(df)

        exclude = ["DateTime"] + target_cols
        numeric_cols = self._get_numeric_features(df_tmp, exclude)

        passthrough_cols = ["DateTime", "day_of_week", "month",
                            "day_of_year", "is_weekend"]
        passthrough_cols = [c for c in passthrough_cols if c in df_tmp.columns]

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

        self.pipeline = Pipeline(steps=[
            ("feature_gen", feature_gen),
            ("preprocess", preproc)
        ])

        return self.pipeline

    def fit_transform(self, df: pd.DataFrame, target_cols: List[str]):
        pipeline = self.build_pipeline(df, target_cols)
        X = pipeline.fit_transform(df)

        os.makedirs(os.path.dirname(self.pipeline_path), exist_ok=True)
        joblib.dump(pipeline, self.pipeline_path)

        # Guardar scaler
        try:
            scaler = pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"]
            joblib.dump(scaler, self.scaler_path)
        except Exception:
            pass

        # nombres de columnas exactos
        final_cols = pipeline.named_steps["preprocess"].get_feature_names_out()

        # Quitar prefijos 'num__' y 'pass__'
        final_cols = [c.replace("num__", "").replace("pass__", "") for c in final_cols]

        df_out = pd.DataFrame(X, columns=final_cols)

        # agregar targets
        df_out = pd.concat([df_out.reset_index(drop=True), df[target_cols].reset_index(drop=True)], axis=1)

        return df_out

    def transform(self, df: pd.DataFrame):
        if self.pipeline is None:
            if os.path.exists(self.pipeline_path):
                self.pipeline = joblib.load(self.pipeline_path)
            else:
                raise ValueError("Pipeline no entrenado.")

        X = self.pipeline.transform(df)

        feature_gen = self.pipeline.named_steps["feature_gen"]
        df_tmp = feature_gen.transform(df)

        numeric_cols = self._get_numeric_features(df_tmp, exclude=["DateTime"])
        passthrough_cols = ["DateTime", "day_of_week", "month",
                            "day_of_year", "is_weekend"]
        passthrough_cols = [c for c in passthrough_cols if c in df_tmp.columns]

        final_cols = numeric_cols + passthrough_cols

        return pd.DataFrame(X, columns=final_cols)
