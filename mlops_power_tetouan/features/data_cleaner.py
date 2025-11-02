import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class DataCleaner:

    def __init__(
        self,
        power_cols: Optional[List[str]] = None,
        radiation_cols: Optional[List[str]] = None
    ):
        self.power_cols = power_cols or [
            "Zone 1 Power Consumption",
            "Zone 2  Power Consumption",
            "Zone 3  Power Consumption"
        ]

        self.radiation_cols = radiation_cols or [
            "general diffuse flows",
            "diffuse flows"
        ]

    # ---------------------------------------------------------
    # 1. Funciones de limpieza básicas
    # ---------------------------------------------------------

    def convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        if "DateTime" not in df.columns:
            raise ValueError("No existe la columna 'DateTime' en el dataset.")

        df = df.copy()
        df["DateTime"] = (
            df["DateTime"].astype(str).str.strip()
        )

        df["DateTime"] = pd.to_datetime(
            df["DateTime"], format="mixed", dayfirst=True, errors="coerce"
        )

        n_bad = df["DateTime"].isna().sum()
        logger.info(f"[convert_datetime] Fechas no convertibles: {n_bad}")

        return df

    def clean_numeric_columns(self, df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
        df = df.copy()
        exclude = exclude or ["DateTime"]

        for col in df.columns:
            if col in exclude:
                continue

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^0-9.\-]", "", regex=True)
            )

            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def remove_meteorological_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        conditions = {
            "Temperature": (df["Temperature"] < -10) | (df["Temperature"] > 80),
            "Humidity": (df["Humidity"] < 0) | (df["Humidity"] > 100),
            "Wind Speed": (df["Wind Speed"] < 0) | (df["Wind Speed"] > 50),
        }

        for col, cond in conditions.items():
            n = cond.sum()
            logger.info(f"[remove_meteorological_outliers] {col}: {n} valores anómalos")
            df.loc[cond, col] = np.nan

        return df

    def winsorize_power(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.power_cols:
            upper = df[col].quantile(0.99)
            n = (df[col] > upper).sum()
            df.loc[df[col] > upper, col] = upper
            logger.info(f"[winsorize_power] {col}: {n} valores ajustados (límite={upper:.2f})")

        return df

    def interpolate_meteorology(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["Temperature", "Humidity", "Wind Speed"]:
            df[col] = df[col].interpolate()
        return df

    def interpolate_power_by_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour"] = df["DateTime"].dt.hour
        for col in self.power_cols:
            df[col] = df.groupby("hour")[col].transform(lambda x: x.fillna(x.mean()))
        return df

    def handle_radiation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour"] = df["DateTime"].dt.hour
        df = df.sort_values("DateTime")

        before = len(df)
        df = df.dropna(subset=["DateTime"])
        dropped = before - len(df)
        if dropped:
            logger.info(f"[handle_radiation] Filas eliminadas antes del index: {dropped}")

        df = df.set_index("DateTime")
        is_night = (df["hour"] >= 19) | (df["hour"] < 5)

        for col in self.radiation_cols:
            df.loc[is_night & df[col].isna(), col] = 0
            df[col] = df[col].interpolate(method="time")
            df[col] = df[col].fillna(method="bfill").fillna(method="ffill")

        df = df.reset_index()
        return df

    # ---------------------------------------------------------
    # 2. Pipeline principal
    # ---------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("[clean] Iniciando limpieza del dataset...")

        df = self.convert_datetime(df)

        if "mixed_type_col" in df.columns:
            logger.info("[clean] Eliminando columna 'mixed_type_col'")
            df = df.drop(columns=["mixed_type_col"])

        df = self.clean_numeric_columns(df)
        df = self.remove_meteorological_outliers(df)
        df = df.drop_duplicates()

        df = self.winsorize_power(df)
        df = self.interpolate_meteorology(df)
        df = self.interpolate_power_by_hour(df)
        df = self.handle_radiation(df)

        df = df.dropna(subset=["DateTime"]).reset_index(drop=True)

        logger.info(f"[clean] Limpieza finalizada. Shape final: {df.shape}")
        return df
