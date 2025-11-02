import argparse
import pandas as pd
import numpy as np
from typing import List
import os

# ----------------------------------------------------------------------
# 1. Helpers
# ----------------------------------------------------------------------

def convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte la columna DateTime a datetime usando formato flexible."""

    df = df.copy()

    if "DateTime" not in df.columns:
        raise ValueError("El DataFrame no contiene la columna 'DateTime'.")

    df["DateTime"] = (
        df["DateTime"]
        .astype(str)
        .str.strip()
    )

    df["DateTime"] = pd.to_datetime(df["DateTime"], format="mixed",
                                    dayfirst=True, errors="coerce")

    print(f"[convert_datetime] Fechas no convertibles: {df['DateTime'].isna().sum()}")
    return df


def clean_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Limpia columnas numéricas eliminando texto, caracteres extraños y convirtiendo a float.
    """
    df = df.copy()
    exclude = exclude or []

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


def remove_meteorological_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cond_temp = (df["Temperature"] < -10) | (df["Temperature"] > 80)
    cond_hum = (df["Humidity"] < 0) | (df["Humidity"] > 100)
    cond_wind = (df["Wind Speed"] < 0) | (df["Wind Speed"] > 50)

    print(f"[remove_meteorological_outliers] Anómalos Temperature: {cond_temp.sum()}")
    print(f"[remove_meteorological_outliers] Anómalos Humidity: {cond_hum.sum()}")
    print(f"[remove_meteorological_outliers] Anómalos Wind Speed: {cond_wind.sum()}")

    df.loc[cond_temp, "Temperature"] = np.nan
    df.loc[cond_hum, "Humidity"] = np.nan
    df.loc[cond_wind, "Wind Speed"] = np.nan

    return df


def winsorize_power_consumption(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()

    for col in cols:
        upper_limit = df[col].quantile(0.99)
        original_outliers = (df[col] > upper_limit).sum()

        df.loc[df[col] > upper_limit, col] = upper_limit

        print(f"[winsorize_power_consumption] Winsorizados {original_outliers} valores en {col} (límite={upper_limit:.2f})")

    return df


def interpolate_meteorological(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Temperature", "Humidity", "Wind Speed"]:
        df[col] = df[col].interpolate()
    return df


def interpolate_power_by_hour(df: pd.DataFrame, power_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["DateTime"].dt.hour
    for col in power_cols:
        df[col] = df.groupby("hour")[col].transform(lambda x: x.fillna(x.mean()))
    return df


def handle_radiation(df: pd.DataFrame, rad_cols: List[str]) -> pd.DataFrame:
    """
    - Radiación nocturna → 0
    - Radiación diurna → interpolación temporal
    """
    df = df.copy()
    # Asegurar que hour exista ANTES de set_index
    if "hour" not in df.columns:
        df["hour"] = df["DateTime"].dt.hour

    # Orden temporal
    df = df.sort_values("DateTime")

    # Eliminamos filas donde DateTime sea NaT ANTES de poner index
    before = len(df)
    df = df.dropna(subset=["DateTime"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[handle_radiation] Filas eliminadas por NaT antes de interpolar: {dropped}")

    df = df.set_index("DateTime")

    is_night = (df["hour"] >= 19) | (df["hour"] < 5)

    for col in rad_cols:
        # Radiación nocturna → 0
        df.loc[is_night & df[col].isna(), col] = 0

        # Interpolación temporal para el día
        df[col] = df[col].interpolate(method="time")

        # Fill residuals
        df[col] = df[col].fillna(method="bfill").fillna(method="ffill")

    df = df.reset_index()
    return df


# ----------------------------------------------------------------------
# 2. Pipeline principal
# ----------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica toda la secuencia de limpieza del proyecto.
    """
    df = convert_datetime(df)

    # Si existe, eliminar mixed_type_col antes de procesar numéricos
    if 'mixed_type_col' in df.columns:
        print("[clean_dataset] Eliminando columna 'mixed_type_col' (excesivos NaNs/inconsistencias).")
        df = df.drop(columns=['mixed_type_col'])

    df = clean_numeric_columns(df, exclude=["DateTime"])

    df = remove_meteorological_outliers(df)

    df = df.drop_duplicates()

    # Winsorización de las zonas
    power_cols = [
        "Zone 1 Power Consumption",
        "Zone 2  Power Consumption",
        "Zone 3  Power Consumption"
    ]
    df = winsorize_power_consumption(df, power_cols)

    # Interpolación meteorológica
    df = interpolate_meteorological(df)

    # Interpolación por hora
    df = interpolate_power_by_hour(df, power_cols)

    # Manejo especial de radiación
    df["hour"] = df["DateTime"].dt.hour
    df = handle_radiation(df, ["general diffuse flows", "diffuse flows"])

    # Drop DateTime nulo
    df = df.dropna(subset=["DateTime"])

    df = df.reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 3. CLI
# ----------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description="Limpieza completa del dataset Tetouan.")
    parser.add_argument("--input", "-i", required=True, help="CSV raw de entrada")
    parser.add_argument("--output", "-o", required=True, help="Ruta del CSV limpio")

    args = parser.parse_args(args)

    print("[clean.py] Cargando dataset...")
    df = pd.read_csv(args.input)

    print("[clean.py] Procesando dataset...")
    df_clean = clean_dataset(df)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df_clean.to_csv(args.output, index=False)
    print(f"[clean.py] Dataset limpio guardado en {args.output}. Shape final: {df_clean.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())