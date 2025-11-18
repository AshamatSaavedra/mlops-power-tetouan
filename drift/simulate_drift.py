# mlops_power_tetouan/drift/simulate_drift.py

import pandas as pd
import numpy as np

def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica cambios controlados en la distribución para simular Data Drift.
    Mantiene la estructura del dataset para no romper el pipeline.
    """

    drifted = df.copy()

    # --- Drift 1: Cambio climático (temperaturas +6 ºC)
    drifted["Temperature"] = drifted["Temperature"] + 6

    # --- Drift 2: Humedad más dispersa (20% más varianza)
    drifted["Humidity"] = drifted["Humidity"] * np.random.normal(1.0, 0.2, len(df))

    # --- Drift 3: Viento con mayor ruido
    drifted["Wind Speed"] = drifted["Wind Speed"] + np.random.normal(0, 0.4, len(df))

    # --- Drift 4: Estacionalidad opuesta (radiación más baja)
    drifted["general diffuse flows"] *= 0.65
    drifted["diffuse flows"] *= 0.65


    return drifted
