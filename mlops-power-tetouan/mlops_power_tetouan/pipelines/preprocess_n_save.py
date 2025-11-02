from pathlib import Path
import pandas as pd
from mlops_power_tetouan.features import preprocess

# Rutas
clean_path = Path("data/interim/power_tetouan_city_modified_clean.csv")    # salida de src.data.clean
scaled_path = Path("data/processed/scaled.csv")

if not clean_path.exists():
    raise SystemExit(f"Error: no existe {clean_path}. Primero corre el proceso de limpieza (src.data.clean)")

# Cargar dataset limpio
df_clean = pd.read_csv(clean_path)

# Ejecutar preprocesamiento (entrena y guarda scaler en models/scaler.pkl)
df_scaled = preprocess(df_clean, scaler_path="models/scaler.pkl")

# Guardar el dataset escalado
scaled_path.parent.mkdir(parents=True, exist_ok=True)
df_scaled.to_csv(scaled_path, index=False)
print(f"Scaled CSV guardado en: {scaled_path}")