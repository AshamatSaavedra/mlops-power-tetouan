import argparse
from pathlib import Path
import pandas as pd

from mlops_power_tetouan.features.feature_engineer import FeatureEngineer


def main():
    parser = argparse.ArgumentParser(description="Pipeline: preprocesamiento y escalado")
    parser.add_argument("--input", "-i", required=True, help="CSV limpio de entrada")
    parser.add_argument("--output", "-o", required=True, help="CSV procesado de salida (scaled)")
    parser.add_argument("--scaler_path", "-s", default="models/scaler.pkl",
                        help="Ruta donde guardar/cargar el scaler")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"[run_preprocess] No existe el archivo limpio: {input_path}")

    df = pd.read_csv(input_path)

    # ✅ patrón OOP correcto 
    feat_eng = FeatureEngineer()

    df_scaled = feat_eng.preprocess(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(output_path, index=False)

    print(f"[run_preprocess] Dataset escalado guardado en {output_path}")


if __name__ == "__main__":
    main()
