import argparse
from pathlib import Path
import pandas as pd

from mlops_power_tetouan.features.feature_engineer import FeaturePipeline


def main():
    parser = argparse.ArgumentParser(description="Pipeline: feature engineering + escalado")
    parser.add_argument("--input", "-i", required=True, help="CSV limpio de entrada")
    parser.add_argument("--output", "-o", required=True, help="CSV procesado de salida")
    parser.add_argument("--pipeline_path", default="models/feature_pipeline.pkl",
                        help="Ruta donde guardar el pipeline entrenado")
    parser.add_argument("--scaler_path", default="models/scaler.pkl",
                        help="Ruta donde guardar el scaler")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"[preprocess] ERROR: no existe el archivo limpio: {input_path}")

    df = pd.read_csv(input_path)

    # =====================
    # Target columns
    # =====================
    target_cols = [
        "Zone 1 Power Consumption",
        "Zone 2  Power Consumption",
        "Zone 3  Power Consumption"
    ]

    # =====================
    # Construcción pipeline
    # =====================
    pipeline = FeaturePipeline(
        pipeline_path=args.pipeline_path,
        scaler_path=args.scaler_path
    )

    print("[preprocess] Ejecutando feature engineering + escalado...")

    # Fit + Transform
    df_processed = pipeline.fit_transform(df, target_cols=target_cols)

    # =====================
    # Guardar resultado
    # =====================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)

    print(f"[preprocess] Dataset procesado guardado en {output_path}")
    print(f"[preprocess] Pipeline guardado en {args.pipeline_path}")
    print(f"[preprocess] Scaler guardado en {args.scaler_path}")


if __name__ == "__main__":
    main()
