import argparse
from pathlib import Path
import pandas as pd

# Importar desde la nueva estructura CCDS
from mlops_power_tetouan.dataset import load_csv
from mlops_power_tetouan.features.cleaning import clean_dataset

def main():
    parser = argparse.ArgumentParser(description="Pipeline de limpieza Tetouan")
    parser.add_argument("--input", "-i", required=True, help="CSV raw de entrada")
    parser.add_argument("--output", "-o", required=True, help="CSV limpio de salida")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[run_cleaning] Leyendo archivo: {input_path}")
    df = load_csv(str(input_path))

    print("[run_cleaning] Ejecutando limpieza completa…")
    df_clean = clean_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print(f"[run_cleaning] Archivo limpio guardado en: {output_path} — shape={df_clean.shape}")

if __name__ == "__main__":
    main()
