import argparse
import pandas as pd
import os

from mlops_power_tetouan.features.data_cleaner import DataCleaner


def main():
    parser = argparse.ArgumentParser(description="Pipeline: limpieza completa")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    cleaner = DataCleaner()
    df_clean = cleaner.clean(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_clean.to_csv(args.output, index=False)

    print(f"[run_cleaning] Dataset limpio guardado en {args.output}")


if __name__ == "__main__":
    main()
