import argparse
from mlops_power_tetouan.dataset import DataLoader, DataSaver


def main():
    parser = argparse.ArgumentParser(description="Pipeline: carga del CSV raw")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    loader = DataLoader()
    saver = DataSaver()

    df = loader.load_csv(args.input)
    saver.save_csv(df, args.output)

    print(f"[run_load] Guardado dataset interim → {args.output}")


if __name__ == "__main__":
    main()
