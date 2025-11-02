from pathlib import Path
from mlops_power_tetouan.features import pca as pca_module
import sys

def main():
    input_path = Path("data/processed/scaled.csv")
    output_path = Path("data/processed/pca_components.csv")
    model_path = Path("models/pca_model.pkl")
    n_components = 3

    # Llamamos a la función main del módulo pca pasándole los argumentos
    argv = [
        "--input", str(input_path),
        "--output", str(output_path),
        "--model", str(model_path),
        "--n_components", str(n_components)
    ]
    print("[run_pca] Ejecutando PCA con args:", argv)
    try:
        pca_module.main(argv)
    except SystemExit as e:
        # pca_module.main usa SystemExit para terminar; propagamos 0 como éxito
        if e.code != 0:
            print(f"[run_pca] PCA terminó con error: {e}", file=sys.stderr)
            raise
    print("[run_pca] PCA ejecutado correctamente.")

if __name__ == "__main__":
    main()