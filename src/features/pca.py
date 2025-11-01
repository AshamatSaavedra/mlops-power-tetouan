import pandas as pd
from sklearn.decomposition import PCA
import joblib
from pathlib import Path


def apply_pca(df: pd.DataFrame, n_components: int = 3):
    """
    Aplica PCA sobre un dataframe ya escalado.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe escalado proveniente de preprocessing.
    n_components : int
        Número de componentes principales a calcular.

    Retorna
    -------
    df_pca : pd.DataFrame
        Dataframe con los componentes principales.
    pca_model : sklearn.decomposition.PCA
        Modelo PCA entrenado.
    """

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    X = df[numeric_cols]

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)

    df_pca = pd.DataFrame(
        components,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # Si existe columna DateTime, conservarla
    if 'DateTime' in df.columns:
        df_pca['DateTime'] = df['DateTime'].reset_index(drop=True)

    return df_pca, pca

def main(argv=None):
    """
    Punto de entrada CLI para ejecutar el PCA desde consola o script.
    Soporta argumentos:
      --input path/to/scaled.csv
      --output path/to/pca_components.csv
      --model path/to/pca_model.pkl
      --n_components 3
    """

    import argparse

    parser = argparse.ArgumentParser(description="Aplicar PCA al dataset escalado.")
    parser.add_argument("--input", type=str, required=True, help="Ruta del archivo escalado CSV.")
    parser.add_argument("--output", type=str, required=True, help="Ruta para guardar los componentes PCA.")
    parser.add_argument("--model", type=str, required=True, help="Ruta para guardar el modelo PCA.")
    parser.add_argument("--n_components", type=int, default=3, help="Número de componentes PCA.")

    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    print(f"[PCA] Cargando dataset desde: {input_path}")
    df_scaled = pd.read_csv(input_path)

    print(f"[PCA] Aplicando PCA con n_components={args.n_components}...")
    df_pca, pca_model = apply_pca(df_scaled, n_components=args.n_components)

    print(f"[PCA] Guardando componentes PCA en: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_pca.to_csv(output_path, index=False)

    print(f"[PCA] Guardando modelo PCA en: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca_model, model_path)

    print("[PCA] Proceso completado correctamente.")
    
if __name__ == "__main__":
    input_path = Path("data/processed/scaled.csv")
    output_pca_path = Path("data/processed/pca_components.csv")
    pca_model_path = Path("models/pca_model.pkl")

    print(f"Cargando dataset escalado desde: {input_path}")
    df_scaled = pd.read_csv(input_path)

    print("Aplicando PCA...")
    df_pca, pca_model = apply_pca(df_scaled, n_components=3)

    print(f"Guardando componentes en: {output_pca_path}")
    df_pca.to_csv(output_pca_path, index=False)

    print(f"Guardando modelo PCA en: {pca_model_path}")
    pca_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca_model, pca_model_path)

    print("PCA completado con éxito.")
