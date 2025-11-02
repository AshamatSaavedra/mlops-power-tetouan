from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional
import argparse
import os
import sys

# Encodings a probar si la lectura con utf-8 falla
COMMON_ENCODINGS: List[str] = ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"]

def load_csv(path: str, encodings: Optional[List[str]] = None, dtype: Optional[dict] = None) -> pd.DataFrame:
    """
    Carga un CSV intentando varios encodings en caso de error.

    Args:
        path: Ruta al archivo CSV.
        encodings: Lista de encodings a probar (por defecto COMMON_ENCODINGS).
        dtype: Diccionario opcional de tipos para pandas.read_csv.

    Returns:
        pd.DataFrame con el contenido del CSV.

    Raises:
        FileNotFoundError: si no existe el archivo.
        ValueError: si no se pudo leer el archivo con los encodings probados.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo no existe: {path}")

    encodings_to_try = encodings or COMMON_ENCODINGS
    last_exc = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc, dtype=dtype, low_memory=False)
            print(f"[load_csv] Archivo cargado correctamente con encoding='{enc}'. Shape: {df.shape}")
            return df
        except Exception as e:
            last_exc = e
            print(f"[load_csv] No se pudo leer con encoding='{enc}': {repr(e)}", file=sys.stderr)

    raise ValueError(f"No se pudo leer el archivo {path} con los encodings probados. Último error: {repr(last_exc)}")


def save_interim_copy(df: pd.DataFrame, out_path: str, index: bool = False) -> None:
    """
    Guarda una copia interim del DataFrame en CSV.

    Args:
        df: DataFrame a guardar.
        out_path: Ruta de salida.
        index: si guardar el índice.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=index)
    print(f"[save_interim_copy] Copia guardada en: {out_path} (index={index})")


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI sencillo para probar la carga.
    """
    parser = argparse.ArgumentParser(description="Cargar CSV raw del dataset Tetouan y opcionalmente guardar copia interim.")
    parser.add_argument("--input", "-i", required=True, help="Ruta al archivo raw CSV (ej: data/raw/power.csv)")
    parser.add_argument("--save_interim", "-s", required=False, help="Ruta para guardar una copia interim (ej: data/interim/power_raw_copy.csv)")
    parser.add_argument("--no_print_head", action="store_true", help="No imprimir head del DataFrame después de cargar")
    args = parser.parse_args(argv)

    try:
        df = load_csv(args.input)
    except Exception as e:
        print(f"[main] Error al cargar el CSV: {e}", file=sys.stderr)
        return 1

    if not args.no_print_head:
        print("\n[main] Head del DataFrame (primeras 5 filas):")
        print(df.head(5))
        print(f"[main] Columnas detectadas ({len(df.columns)}): {list(df.columns)}\n")

    if args.save_interim:
        try:
            save_interim_copy(df, args.save_interim)
        except Exception as e:
            print(f"[main] Error al guardar copia interim: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

