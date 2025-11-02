import os
import pandas as pd
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Lector robusto de CSVs.
    - Prueba múltiples encodings.
    - Soporta dtype opcional.
    - Compatible con DVC y pipelines reproducibles.
    """

    DEFAULT_ENCODINGS: List[str] = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "iso-8859-1",
        "cp1252",
    ]

    def __init__(
        self,
        encodings: Optional[List[str]] = None,
        dtypes: Optional[Dict[str, str]] = None,
    ):
        self.encodings = encodings or self.DEFAULT_ENCODINGS
        self.dtypes = dtypes

    def load_csv(self, path: str) -> pd.DataFrame:
        """Carga un CSV probando múltiples encodings."""

        if not os.path.exists(path):
            raise FileNotFoundError(f"[DataLoader] Archivo no encontrado: {path}")

        last_error = None

        for enc in self.encodings:
            try:
                df = pd.read_csv(path, encoding=enc, dtype=self.dtypes, low_memory=False)
                logger.info(
                    f"[DataLoader] Cargado con encoding='{enc}'. Shape={df.shape}"
                )
                return df
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[DataLoader] Error con encoding '{enc}': {repr(e)}"
                )

        raise ValueError(
            f"[DataLoader] Ningún encoding funcionó para {path}. Último error: {repr(last_error)}"
        )


class DataSaver:
    """Utilidad para guardar copias interim y finales con seguridad."""

    @staticmethod
    def save_csv(df: pd.DataFrame, out_path: str, index: bool = False) -> None:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        df.to_csv(out_path, index=index)
        logger.info(f"[DataSaver] CSV guardado en {out_path}. Shape={df.shape}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Carga un CSV raw y opcionalmente guarda copia interim")
    parser.add_argument("--input", "-i", required=True, help="Ruta del CSV raw")
    parser.add_argument("--output", "-o", required=False, help="Ruta para guardar copia interim")
    parser.add_argument("--no_head", action="store_true", help="No imprimir head del DF")
    args = parser.parse_args()

    loader = DataLoader()
    saver = DataSaver()

    df = loader.load_csv(args.input)

    if not args.no_head:
        print(df.head(5))
        print(f"Columnas: {list(df.columns)}")

    if args.output:
        saver.save_csv(df, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
