import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import logging

logger = logging.getLogger(__name__)


# ============================================================
#                     CONFIGURACIÓN DE MODELOS
# ============================================================

class ModelConfig:
    """Contiene y construye los modelos a evaluar."""

    @staticmethod
    def get_models(cv):
        return {
            "linear": LinearRegression(),

            "ridge": RidgeCV(
                alphas=np.logspace(-3, 3, 20),
                cv=cv
            ),

            "lasso": LassoCV(
                alphas=np.logspace(-3, 3, 20),
                cv=cv,
                max_iter=10_000
            ),

            "random_forest": GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid={
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5]
                },
                scoring="neg_mean_squared_error",
                cv=cv,
                n_jobs=-1,
                verbose=0
            )
        }


# ============================================================
#                           ZONE TRAINER
# ============================================================

class ZoneTrainer:
    """Entrena múltiples modelos para una zona, selecciona el mejor y lo guarda."""

    def __init__(self, zone_name: str, output_dir: Path):
        self.zone_name = zone_name
        self.output_dir = output_dir

    # --------------------------------------------------------
    #      MÉTRICAS
    # --------------------------------------------------------
    @staticmethod
    def evaluate(y_true, y_pred) -> Tuple[float, float, float]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    # --------------------------------------------------------
    #      ENTRENAMIENTO COMPLETO
    # --------------------------------------------------------
    def train(self, X: pd.DataFrame, y: pd.Series):
        logger.info(f"Entrenando modelos para {self.zone_name}")

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        models = ModelConfig.get_models(cv)

        metrics_map: Dict[str, Tuple[float, float, float]] = {}
        trained_models = {}

        # Entrenar todos
        for name, model in models.items():
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            metrics = self.evaluate(y_test, pred)

            metrics_map[name] = metrics
            trained_models[name] = model

            logger.info(
                f"{self.zone_name} | Modelo {name} -> "
                f"MAE={metrics[0]:.3f}, RMSE={metrics[1]:.3f}, R²={metrics[2]:.3f}"
            )

        # ----------------------------------------------------
        # SELECCIÓN DEL MEJOR MODELO (por RMSE)
        # ----------------------------------------------------
        best_model_name = min(metrics_map, key=lambda m: metrics_map[m][1])
        best_model = trained_models[best_model_name]

        logger.info(
            f"Mejor modelo para {self.zone_name}: {best_model_name.upper()}"
        )

        # ----------------------------------------------------
        # GUARDADO
        # ----------------------------------------------------
        self.output_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.output_dir / f"{self.zone_name}_best_model.pkl"
        metrics_path = self.output_dir / f"{self.zone_name}_metrics.json"

        joblib.dump(best_model, model_path)

        mae, rmse, r2 = metrics_map[best_model_name]

        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "model": best_model_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                },
                f,
                indent=4
            )

        logger.info(f"Modelo guardado en: {model_path}")
        logger.info(f"Métricas guardadas en: {metrics_path}")


# ============================================================
#                        MODELING PIPELINE
# ============================================================

class ModelingPipeline:
    """Orquesta el entrenamiento para todas las zonas"""

    def __init__(self, features, target_map: Dict[str, str], output_dir: str = "models"):
        self.features = features
        self.target_map = target_map
        self.output_dir = Path(output_dir)

    def run(self, df: pd.DataFrame):
        X = df[self.features]

        for zone_key, target_col in self.target_map.items():
            trainer = ZoneTrainer(zone_key, self.output_dir)
            y = df[target_col]
            trainer.train(X, y)


# ============================================================
#                               MAIN
# ============================================================

def main():
    logger.info("Cargando data/processed/scaled.csv")
    df = pd.read_csv("data/processed/scaled.csv")

    features = [
        "Temperature", "Humidity", "Wind Speed",
        "general diffuse flows", "diffuse flows"
    ]

    targets = {
        "zone1": "Zone 1 Power Consumption",
        "zone2": "Zone 2  Power Consumption",
        "zone3": "Zone 3  Power Consumption",
    }

    pipeline = ModelingPipeline(features, targets, output_dir="models")

    logger.info("Iniciando entrenamiento de las zonas...")
    pipeline.run(df)
    logger.info("Entrenamiento completado.")


if __name__ == "__main__":
    main()
