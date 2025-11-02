import json
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import logging

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================

class ModelConfig:
    """Construye los modelos a evaluar."""

    @staticmethod
    def get_models(cv):
        return {
            "linear": LinearRegression(),
            "ridge": RidgeCV(alphas=np.logspace(-3, 3, 20), cv=cv),
            "lasso": LassoCV(alphas=np.logspace(-3, 3, 20), cv=cv, max_iter=10000),
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
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }

# ============================================================
# ZONE TRAINER
# ============================================================

class ZoneTrainer:
    """Entrena múltiples modelos para una zona, selecciona el mejor y registra en MLFlow."""

    def __init__(self, zone_name: str, output_dir: Path):
        self.zone_name = zone_name
        self.output_dir = output_dir

    @staticmethod
    def evaluate(y_true, y_pred) -> Tuple[float, float, float]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    def train(self, X: pd.DataFrame, y: pd.Series):
        logger.info(f"Entrenando modelos para {self.zone_name}")

        # Separar dataset
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
        )  # 0.25*0.8 = 0.2 → final split 60/20/20

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        models = ModelConfig.get_models(cv)

        metrics_map: Dict[str, Tuple[float, float, float]] = {}
        trained_models = {}

        # Entrenar cada modelo y loguear en MLFlow
        for name, model in models.items():
            logger.info(f"Entrenando {name} para {self.zone_name}")
            with mlflow.start_run(run_name=f"{self.zone_name}_{name}"):
                model.fit(X_train, y_train)
                y_pred_val = model.predict(X_val)
                mae, rmse, r2 = self.evaluate(y_val, y_pred_val)

                metrics_map[name] = (mae, rmse, r2)
                trained_models[name] = model

                # Log de MLFlow
                mlflow.log_params({"model_type": name})
                mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
                mlflow.sklearn.log_model(model, artifact_path="model")
                logger.info(f"{self.zone_name} | {name} -> MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

        # Mejor modelo por RMSE
        best_model_name = min(metrics_map, key=lambda m: metrics_map[m][1])
        best_model = trained_models[best_model_name]
        mae, rmse, r2 = metrics_map[best_model_name]

        logger.info(f"Mejor modelo para {self.zone_name}: {best_model_name} (RMSE={rmse:.3f})")

        # Guardado local
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.output_dir / f"{self.zone_name}_best_model.pkl"
        metrics_path = self.output_dir / f"{self.zone_name}_metrics.json"

        joblib.dump(best_model, model_path)
        with open(metrics_path, "w") as f:
            json.dump({
                "model": best_model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            }, f, indent=4)

        logger.info(f"Modelo guardado en: {model_path}")
        logger.info(f"Métricas guardadas en: {metrics_path}")


# ============================================================
# MODELING PIPELINE
# ============================================================

class ModelingPipeline:
    """Orquesta entrenamiento de todas las zonas"""

    def __init__(self, features: List[str], target_map: Dict[str, str], output_dir: str = "models"):
        self.features = features
        self.target_map = target_map
        self.output_dir = Path(output_dir)

    def run(self, df: pd.DataFrame):
        X = df[self.features]
        for zone_key, target_col in self.target_map.items():
            y = df[target_col]
            trainer = ZoneTrainer(zone_key, self.output_dir)
            trainer.train(X, y)


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("Cargando data/processed/scaled.csv")
    df = pd.read_csv("data/processed/scaled.csv")


    features = [
        "Temperature", "Humidity", "Wind Speed",
        "general diffuse flows", "diffuse flows", "hour",
        "day_of_week", "day_of_year", "is_weekend", "hour_cos",
        "hour_sin", "month", "temp_x_hum", "radiation_total"
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
