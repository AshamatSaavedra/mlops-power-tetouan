import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================
#  MÉTRICAS
# ============================

def evaluate(y_true, y_pred):
    """Devuelve MAE, RMSE y R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ============================
#  ENTRENAMIENTO DE MODELOS
# ============================

def train_single_zone(zone_name: str, X: pd.DataFrame, y: pd.Series,
                      output_dir: Path):
    """
    Entrena todos los modelos para una zona y guarda el mejor.
    """

    print(f"\n=== Entrenando modelos para {zone_name} ===")

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # -----------------------------------------------------------------
    # Modelo 1: Linear Regression
    # -----------------------------------------------------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_metrics = evaluate(y_test, lr_pred)

    # -----------------------------------------------------------------
    # Modelo 2: Ridge
    # -----------------------------------------------------------------
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=cv)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_metrics = evaluate(y_test, ridge_pred)

    # -----------------------------------------------------------------
    # Modelo 3: Lasso
    # -----------------------------------------------------------------
    lasso = LassoCV(alphas=np.logspace(-3, 3, 20), cv=cv, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_metrics = evaluate(y_test, lasso_pred)

    # -----------------------------------------------------------------
    # Modelo 4: Random Forest
    # -----------------------------------------------------------------
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }

    rf = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(
        rf,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=cv,
        verbose=0
    )
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    rf_pred = best_rf.predict(X_test)
    rf_metrics = evaluate(y_test, rf_pred)

    # =========================
    # Selección del mejor modelo
    # =========================
    metrics_map = {
        "linear": lr_metrics,
        "ridge": ridge_metrics,
        "lasso": lasso_metrics,
        "random_forest": rf_metrics,
    }

    best_model_name = min(metrics_map, key=lambda m: metrics_map[m][1])  # RMSE
    best_model = {
        "linear": lr,
        "ridge": ridge,
        "lasso": lasso,
        "random_forest": best_rf
    }[best_model_name]

    print(f"Mejor modelo para {zone_name}: {best_model_name.upper()}")
    mae, rmse, r2 = metrics_map[best_model_name]
    print(f"MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

    # =========================
    # Guardado de resultados
    # =========================
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{zone_name}_best_model.pkl"
    metrics_path = output_dir / f"{zone_name}_metrics.json"

    joblib.dump(best_model, model_path)

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

    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")


# ============================
#  MAIN
# ============================

def main():
    print("=== Cargando scaled.csv ===")
    df = pd.read_csv("data/processed/scaled.csv")

    features = ["Temperature", "Humidity", "Wind Speed",
                "general diffuse flows", "diffuse flows"]
    targets = {
        "zone1": "Zone 1 Power Consumption",
        "zone2": "Zone 2  Power Consumption",
        "zone3": "Zone 3  Power Consumption"
    }

    X = df[features]

    print("=== Iniciando entrenamiento de las tres zonas ===")

    for zone_key, col in targets.items():
        y = df[col]
        train_single_zone(zone_key, X, y, Path("models"))

    print("\n=== Entrenamiento completado ===")


if __name__ == "__main__":
    main()