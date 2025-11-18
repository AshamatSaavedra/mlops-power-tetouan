import pandas as pd
from mlops_power_tetouan.serving.app import load_model_for_zone
from simulate_drift import simulate_drift
from mlops_power_tetouan.features.feature_engineer import FeaturePipeline
from mlops_power_tetouan.modeling.modeling import ZoneTrainer


def evaluate_drift(
    raw_csv: str,
    pipeline_path: str,
    scaler_path: str,
    model_paths: dict,  # {zone1: path, zone2: path, ...}
    target_cols: list,
):
    print("=== Cargando dataset base ===")
    df = pd.read_csv(raw_csv)
    num_cols = [
        "Temperature","Humidity","Wind Speed",
        "general diffuse flows","diffuse flows"
    ]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    for zone, col in target_cols.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(target_cols.values()))
    
    print("=== Simulando drift ===")
    df_drift = simulate_drift(df)

    print("=== Cargando pipeline ===")
    fp = FeaturePipeline(pipeline_path, scaler_path)

    print("=== Transformando datos base ===")
    df_base_ft = fp.transform(df)

    print("=== Transformando datos con drift ===")
    df_drift_ft = fp.transform(df_drift)

    report = {}

    for zone, model_path in model_paths.items():
        print(f"\n--- Evaluando {zone} ---")
        model = load_model_for_zone(zone)
        model_features = model.best_estimator_.feature_names_in_

        df_base_ft = df_base_ft.reindex(columns=model_features)
        df_drift_ft = df_drift_ft.reindex(columns=model_features)

        y_true = df[target_cols[zone]].values

        preds_base = model.predict(df_base_ft)
        preds_drift = model.predict(df_drift_ft)

        mae_base, rmse_base, r2_base = ZoneTrainer.evaluate(y_true, preds_base)
        mae_drift, rmse_drift, r2_drift = ZoneTrainer.evaluate(y_true, preds_drift)

        report[zone] = {
            "MAE_base": mae_base,
            "RMSE_base": rmse_base,
            "R2_base": r2_base,
            "MAE_drift": mae_drift,
            "RMSE_drift": rmse_drift,
            "R2_drift": r2_drift,
            "MAE_change_pct": (mae_drift - mae_base) / mae_base * 100,
            "RMSE_change_pct": (rmse_drift - rmse_base) / rmse_base * 100,
        }

        print(report[zone])

    return report
