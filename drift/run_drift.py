from evaluate_drift import evaluate_drift

model_paths = {
    "zone1": "models/model_zone1.pkl",
    "zone2": "models/model_zone2.pkl",
    "zone3": "models/model_zone3.pkl",
}

target_cols = {
    "zone1": "Zone 1 Power Consumption",
    "zone2": "Zone 2  Power Consumption",
    "zone3": "Zone 3  Power Consumption",
}

report = evaluate_drift(
    raw_csv="data/raw/power_tetouan_city_modified.csv",
    pipeline_path="models/feature_pipeline.pkl",
    scaler_path="models/scaler.pkl",
    model_paths=model_paths,
    target_cols=target_cols,
)

print(report)
