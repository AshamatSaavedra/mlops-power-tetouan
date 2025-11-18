import pandas as pd
import os

def test_pipeline_end_to_end(tmp_path):
    raw = "tests/fixtures/small_fixture.csv"
    df = pd.read_csv(raw)

    processed = tmp_path / "processed.csv"
    from mlops_power_tetouan.features.feature_engineer import FeaturePipeline

    target_cols = [
        "Zone 1 Power Consumption",
        "Zone 2  Power Consumption",
        "Zone 3  Power Consumption"
    ]

    fp = FeaturePipeline(
        pipeline_path=str(tmp_path / "pipeline.pkl"),
        scaler_path=str(tmp_path / "scaler.pkl")
    )

    df_out = fp.fit_transform(df, target_cols)
    df_out.to_csv(processed, index=False)

    assert processed.exists()
    assert df_out.shape[0] == df.shape[0]
    assert all(col in df_out.columns for col in target_cols)
    assert df_out.drop(columns=target_cols).shape[1] == 14
