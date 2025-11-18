import pandas as pd
from mlops_power_tetouan.features.feature_engineer import FeaturePipeline
# from mlops_power_tetouan.config import set_seed

# set_seed()

def test_fit_transform_output_shape():
    df = pd.read_csv("tests/fixtures/small_fixture.csv")
    target_cols = ["Zone 1 Power Consumption","Zone 2  Power Consumption","Zone 3  Power Consumption"]
    fp = FeaturePipeline(pipeline_path="models/test_pipeline.pkl", scaler_path="models/test_scaler.pkl")
    df_processed = fp.fit_transform(df, target_cols)
    # debe contener targets al final
    for t in target_cols:
        assert t in df_processed.columns
    # filas igual al input
    assert df_processed.shape[0] == df.shape[0]
