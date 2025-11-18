import pandas as pd
import os
from mlops_power_tetouan.pipelines.preprocess_n_save import main as preprocess_main
from mlops_power_tetouan.modeling.modeling import main as modeling_main
# from mlops_power_tetouan.config import set_seed

# set_seed()

def test_pipeline_end_to_end(tmp_path):
    # usar fixture como raw (pequeño)
    raw = "tests/fixtures/small_fixture.csv"
    interim = tmp_path / "data/interim.csv"
    processed = tmp_path / "data/processed.csv"

    # copiar fixture
    df = pd.read_csv(raw)
    interim.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(interim, index=False)

    # Ejecutar preprocessing script similar al pipeline real:
    # llamar FeaturePipeline directamente para evitar DVC en tests
    from mlops_power_tetouan.features.feature_engineer import FeaturePipeline
    target_cols = ["Zone 1 Power Consumption","Zone 2  Power Consumption","Zone 3  Power Consumption"]
    fp = FeaturePipeline(pipeline_path=str(tmp_path/"feature_pipeline.pkl"),
                         scaler_path=str(tmp_path/"scaler.pkl"))
    df_processed = fp.fit_transform(df, target_cols)
    df_processed.to_csv(processed, index=False)

    # cargar modelo de prueba si existe -> en este test solo comprobamos que pipeline produce csv
    assert os.path.exists(processed)
    assert df_processed.shape[0] == df.shape[0]
