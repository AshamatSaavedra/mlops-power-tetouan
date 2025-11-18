import pandas as pd
from mlops_power_tetouan.features.feature_engineer import FeatureGenerator
# from mlops_power_tetouan.config import set_seed

# set_seed()

def test_feature_generator_basic():
    df = pd.read_csv("tests/fixtures/small_fixture.csv")
    fg = FeatureGenerator(datetime_col="DateTime")
    out = fg.transform(df)
    # comprobar columnas temporales
    assert "hour" in out.columns
    assert "day_of_week" in out.columns
    assert "temp_x_hum" in out.columns
    assert "radiation_total" in out.columns
    # valores no nulos en tiempo convertido
    assert pd.api.types.is_datetime64_any_dtype(out["DateTime"])