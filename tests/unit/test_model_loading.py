import pickle
import numpy as np
import pytest
import os

MODEL_PATH = "models/zone1_model.pkl"

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not trained yet")
def test_model_loading_and_predict():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X_dummy = np.random.rand(1, 14)
    pred = model.predict(X_dummy)

    assert pred is not None
    assert len(pred) == 1
    assert np.isfinite(pred[0])
