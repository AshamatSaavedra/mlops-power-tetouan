import numpy as np
from mlops_power_tetouan.modeling.modeling import ZoneTrainer 

def test_evaluate_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    mae, rmse, r2 = ZoneTrainer.evaluate(y_true, y_pred)
    assert mae >= 0
    assert rmse >= 0
