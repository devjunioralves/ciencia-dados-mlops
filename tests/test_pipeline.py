"""Testes de fumaça do pipeline ponta a ponta."""
from __future__ import annotations

import numpy as np

from src.data_prep import prepare
from src.train import compute_metrics, train
from src import predict as predict_mod
from src.utils import load_config


def test_prepare_shapes():
    data = prepare()
    n_feat = len(data["feature_names"])
    assert data["X_train"].shape[1] == n_feat
    assert len(data["classes"]) >= 2
    # splits não-vazios
    assert len(data["X_train"]) > 0
    assert len(data["X_test"]) > 0


def test_compute_metrics_perfect():
    y = np.array([0, 1, 2, 1, 0])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0


def test_train_and_predict_end_to_end():
    meta = train()
    # O vencedor deve ter desempenho razoável no dataset Wine.
    assert meta["test_metrics"]["accuracy"] > 0.8
    assert meta["framework"] in {"sklearn", "tensorflow"}

    predict_mod.reload()
    info = predict_mod.model_info()
    sample = [0.0] * info["n_features"]
    result = predict_mod.predict(sample)
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] in info["classes"]


def test_predict_wrong_dimension():
    predict_mod.reload()
    import pytest

    with pytest.raises(ValueError):
        predict_mod.predict([1.0, 2.0])  # menos features que o esperado
