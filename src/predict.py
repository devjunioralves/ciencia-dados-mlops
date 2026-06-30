"""Carrega o melhor modelo + scaler e faz predições.

Usado pela API e pelo app web. Mantém um cache em memória para não
recarregar o artefato a cada requisição.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np

from .utils import get_logger, load_config, resolve

log = get_logger("predict")


class ModelNotTrainedError(RuntimeError):
    """Levantada quando nenhum artefato treinado foi encontrado."""


@lru_cache(maxsize=1)
def _load_bundle():
    cfg = load_config()
    meta_path = resolve(cfg["training"]["metadata_file"])
    if not meta_path.exists():
        raise ModelNotTrainedError(
            "Nenhum modelo treinado. Rode `python -m src.train` primeiro."
        )
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    models_dir = resolve(cfg["training"]["models_dir"])
    artifact = models_dir / meta["artifact"]

    if meta["framework"] == "tensorflow":
        import tensorflow as tf

        model = tf.keras.models.load_model(artifact)
        predict_fn = lambda X: model.predict(X, verbose=0)
    else:
        model = joblib.load(artifact)
        predict_fn = lambda X: model.predict_proba(X)

    scaler_path = models_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return meta, predict_fn, scaler


def reload():
    """Limpa o cache (chamar após um retreino)."""
    _load_bundle.cache_clear()


def model_info() -> dict:
    meta, _, _ = _load_bundle()
    return {
        "model_name": meta["model_name"],
        "framework": meta["framework"],
        "trained_at": meta["trained_at"],
        "test_metrics": meta["test_metrics"],
        "feature_names": meta["feature_names"],
        "n_features": meta["n_features"],
        "classes": meta["classes"],
    }


def predict(features: list[float] | list[list[float]]) -> dict:
    """Prediz a partir de uma amostra (lista) ou várias (lista de listas).

    Retorna classe predita, probabilidades e nomes das features esperadas.
    """
    meta, predict_fn, scaler = _load_bundle()
    X = np.atleast_2d(np.asarray(features, dtype=float))

    expected = meta["n_features"]
    if X.shape[1] != expected:
        raise ValueError(
            f"Esperado {expected} features, recebido {X.shape[1]}."
        )
    if scaler is not None:
        X = scaler.transform(X)

    proba = np.asarray(predict_fn(X))
    pred_idx = proba.argmax(axis=1)
    classes = meta["classes"]
    return {
        "model_name": meta["model_name"],
        "predictions": [classes[i] for i in pred_idx],
        "probabilities": proba.round(4).tolist(),
        "classes": classes,
    }


if __name__ == "__main__":
    info = model_info()
    log.info("Modelo ativo: %s", info["model_name"])
