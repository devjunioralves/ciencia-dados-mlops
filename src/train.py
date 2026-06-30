"""Treino dos 3 modelos candidatos, avaliação e escolha AUTOMÁTICA do melhor.

Fluxo:
  1. prepare() -> splits escalonados
  2. para cada modelo em config/models.yaml: treina e avalia na validação
  3. seleciona o melhor segundo training.selection_metric
  4. reavalia o vencedor no conjunto de teste
  5. salva modelo + scaler + metadata.json e registra histórico

Rode:
    python -m src.train
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .data_prep import prepare
from .utils import get_logger, load_config, resolve

log = get_logger("train")

# TensorFlow é opcional: o pipeline ainda roda com os modelos sklearn se faltar.
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except Exception:  # pragma: no cover
    TF_AVAILABLE = False
    log.warning("TensorFlow indisponível — modelo keras_mlp será ignorado.")


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
    }


# --------------------------------------------------------------------------- #
# Construção/treino de cada modelo
# --------------------------------------------------------------------------- #
def _train_sklearn(spec, data):
    estimators = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
    }
    cls = estimators[spec["estimator"]]
    model = cls(**spec.get("params", {}))
    model.fit(data["X_train"], data["y_train"])
    y_pred = model.predict(data["X_val"])
    return model, y_pred


def _build_keras(spec, n_features, n_classes):
    p = spec["params"]
    layers = [tf.keras.layers.Input(shape=(n_features,))]
    for units in p["hidden_units"]:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
        if p.get("dropout"):
            layers.append(tf.keras.layers.Dropout(p["dropout"]))
    layers.append(tf.keras.layers.Dense(n_classes, activation="softmax"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _train_keras(spec, data):
    p = spec["params"]
    n_classes = len(data["classes"])
    model = _build_keras(spec, data["X_train"].shape[1], n_classes)
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=p["patience"], restore_best_weights=True
    )
    model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=p["epochs"], batch_size=p["batch_size"],
        callbacks=[es], verbose=0,
    )
    y_pred = model.predict(data["X_val"], verbose=0).argmax(axis=1)
    return model, y_pred


# --------------------------------------------------------------------------- #
# Persistência do vencedor
# --------------------------------------------------------------------------- #
def _save_winner(framework, model, cfg, metadata):
    models_dir = resolve(cfg["training"]["models_dir"])
    base = cfg["training"]["best_model_name"]

    # Remove artefatos antigos do "melhor modelo"
    for old in [f"{base}.joblib", f"{base}.keras"]:
        p = models_dir / old
        if p.exists():
            p.unlink()

    if framework == "tensorflow":
        path = models_dir / f"{base}.keras"
        model.save(path)
    else:
        path = models_dir / f"{base}.joblib"
        joblib.dump(model, path)

    metadata["artifact"] = path.name
    with open(resolve(cfg["training"]["metadata_file"]), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    log.info("Modelo vencedor salvo em %s", path)
    return path


def _append_history(cfg, rows):
    hist_path = resolve(cfg["training"]["metrics_history"])
    df = pd.DataFrame(rows)
    if hist_path.exists():
        df = pd.concat([pd.read_csv(hist_path), df], ignore_index=True)
    df.to_csv(hist_path, index=False)


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def train(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    models_cfg = load_config("config/models.yaml")["models"]
    metric = cfg["training"]["selection_metric"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    data = prepare(cfg)

    results = []
    for spec in models_cfg:
        fw = spec["framework"]
        if fw == "tensorflow" and not TF_AVAILABLE:
            continue
        log.info("Treinando '%s' (%s)...", spec["name"], fw)
        if fw == "tensorflow":
            model, y_val_pred = _train_keras(spec, data)
        else:
            model, y_val_pred = _train_sklearn(spec, data)

        m = compute_metrics(data["y_val"], y_val_pred)
        log.info("  '%s' val %s=%.4f", spec["name"], metric, m[metric])
        results.append({"name": spec["name"], "framework": fw, "model": model, "val_metrics": m})

    if not results:
        raise RuntimeError("Nenhum modelo treinado (verifique frameworks/instalação).")

    # Escolha AUTOMÁTICA pelo melhor valor da métrica na validação
    winner = max(results, key=lambda r: r["val_metrics"][metric])
    log.info("Vencedor: '%s' (%s=%.4f)", winner["name"], metric, winner["val_metrics"][metric])

    # Avaliação final no conjunto de TESTE
    if winner["framework"] == "tensorflow":
        y_test_pred = winner["model"].predict(data["X_test"], verbose=0).argmax(axis=1)
    else:
        y_test_pred = winner["model"].predict(data["X_test"])
    test_metrics = compute_metrics(data["y_test"], y_test_pred)
    log.info("Teste do vencedor: %s", {k: round(v, 4) for k, v in test_metrics.items()})

    metadata = {
        "run_id": run_id,
        "model_name": winner["name"],
        "framework": winner["framework"],
        "selection_metric": metric,
        "val_metrics": winner["val_metrics"],
        "test_metrics": test_metrics,
        "feature_names": data["feature_names"],
        "classes": data["classes"],
        "n_features": len(data["feature_names"]),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_winner(winner["framework"], winner["model"], cfg, metadata)

    _append_history(cfg, [
        {"run_id": run_id, "model": r["name"], "selected": r["name"] == winner["name"],
         **{f"val_{k}": v for k, v in r["val_metrics"].items()}}
        for r in results
    ])
    return metadata


if __name__ == "__main__":
    train()
