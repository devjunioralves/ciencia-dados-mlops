"""Ingestão e preparação dos dados ("impacta base").

Etapas:
  1. Carrega o dataset Wine do scikit-learn (ou um CSV existente).
  2. Limpa: remove duplicatas e valores ausentes.
  3. Faz split treino/validação/teste estratificado.
  4. Ajusta o scaler nos dados de treino e o persiste para o serving.

Rode diretamente para (re)gerar o CSV bruto:
    python -m src.data_prep
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .utils import get_logger, load_config, resolve

log = get_logger("data_prep")


def load_raw(cfg: dict) -> pd.DataFrame:
    """Carrega os dados brutos. Usa CSV em cache se já existir, senão o sklearn."""
    raw_path = resolve(cfg["data"]["raw_path"])
    if raw_path.exists():
        log.info("Lendo dados em cache: %s", raw_path)
        return pd.read_csv(raw_path)

    log.info("Gerando dataset Wine a partir do scikit-learn")
    bunch = load_wine(as_frame=True)
    df = bunch.frame.rename(columns={"target": cfg["data"]["target_column"]})
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    log.info("Salvo dataset bruto (%d linhas) em %s", len(df), raw_path)
    return df


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Limpeza básica: duplicatas e NA."""
    n0 = len(df)
    if cfg["preprocessing"]["drop_duplicates"]:
        df = df.drop_duplicates()
    if cfg["preprocessing"]["drop_na"]:
        df = df.dropna()
    log.info("Limpeza: %d -> %d linhas", n0, len(df))
    return df.reset_index(drop=True)


def _make_scaler(kind: str):
    return {"standard": StandardScaler(), "minmax": MinMaxScaler()}.get(kind)


def prepare(cfg: dict | None = None):
    """Retorna splits escalonados + nomes de features/classes e persiste o scaler.

    Returns:
        dict com X_train/X_val/X_test (np.ndarray), y_* (np.ndarray),
        feature_names (list[str]) e classes (list[int]).
    """
    cfg = cfg or load_config()
    target = cfg["data"]["target_column"]

    df = clean(load_raw(cfg), cfg)
    feature_names = [c for c in df.columns if c != target]
    X = df[feature_names].values
    y = df[target].values

    rs = cfg["data"]["random_state"]
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]

    # 1º split: separa teste
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y
    )
    # 2º split: separa validação do restante
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=rs, stratify=y_tmp
    )

    scaler = _make_scaler(cfg["preprocessing"]["scaler"])
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        scaler_path = resolve(cfg["training"]["models_dir"]) / "scaler.joblib"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        log.info("Scaler '%s' salvo em %s", cfg["preprocessing"]["scaler"], scaler_path)

    log.info(
        "Splits -> treino=%d val=%d teste=%d | %d features | %d classes",
        len(X_train), len(X_val), len(X_test), len(feature_names), len(set(y)),
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names,
        "classes": sorted(set(int(v) for v in y)),
    }


if __name__ == "__main__":
    prepare()
