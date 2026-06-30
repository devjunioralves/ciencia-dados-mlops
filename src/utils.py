"""Utilidades compartilhadas: leitura de config e logging."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

# Raiz do projeto (este arquivo está em src/)
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Carrega um arquivo YAML de configuração."""
    full = ROOT / path
    with open(full, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_logger(name: str = "mlops") -> logging.Logger:
    """Logger simples e consistente em todo o pipeline."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def resolve(path: str | Path) -> Path:
    """Resolve um caminho relativo à raiz do projeto."""
    return ROOT / path
