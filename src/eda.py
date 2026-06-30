"""Análise exploratória e multivariada.

Gera, em reports/:
  - estatísticas descritivas (describe.csv)
  - matriz de correlação (heatmap) + alerta de multicolinearidade
  - distribuição das classes
  - pairplot de um subconjunto de features

Rode:
    python -m src.eda
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sem display (CI-friendly)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_prep import clean, load_raw
from .utils import get_logger, load_config, resolve

log = get_logger("eda")


def run_eda(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    target = cfg["data"]["target_column"]
    reports = resolve(cfg["eda"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    df = clean(load_raw(cfg), cfg)
    features = [c for c in df.columns if c != target]

    # 1. Estatísticas descritivas
    desc = df[features].describe().T
    desc.to_csv(reports / "describe.csv")

    # 2. Distribuição das classes
    plt.figure(figsize=(6, 4))
    df[target].value_counts().sort_index().plot(kind="bar", color="#4C72B0")
    plt.title("Distribuição das classes")
    plt.xlabel("classe")
    plt.ylabel("contagem")
    plt.tight_layout()
    plt.savefig(reports / "class_distribution.png", dpi=110)
    plt.close()

    # 3. Correlação (análise multivariada) + alerta de multicolinearidade
    corr = df[features].corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
    plt.title("Matriz de correlação das features")
    plt.tight_layout()
    plt.savefig(reports / "correlation_heatmap.png", dpi=110)
    plt.close()

    thr = cfg["eda"]["correlation_threshold"]
    high_corr = [
        (features[i], features[j], round(float(corr.iloc[i, j]), 3))
        for i in range(len(features))
        for j in range(i + 1, len(features))
        if abs(corr.iloc[i, j]) >= thr
    ]
    if high_corr:
        log.warning("Pares com |corr| >= %.2f (multicolinearidade): %s", thr, high_corr)
    pd.DataFrame(high_corr, columns=["feature_a", "feature_b", "corr"]).to_csv(
        reports / "high_correlation_pairs.csv", index=False
    )

    # 4. Pairplot (subconjunto para não explodir o tempo)
    subset = features[:4] + [target]
    try:
        g = sns.pairplot(df[subset], hue=target, corner=True)
        g.fig.suptitle("Pairplot (subconjunto)", y=1.02)
        g.savefig(reports / "pairplot.png", dpi=100)
        plt.close("all")
    except Exception as exc:  # pragma: no cover - apenas visual
        log.warning("Pairplot ignorado: %s", exc)

    log.info("Relatórios de EDA salvos em %s", reports)
    return {"n_high_corr_pairs": len(high_corr), "reports_dir": str(reports)}


if __name__ == "__main__":
    run_eda()
