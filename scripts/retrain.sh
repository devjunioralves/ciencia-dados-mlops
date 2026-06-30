#!/usr/bin/env bash
# =============================================================
# Ciclo de retreino MLOps (local ou em CI).
#   1. Executa EDA
#   2. Retreina os 3 modelos e escolhe o melhor automaticamente
#   3. Se houver mudança nos artefatos, comita e (opcional) faz push
#
# Uso:
#   ./scripts/retrain.sh           # retreina e comita localmente
#   PUSH=1 ./scripts/retrain.sh    # também faz push para o remoto
# =============================================================
set -euo pipefail

cd "$(dirname "$0")/.."

echo ">> [1/3] Análise exploratória (EDA)"
python -m src.eda

echo ">> [2/3] Retreino + seleção automática do melhor modelo"
python -m src.train

echo ">> [3/3] Versionando artefatos no git"
git add models/ reports/ data/ || true

if git diff --cached --quiet; then
  echo "Nenhuma mudança nos artefatos — nada a comitar."
  exit 0
fi

STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git commit -m "chore(mlops): retreino automático do modelo (${STAMP})"
echo "Commit de retreino criado."

if [[ "${PUSH:-0}" == "1" ]]; then
  echo ">> Enviando para o servidor (git push)"
  git push origin HEAD
fi
