"""Serviço REST (FastAPI) que expõe o modelo treinado.

Endpoints:
  GET  /health   -> status do serviço
  GET  /info     -> metadados do modelo ativo (nome, métricas, features)
  POST /predict  -> recebe novos dados e retorna a predição
  POST /reload   -> recarrega o modelo do disco (após retreino)

Subir:
    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import ModelNotTrainedError, model_info, predict, reload

app = FastAPI(
    title="Wine Classifier — MLOps API",
    version="1.0.0",
    description="Serviço de inferência do classificador de vinhos.",
)


class PredictRequest(BaseModel):
    # Uma amostra (lista de floats) ou várias (lista de listas).
    features: list = Field(
        ...,
        description="13 features na ordem de /info. Ex.: [13.2, 1.78, ...] ou [[...],[...]]",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    try:
        return model_info()
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict")
def do_predict(req: PredictRequest):
    try:
        return predict(req.features)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/reload")
def do_reload():
    reload()
    return {"status": "reloaded", **model_info()}
