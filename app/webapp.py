"""Aplicação Web (Streamlit): submeter novos dados ao modelo.

Permite preencher as 13 features de um vinho e obter a classe predita,
seja chamando a API REST, seja usando o modelo localmente.

Subir:
    streamlit run app/webapp.py
"""
from __future__ import annotations

import requests
import streamlit as st

from src.predict import ModelNotTrainedError, model_info, predict
from src.utils import load_config

st.set_page_config(page_title="Wine Classifier — MLOps", page_icon="🍷")
st.title("🍷 Classificador de Vinhos — MLOps")

cfg = load_config()
api_url = cfg["serving"]["api_url"]

# ----------------------------------------------------------------------- #
# Info do modelo
# ----------------------------------------------------------------------- #
try:
    info = model_info()
except ModelNotTrainedError:
    st.error("Nenhum modelo treinado. Rode `python -m src.train` primeiro.")
    st.stop()

with st.expander("ℹ️ Modelo ativo", expanded=True):
    c1, c2 = st.columns(2)
    c1.metric("Modelo", info["model_name"])
    c2.metric("Acurácia (teste)", f"{info['test_metrics']['accuracy']:.3f}")
    st.caption(f"Treinado em: {info['trained_at']}")

# ----------------------------------------------------------------------- #
# Formulário de submissão de novos dados
# ----------------------------------------------------------------------- #
st.subheader("Submeter novo vinho")
mode = st.radio("Origem da predição", ["Via API REST", "Modelo local"], horizontal=True)

features = info["feature_names"]
with st.form("predict_form"):
    cols = st.columns(3)
    values = []
    for i, feat in enumerate(features):
        with cols[i % 3]:
            values.append(st.number_input(feat, value=0.0, format="%.4f"))
    submitted = st.form_submit_button("Classificar")

if submitted:
    try:
        if mode == "Via API REST":
            resp = requests.post(f"{api_url}/predict", json={"features": values}, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        else:
            result = predict(values)

        pred = result["predictions"][0]
        st.success(f"Classe predita: **{pred}**")

        proba = result["probabilities"][0]
        st.bar_chart(
            {str(c): [p] for c, p in zip(result["classes"], proba)}
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"Falha ao chamar a API ({api_url}): {exc}")
    except Exception as exc:
        st.error(f"Erro na predição: {exc}")
