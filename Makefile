# Atalhos do pipeline de MLOps
.PHONY: help install data eda train predict api web test retrain clean

help:
	@echo "Alvos disponíveis:"
	@echo "  install   instala dependências (requirements.txt)"
	@echo "  data      gera/limpa o dataset bruto"
	@echo "  eda       roda a análise exploratória (reports/)"
	@echo "  train     treina os 3 modelos e salva o melhor"
	@echo "  api       sobe o serviço FastAPI (porta 8000)"
	@echo "  web       sobe o app web Streamlit"
	@echo "  test      roda os testes (pytest)"
	@echo "  retrain   executa o ciclo completo de retreino + commit"

install:
	pip install -r requirements.txt

data:
	python -m src.data_prep

eda:
	python -m src.eda

train:
	python -m src.train

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

web:
	streamlit run app/webapp.py

test:
	pytest -q

retrain:
	bash scripts/retrain.sh

clean:
	rm -rf __pycache__ .pytest_cache src/__pycache__ app/__pycache__ tests/__pycache__
