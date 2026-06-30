# 🍷 Produtização de ML (MLOps) — Classificador de Vinhos

Exercício completo de **produtização de um modelo de classificação** seguindo
boas práticas de **MLOps**: da análise dos dados ao serviço em produção com
**ciclo automatizado de retreino** via GitHub Actions.

Dataset: **Wine** (scikit-learn) — 178 amostras, **13 features**, **3 classes**.

---

## 🎯 O que este exercício cobre

| Requisito | Onde está |
|-----------|-----------|
| Impacta/prepara a base | [src/data_prep.py](src/data_prep.py) |
| Análise multivariada (correlação, pairplot) | [src/eda.py](src/eda.py) |
| Desenvolvimento de **3 modelos** | [config/models.yaml](config/models.yaml) + [src/train.py](src/train.py) |
| Avaliação e **escolha automática** por métrica | [src/train.py](src/train.py) |
| Salvar o modelo vencedor | `models/` (artefato + `metadata.json`) |
| **TensorFlow / Keras** | modelo `keras_mlp` em [src/train.py](src/train.py) |
| **Arquivos YAML** | [config/](config/) |
| **Git** | versiona código **e** artefatos do modelo |
| **Serviço** (API REST) | [app/api.py](app/api.py) (FastAPI) |
| **App Web** (submeter novos dados) | [app/webapp.py](app/webapp.py) (Streamlit) |
| **Ciclo MLOps** (retreino + atualização no servidor) | [scripts/retrain.sh](scripts/retrain.sh) + [.github/workflows/retrain.yml](.github/workflows/retrain.yml) |

---

## 📁 Estrutura

```
ciencia-dados-mlops/
├── config/                 # Configuração em YAML (sem hard-code)
│   ├── config.yaml         #   pipeline, métrica de seleção, serving
│   └── models.yaml         #   os 3 modelos candidatos e hiperparâmetros
├── src/
│   ├── data_prep.py        # ingestão + limpeza + split + scaler
│   ├── eda.py              # análise multivariada -> reports/
│   ├── train.py            # treina 3 modelos, escolhe o melhor, salva
│   ├── predict.py          # carrega o modelo e prediz (usado no serving)
│   └── utils.py
├── app/
│   ├── api.py              # serviço FastAPI (/predict, /info, /reload)
│   └── webapp.py           # app web Streamlit (formulário de submissão)
├── scripts/retrain.sh      # ciclo de retreino + commit/push
├── .github/workflows/
│   ├── ci.yml              # CI: treina + testa a cada push/PR
│   └── retrain.yml         # retreino agendado + push do novo modelo
├── tests/                  # testes de fumaça ponta a ponta
├── models/                 # artefatos versionados (saída do treino)
├── reports/                # saídas da EDA (saída do treino)
├── requirements.txt
└── Makefile
```

---

## 🚀 Como rodar

### 1. Instalar dependências
```bash
make install        # ou: pip install -r requirements.txt
```

### 2. Análise exploratória (multivariada)
```bash
make eda            # gera gráficos e CSVs em reports/
```

### 3. Treinar (desenvolve os 3 modelos e escolhe o melhor)
```bash
make train
```
O treino avalia `logistic_regression`, `random_forest` e `keras_mlp`,
compara pela métrica `training.selection_metric` (padrão `f1_macro`) e
salva **automaticamente** o vencedor em `models/` junto com `metadata.json`.

### 4. Subir o serviço (API REST)
```bash
make api            # http://localhost:8000/docs
```
Exemplo de requisição:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [13.2,1.78,2.14,11.2,100,2.65,2.76,0.26,1.28,4.38,1.05,3.4,1050]}'
```

### 5. Subir o App Web (submeter novos dados)
```bash
make web            # http://localhost:8501
```

### 6. Testes
```bash
make test
```

---

## 🔁 Ciclo automatizado de MLOps

O retreino é o coração do MLOps aqui:

```bash
make retrain                 # retreina e comita os artefatos localmente
PUSH=1 ./scripts/retrain.sh  # também faz push para o servidor (repo remoto)
```

Em produção, o workflow [retrain.yml](.github/workflows/retrain.yml):
1. roda **agendado** (cron semanal) ou sob demanda (`workflow_dispatch`);
2. retreina os 3 modelos e escolhe o melhor pela métrica;
3. **versiona os novos artefatos** (`models/`, `reports/`);
4. faz **commit e push** de volta ao repositório — atualizando o "servidor".

Após um retreino com o serviço no ar, recarregue o modelo sem reiniciar:
```bash
curl -X POST http://localhost:8000/reload
```

---

## 🧩 Como estender (sugestões de exercício)

- Trocar a métrica de seleção em `config/config.yaml`.
- Adicionar um 4º modelo em `config/models.yaml`.
- Adicionar um *gate* de qualidade: só promover o modelo se superar o anterior
  (comparar com `models/metadata.json` antes de salvar).
- Conteinerizar com Docker e publicar a imagem no pipeline.
