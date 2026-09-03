 Home Credit — Previsão de Risco de Inadimplência

Modelo de credit scoring que estima a probabilidade de um cliente não honrar um empréstimo, usando o dataset [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).

O projeto cobre o ciclo completo: integração de 7 tabelas relacionais, engenharia de features, treino com validação cruzada e avaliação.

---

## O problema

A Home Credit concede crédito a pessoas com histórico bancário limitado ou inexistente. Aprovar um cliente que vai inadimplir gera perda direta; recusar um bom pagador gera receita perdida. O objetivo é ordenar os clientes por risco, para que a política de crédito possa definir onde cortar.

A base é desbalanceada: cerca de 8% dos contratos são inadimplentes. Por isso a métrica é **ROC-AUC** (capacidade de ordenação), não acurácia.

## Resultado

| Modelo | Features | ROC-AUC (OOF, 5 folds) |
|---|---|---|
| LightGBM — baseline | <!-- PREENCHER --> | <!-- PREENCHER --> |
| LightGBM — após seleção de features | <!-- PREENCHER --> | <!-- PREENCHER --> |

<!-- PREENCHER: se você submeteu no Kaggle, acrescente o score do leaderboard aqui. Vale mais que o OOF. -->

As features mais preditivas foram as `EXT_SOURCE` (scores externos de crédito) e os indicadores de atraso de pagamento construídos a partir do histórico de parcelas.

<!-- PREENCHER: salve o gráfico de importância em docs/ e descomente a linha abaixo
![Top 20 features](docs/feature_importance.png)
-->

## Como funciona

```
application_train/test  ──┐
bureau + bureau_balance ──┤
previous_application    ──┼──▶  agregação por SK_ID_CURR  ──▶  feature engineering  ──▶  LightGBM
POS_CASH_balance        ──┤                                                                   │
installments_payments   ──┤                                                          StratifiedKFold (5)
credit_card_balance     ──┘                                                                   │
                                                                                          ROC-AUC
```

**1. Integração.** Cada tabela auxiliar tem várias linhas por cliente (um contrato anterior, uma parcela, um mês de fatura). Todas são reduzidas a uma linha por `SK_ID_CURR` com agregações estatísticas, e depois unidas à base principal.

**2. Feature engineering.** Além das agregações, foram construídas features de domínio:

- **Rácios financeiros** — crédito sobre anuidade, renda sobre crédito, valor de entrada.
- **Comportamento recente** — proporção de parcelas pagas em atraso nos últimos 180 dias e média de dias de atraso, que capturam deterioração recente melhor que a média histórica.
- **Scores externos** — média, desvio e mínimo entre as três `EXT_SOURCE`, além da contagem de valores ausentes (a própria ausência do score é informativa).
- **Tratamento de sentinelas** — `DAYS_EMPLOYED = 365243` significa "não empregado" e vira nulo, em vez de ser lido como mil anos de casa.

**3. Modelagem.** LightGBM com `StratifiedKFold` de 5 folds e early stopping. As predições out-of-fold servem para avaliar sem vazamento, e a submissão é a média das predições dos 5 modelos.

## Estrutura

```
├── notebooks/
│   └── 01_eda.ipynb        # EDA, modelagem e avaliação
├── src/projeto/
│   ├── config.py           # caminhos do projeto
│   └── utils.py            # ETL, agregações e feature engineering
├── pyproject.toml
└── requirements.txt
```

## Como reproduzir

```bash
git clone https://github.com/diogo-moitinho/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk
pip install -r requirements.txt
pip install -e .
```

Baixe os dados da [competição no Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) e extraia os CSVs em `data/raw/`. Depois:

```python
from projeto.utils import load_and_merge_home_credit_data

df_train, df_test = load_and_merge_home_credit_data()
```

O pipeline grava as tabelas integradas em `data/bronze/` — execuções seguintes reaproveitam esses arquivos.

## Próximos passos

- Otimização de hiperparâmetros com Optuna
- Métricas de negócio: KS, tabela de decis e análise de lift
- Interpretabilidade com SHAP para explicar decisões individuais
- Comparação com XGBoost e CatBoost

## Stack

Python · pandas · NumPy · LightGBM · scikit-learn · Matplotlib · Seaborn