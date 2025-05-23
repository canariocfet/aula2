# Pipeline de Machine Learning com DVC – Aula 3 MLOps

Este repositório contém uma pipeline completa de Machine Learning aplicada a dados do setor de óleo e gás. O objetivo é prever uma propriedade física em dados sísmicos a partir de um modelo de simulação, substituindo uma solução anterior baseada em software comercial (caixa-preta).

A pipeline é modular, rastreável e reproduzível, construída com **DVC (Data Version Control)** e **Git** para controle de código e dados.

---

## Estrutura da Pipeline

A pipeline contém 4 estágios principais, definidos no `dvc.yaml`:

```text
cleaning ──▶ training ──▶ inference ──▶ evaluation
cleaning: trata os dados de simulação e sísmica (removendo valores inválidos, ajustando profundidades)
training: treina um modelo ExtraTreesRegressor com os dados limpos e salva o modelo
inference: aplica o modelo treinado aos dados sísmicos para estimar a propriedade
evaluation: compara o resultado da inferência com o Ground Truth fornecido pela empresa (modelo comercial)

📁 Estrutura de Pastas
text
Copiar
Editar
Codigos_Dados/
├── data/                     # Dados de entrada (.npy)
├── outputs/                  # Saídas da pipeline (geradas via DVC)
├── src/                      # Scripts modulares
│   ├── cleaning.py
│   ├── training.py
│   ├── inference.py
│   └── evaluation.py
├── aula2_modulo_mlops.py     # Módulo com funções reutilizáveis
├── dvc.yaml                  # Definição dos estágios da pipeline
├── dvc.lock                  # Snapshot das versões usadas
├── .dvcignore                # Arquivos ignorados pelo DVC
└── .gitignore                # Arquivos ignorados pelo Git

Como rodar a pipeline
Requisitos: Python, Git, DVC e dependências do projeto instaladas
1. Clone o repositório
bash
Copiar
Editar
git clone https://github.com/canariocfet/aula2.git
cd aula2
2. Instale as dependências
bash
Copiar
Editar
pip install -r requirements.txt
3. Restaure os dados (se estiver usando armazenamento remoto)
bash
Copiar
Editar
dvc pull
4. Execute a pipeline completa
bash
Copiar
Editar
dvc repro

Visualização da DAG
Você pode visualizar o fluxo da pipeline com:

bash
Copiar
Editar
dvc dag
Resultado esperado:

nginx
Copiar
Editar
cleaning
   │
training
   │
inference
   │
evaluation

Comandos úteis
dvc add data/<arquivo> – adiciona dados ao controle de versão
dvc repro – executa os estágios da pipeline de acordo com as dependências
dvc push – envia os outputs para o armazenamento remoto
dvc pull – baixa os dados do armazenamento remoto
dvc exp run – executa experimentos variando hiperparâmetros

Observação
O projeto foi desenvolvido como prática de MLOps dentro de uma disciplina aplicada, focando em:

Modularização de código
Versionamento de dados
Reprodutibilidade via DVC
Automação com pipelines

👨‍💻 Autor
Turma MLOps

📜 Licença
MIT
