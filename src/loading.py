#!/usr/bin/env python
# coding: utf-8

# ## Desafio Aula 3:

# ### 1. Modularização para organizar no DVC
# ### 2. Uso do AutoML para automatizar o emprego dos modelos de Machine Learning utilizando  - Pycaret
# ### 3. Criação de uma Pipeline de ML no DVC

# ## Conceitos-Chave
# 
# ### Reprodutibilidade: garantir que os experimentos sejam executáveis a qualquer momento, com os mesmos dados e parâmetros.
# 
# ### Versionamento de dados: aplicar o mesmo controle de versão que temos no código (Git) também aos dados.
# 
# ### Pipelines: criação de etapas encadeadas com dependências rastreáveis.

# ## Desafio Aula 2 (última aula):

# ### Atuando como consultor para uma empresa, a mesma lhe forneceu um código legado de um projeto que não foi para frente com o time de analytics deles.
# ### A empresa é da área de óleo e gás e trabalha mapeando áreas com potencial para explorar.
# ### O projeto deles trata de tentar aumentar a granularidade(resolução) de um conjunto de dados inicial para um conjunto de dados final com "melhor resolução" que permita um mapeamento melhor.
# ### A empresa trabalha com um software comercial que produz resultados razoáveis, mas que é uma caixa preta e o time de negócios da empresa agora resolveu criar suas próprias soluções para ter mais controle e não precisar pagar mais a licença desse software e automatizar os processos.
# ### A empresa lhe forneceu os dados de treino, e os dados de inferência. Ambos em estrutura numpy array com coordenadas X,Y,Z,Propriedade(target).
# ### A empresa também lhe forneceu os dados do resultado gerado por eles com o software comercial, com o mesmo tipo de estrutura dos dados de treino e de inferencia, para que você compare com a solução criada por você.
# ### Cabe a você realizar experimentos novos que melhorem (em relação à solução do software comercial).
# ### Repare que a solução atual que já consta no código legado claramente apresenta artefatos estranhos, explore isso.

# ### Modulo 1. loading.py ➞ carrega os dados

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
<<<<<<< HEAD
import aula2_modulo_mlops as sts
=======
import aula3_modulo_mlops as sts
>>>>>>> 2950874 (Versão com AutoML PyCaret + DVC + MLflow funcionando)
import mlflow

#Dados Treino
sim_slice = np.load("sim_slice.npy")

#Dados para Inferência
seismic_slice = np.load("seismic_slice.npy")

#Dados de Referência para a modelagem(Software Comnercial)
seismic_slice_GT = np.load("seismic_slice_GT.npy")

print("Loading Pronto")

### FIM
