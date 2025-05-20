#!/usr/bin/env python
# coding: utf-8

## Desafio:
### Atuando como consultor para uma empresa, a mesma lhe forneceu um código legado de um projeto que não foi para frente com o time de analytics deles.
### A empresa é da área de óleo e gás e trabalha mapeando áreas com potencial para explorar.
### O projeto deles trata de tentar aumentar a granularidade(resolução) de um conjunto de dados inicial para um conjunto de dados final com "melhor resolução" que permita um mapeamento melhor.
### A empresa trabalha com um software comercial que produz resultados razoáveis, mas que é uma caixa preta e o time de negócios da empresa agora resolveu criar suas próprias soluções para ter mais controle e não precisar pagar mais a licença desse software e automatizar os processos.
### A empresa lhe forneceu os dados de treino, e os dados de inferência. Ambos em estrutura numpy array com coordenadas X,Y,Z,Propriedade(target).
### A empresa também lhe forneceu os dados do resultado gerado por eles com o software comercial, com o mesmo tipo de estrutura dos dados de treino e de inferencia, para que você compare com a solução criada por você.
### Cabe a você realizar experimentos novos que melhorem (em relação à solução do software comercial).
### Repare que a solução atual que já consta no código legado claramente apresenta artefatos estranhos, explore isso.

### Importando Bicliotecas

# In[1]:


import numpy as np
import simtoseis_library as sts
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib.colors import TwoSlopeNorm


# ### Carrega os dados

# In[2]:


#Dados Treino
dados_treino = np.load("sim_slice.npy")
dados_treino


# In[3]:


dados_treino.shape


# In[4]:


#Dados para Inferência
dados_inferencia = np.load("seismic_slice.npy")
dados_inferencia


# In[5]:


dados_inferencia.shape


# In[6]:


#Dados de Referência para a modelagem(Software Comnercial)
dados_referencia_comercial = np.load("seismic_slice_GT.npy")
dados_referencia_comercial


# In[7]:


dados_referencia_comercial.shape


# ### Tratamento dos dados

# In[8]:


#Checando a quantidade original de dados
original_slice_shape = dados_treino.shape[0]
print(f"Original number of samples in simulation model: {original_slice_shape}")


# In[9]:


# Filtrando os dados
filtered_slice = dados_treino[ dados_treino[:, -1] != -99.0 ]
print(f"Final number of samples after cleaning: {filtered_slice.shape[0]}")


# In[10]:


# Calculate and report the percentage of data removed
percentage_loss = ((original_slice_shape - filtered_slice.shape[0]) / original_slice_shape) * 100
print(f"Percentage loss: {round(percentage_loss, 2)}%")


# In[11]:


dados_treino = sts.simulation_data_cleaning(simulation_data = dados_treino, value_to_clean = -99.0)


# In[12]:


dados_treino = sts.simulation_nan_treatment(simulation = dados_treino, value = 0, method = 'replace')


# ### Conversão de sinais

# In[13]:


dados_treino, dados_inferencia = sts.depth_signal_checking(simulation_data=dados_treino, seismic_data=dados_inferencia)


# ### Plotar os dados de treino

# In[14]:


sts.plot_simulation_distribution(dados_treino, bins=35);


# ### Treinamento/Validação do Modelo de ML

# In[15]:


def ML_model_evaluation(dados_simulacao=None, proporcao_treino=0.75, modelo="extratrees"):    
    """
    Treina e avalia um modelo de machine learning ExtraTreesRegressor com dados de simulação.
    Entrada:
        dados_simulacao: numpy array
                         Conjunto de dados com colunas [X, Y, Z, Propriedade], onde a última coluna é a variável alvo.
        proporcao_treino: float (padrão=0.7)
                          Proporção dos dados utilizada para o treinamento (o restante será usado para teste).
    Saída:
        sim_estimado: numpy array
                      Previsões do modelo para o conjunto de teste.
        y: numpy array
           Valores reais da propriedade para o conjunto completo (antes da divisão).
    """

    global X, y, ET, X_treino, X_teste, y_treino, y_teste, sim_estimado, dict_params

    # Separar atributos e variável alvo
    X = dados_simulacao[:,:-1]
    y = dados_simulacao[:,-1]

    # Dividir em conjuntos de treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, train_size=proporcao_treino, random_state=0)
    
    dict_params = {"n_estimators":100,"max_depth":20, "n_jobs":-1,"proporcao_treino":proporcao_treino}
    
    if modelo == "extratrees":    
        # Inicializar e treinar o ExtraTreesRegressor
        ET = ExtraTreesRegressor(n_estimators = dict_params["n_estimators"], max_depth = dict_params["max_depth"], n_jobs = dict_params["n_jobs"])
        ET.fit(X_treino, y_treino)

        # Prever nos conjuntos de treino e teste
        sim_estimado = ET.predict(X_teste)
        sim_treinado = ET.predict(X_treino)

    # Avaliar o desempenho do modelo
    tolerancia = 0.01

    # Métricas de desempenho no treino
    nrms_treino = (np.sqrt(mean_squared_error(sim_treinado, y_treino)) / np.std(y_treino)) * 100
    r2_treino = r2_score(y_treino, sim_treinado)
    mape_treino = np.mean(np.abs((sim_treinado - y_treino) / (y_treino + tolerancia)))

    print("Desempenho no conjunto de treino")
    print(f'Erro percentual absoluto médio: {round(mape_treino, 1)}%')
    print(f'NRMS: {round(nrms_treino, 1)}%')
    print(f'R²: {round(r2_treino, 2)}')

    # Métricas de desempenho no teste
    nrms_teste = (np.sqrt(mean_squared_error(sim_estimado, y_teste)) / np.std(y_teste)) * 100
    r2_teste = r2_score(y_teste, sim_estimado)
    mape_teste = np.mean(np.abs((sim_estimado - y_teste) / (y_teste + tolerancia)))

    print("Desempenho no conjunto de teste")
    print(f'Erro percentual absoluto médio: {round(mape_teste, 1)}%')
    print(f'NRMS: {round(nrms_teste, 1)}%')
    print(f'R²: {round(r2_teste, 2)}')

    print("Concluído!")

    return sim_estimado, y, nrms_teste, r2_teste, mape_teste, modelo


# In[16]:


dados_validacao, y, nrms_teste, r2_teste, mape_teste, modelo = ML_model_evaluation(dados_simulacao=dados_treino, proporcao_treino=0.75, modelo="extratrees")


# ### Inferência do Modelo ML treinado

# In[17]:


def transfer_to_seismic_scale(dados_sismicos=None, nome_arquivo_segy=None):
    """
    Transfere a estimativa da propriedade da escala de simulação para a escala sísmica,
    aplicando um modelo ExtraTreesRegressor previamente treinado.
    
    Entrada:
        dados_sismicos: numpy array
                        Dados sísmicos com colunas [X, Y, Z], onde a propriedade será prevista.
        nome_arquivo_segy: string (opcional)
                           Não é utilizado dentro da função, mantido apenas por compatibilidade.
    
    Saída:
        vetor_prop_sismica: numpy array
                            Vetor com os valores previstos da propriedade para os dados sísmicos.
        sismica_estimada: numpy array
                          Dados sísmicos com colunas [X, Y, Z, Propriedade Prevista].
    """
    
    global vetor_prop_sismica, prop_sismica_reshape, sismica_estimada

    # Reajusta o modelo com todos os dados de simulação
    ET.fit(X, y)

    # Copia as coordenadas sísmicas
    coordenadas_sismicas = dados_sismicos[:, :3].copy()

    # Prediz a propriedade nas coordenadas sísmicas
    vetor_prop_sismica = ET.predict(coordenadas_sismicas)

    # Redimensiona e combina coordenadas com previsões
    prop_sismica_reshape = vetor_prop_sismica.reshape(len(vetor_prop_sismica), 1)
    sismica_estimada = np.hstack((coordenadas_sismicas, prop_sismica_reshape))

    print("Concluído!")
    
    return vetor_prop_sismica, sismica_estimada


# In[18]:


dados_estimados_prop_vector, dados_estimados = transfer_to_seismic_scale(dados_sismicos=dados_inferencia)


# ### Histograma dos dados de inferência

# In[19]:


sts.plot_simulation_distribution(dados_estimados, bins=35)


# ### Calculo dos Residuos: Dados de Referencia(software comercial) - Dados da Inferência ML

# In[20]:


dados_estimados_residual = dados_referencia_comercial[:,-1] - dados_estimados[:,-1]

dados_estimados_residual_reshape = dados_estimados_residual.reshape(len(dados_estimados_residual),1)

dados_estimados_residual_final = np.hstack([dados_estimados[:,:-1], dados_estimados_residual_reshape])


# ### Plotando resultados dos Resíduos

# In[21]:


sts.plot_simulation_distribution(dados_estimados_residual_final, bins=35)


# In[22]:


def plot_seismic_slice(seismic_slice, title="Slice at Depth ~5000m", cmap='seismic'):
    """
    Plots a seismic slice (X, Y, Amplitude/Property) using scatter plot.
    
    Input:
        seismic_slice: numpy array of shape (n_points, 3)
                       Columns = [X, Y, Property/Amplitude]
        title: str
               Plot title.
        cmap: str
              Colormap to use for plotting.
    Output:
        Matplotlib figure.
    """
    vmin = seismic_slice[:,-1].min() #vmin=-38 
    vmax = seismic_slice[:,-1].max() #vmax=51
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
                    seismic_slice[:, 0], #X
                    seismic_slice[:, 1], #Y
                    c=seismic_slice[:, 3], #Property or Amplitude
                    cmap=cmap,
                    norm=norm,
                    s=10,
                    edgecolors='none'
                    )
    
    plt.colorbar(sc, label="Property/Amplitude")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout();


# In[23]:


plot_seismic_slice(dados_treino, title="Slice a ~5000m dos dados de treino")


# In[24]:


plot_seismic_slice(dados_referencia_comercial, title="Slice a ~5000m do Resultado-Referência(software comercial)")


# In[25]:


plot_seismic_slice(dados_estimados, title="Slice a ~5000m da Inferência ML")


# In[26]:


plot_seismic_slice(dados_estimados_residual_final, title = "Slice a ~5000m - Residuo da Inferência")


# ### MLFLOW Tracking

# #No prompt do Anaconda, no Terminal do VSCode, ou Terminal Python, digitar o comando abaixo para pegar a URL que gerencia a conexão com o MLFLOw(vide aula2):
# -> mlflow ui

# ### Criando uma lista com as métricas

# In[27]:


tuple_1 = ["nrms_teste", "r2_teste", "mape_teste"]


# In[28]:


metrics_tuple = [64.2, 0.58, 71.8]


# In[29]:


dict_metrics = dict(zip(tuple_1, metrics_tuple))
dict_metrics


# In[30]:




# ### FIM
