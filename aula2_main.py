#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import simtoseis_library as sts


# In[2]:


def carregar_dados():
    sim_slice = np.load("sim_slice.npy")
    seismic_slice = np.load("seismic_slice.npy")
    seismic_slice_GT = np.load("seismic_slice_GT.npy")
    return sim_slice, seismic_slice, seismic_slice_GT


def tratar_dados(sim_slice, seismic_slice):
    sim_data = sts.simulation_data_cleaning(simulation_data=sim_slice, value_to_clean=-99.0)
    sim_data = sts.simulation_nan_treatment(simulation=sim_data, value=0, method='replace')
    sim_data, _ = sts.depth_signal_checking(simulation_data=sim_data, seismic_data=seismic_slice)
    return sim_data


def plotar_histograma(array, titulo="Distribuição", bins=35):
    plt.hist(array[:, -1], bins=bins)
    plt.title(titulo)
    plt.xlabel("Valor da Propriedade")
    plt.ylabel("Frequência")
    plt.grid(alpha=0.3)
    plt.show()

def treinar_modelo(sim_data, proporcao_treino=0.75):
    X = sim_data[:, :-1]
    y = sim_data[:, -1]

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, train_size=proporcao_treino, random_state=0)

    modelo = ExtraTreesRegressor(max_depth=20, n_jobs=-1)
    modelo.fit(X_treino, y_treino)

    pred_teste = modelo.predict(X_teste)

    nrms = (np.sqrt(mean_squared_error(pred_teste, y_teste)) / np.std(y_teste)) * 100
    r2 = r2_score(y_teste, pred_teste)
    mape = np.mean(np.abs((pred_teste - y_teste) / (y_teste + 0.01)))

    print("Desempenho no conjunto de teste")
    print(f"NRMS: {nrms:.2f}%, R2: {r2:.2f}, MAPE: {mape:.2f}")

    return modelo, X, y


def inferir_propriedade(modelo, seismic_slice):
    coords = seismic_slice[:, :3].copy()
    prop_predita = modelo.predict(coords)
    prop_predita = prop_predita.reshape(-1, 1)
    seis_estimado = np.hstack((coords, prop_predita))
    return seis_estimado


def calcular_residuos(seismic_slice_GT, seis_estimado):
    residuos = seismic_slice_GT[:, -1] - seis_estimado[:, -1]
    residuos = residuos.reshape(-1, 1)
    return np.hstack((seis_estimado[:, :3], residuos))


def main(proporcao_treino=0.75):
    sim_slice, seismic_slice, seismic_slice_GT = carregar_dados()
    sim_data = tratar_dados(sim_slice, seismic_slice)

    modelo, X, y = treinar_modelo(sim_data, proporcao_treino)

    seis_estimado = inferir_propriedade(modelo, seismic_slice)
    np.save("slice_seis_estimado.npy", seis_estimado)

    residuos = calcular_residuos(seismic_slice_GT, seis_estimado)
    np.save("slice_residuos.npy", residuos)

    plotar_histograma(seis_estimado, titulo="Inferência Sísmica - Histograma")
    plotar_histograma(residuos, titulo="Resíduos - Histograma")


if __name__ == "__main__":
    main()


# In[ ]:




