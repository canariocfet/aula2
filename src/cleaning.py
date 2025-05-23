### Modulo 1. loading.py ➞ carrega os dados

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from modulo import aula2_modulo_mlops as sts
import mlflow

#Dados Treino
sim_slice = np.load("data/sim_slice.npy")

#Dados para Inferência
seismic_slice = np.load("data/seismic_slice.npy")

print("Loading Pronto")

#----------------------------------------------------------------------------------------------------

### Modulo 2. cleaning.py ➞ limpeza do sim_slice

sim_data = sts.simulation_data_cleaning(simulation_data = sim_slice, value_to_clean = -99.0)

sim_data = sts.simulation_nan_treatment(simulation = sim_data, value = 0, method = 'replace')

sim_data, seismic_slice = sts.depth_signal_checking(simulation_data=sim_data, seismic_data=seismic_slice)

np.save("outputs/sim_clean.npy", sim_data)

np.save("outputs/seismic_slice_clean.npy", seismic_slice)

print("Data Cleaning Pronto")

### FIM
