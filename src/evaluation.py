### Modulo 5. evaluation.py ➞ compara com seismic_slice_GT

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from modulo import aula2_modulo_mlops as sts
from modulo import aula3_modulo_mlops as sts
import mlflow

#Dados Treino
sim_clean = np.load("outputs/sim_clean.npy")

#Dados da inferencia
seis_estimated = np.load("outputs/seis_estimated.npy")

#Dados de Referência para a modelagem(Software Comnercial)
seismic_slice_GT = np.load("data/seismic_slice_GT.npy")

print("Loading Pronto")

### Calculo dos Residuos: Dados de Referencia - Dados da Inferência ML
seismic_slice_residual_final = sts.residual_calculator(seismic_GT = seismic_slice_GT, seismic_original = seis_estimated)

### Plot dos histogramas com as distribuições dos dados originais, dados de inferencia e dados de referencia(Ground Truth)
sts.plot_simulation_distribution(sim_clean, bins=35)

sts.plot_simulation_distribution(seis_estimated, bins=35)

sts.plot_simulation_distribution(seismic_slice_GT, bins=35)

sts.plot_simulation_distribution(seismic_slice_residual_final, bins=35)

### Plot das imagens dos dados originais, dados de inferencia e dados de referencia(Ground Truth)
sts.plot_seismic_slice(sim_clean, title="Slice a profundidade ~5000m dos dados de treino")

sts.plot_seismic_slice(seismic_slice_GT, title="Slice a profundidade ~5000m do Resultado-Referência(software comercial)")

sts.plot_seismic_slice(seis_estimated, title="Slice a profundidade ~5000m da Inferência ML")

sts.plot_seismic_slice(seismic_slice_residual_final, title = "Slice a profundidade ~5000m - Residuo da Inferência")

np.save("outputs/residuos.npy", seismic_slice_residual_final)

# Log no MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("projeto_aula4_mlops")

with mlflow.start_run(run_name="evaluation"):
    mlflow.log_artifact("outputs/residuos.npy")    
    print("Avaliação registrada no MLflow")

print("Plotting Pronto")

### FIM
